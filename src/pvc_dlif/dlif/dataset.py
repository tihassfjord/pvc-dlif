"""Dataset and cross-validation splitting for DLIF training.

Two properties matter more here than anything else:

* the network must see exactly the tensor layout the group's model expects,
  ``(channels, time, Z, Y, X)``, so a retrained model is comparable with the
  pretrained one;
* the fold assignment must be **identical** across every condition.  The design
  is paired -- each scan contributes one prediction per condition -- and that
  only holds if scan *S* sits in the same test fold whether the inputs are
  uncorrected or PVC-corrected.  :func:`make_folds` therefore depends only on
  the scan IDs and the seed, never on the input data.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..logging_utils import get_logger
from ..data import pkl_io

LOGGER = get_logger(__name__)

__all__ = ["DlifDataset", "Fold", "make_folds", "load_folds", "stratified_val_split", "WeightedMSELoss"]


@dataclass(frozen=True)
class Fold:
    """One cross-validation fold: which scans train and which are held out."""

    index: int
    train_ids: tuple[str, ...]
    test_ids: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {"fold": self.index, "train_ids": list(self.train_ids), "test_ids": list(self.test_ids)}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Fold":
        return cls(index=int(payload["fold"]), train_ids=tuple(payload["train_ids"]),
                   test_ids=tuple(payload["test_ids"]))


def load_folds(path) -> list[Fold]:
    """Read the ``folds.json`` stage 05 writes beside its checkpoints.

    Evaluation must use the partition the models were *trained* with, not a
    recomputed one - the two differ whenever the scan set differed (a pilot
    with --limit, say), and a mismatch would score held-out predictions from
    the wrong model.
    """
    import json
    from pathlib import Path as _Path

    payload = json.loads(_Path(path).read_text(encoding="utf-8"))
    return [Fold.from_dict(row) for row in payload]


def make_folds(scan_ids: Sequence[str], n_folds: int = 10, seed: int = 42) -> list[Fold]:
    """K-fold split over scan IDs, reproducing the group's protocol.

    ``KFold(shuffle=True, random_state=seed)`` over the ID list, exactly as in
    the group's ``MultiRunTrainer``.  The IDs are sorted first so the split does
    not depend on filesystem ordering.
    """
    from sklearn.model_selection import KFold

    ids = sorted(str(s) for s in scan_ids)
    if len(ids) < n_folds:
        raise ValueError(f"cannot make {n_folds} folds from {len(ids)} scans")

    splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds: list[Fold] = []
    for index, (train_idx, test_idx) in enumerate(splitter.split(ids), start=1):
        folds.append(
            Fold(
                index=index,
                train_ids=tuple(ids[i] for i in train_idx),
                test_ids=tuple(ids[i] for i in test_idx),
            )
        )
    return folds


def stratified_val_split(
    train_ids: Sequence[str],
    group_of: Mapping[str, str | None],
    validation_size: float = 0.15,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Split training scans into train/validation, keeping groups represented.

    Mirrors the group's ``stratified_split``: when the validation set is smaller
    than the number of experimental groups, one scan is drawn from each of a
    random subset of groups; otherwise a stratified split is used.
    """
    from sklearn.model_selection import train_test_split

    ids = list(train_ids)
    memberships = [group_of.get(i) or "ungrouped" for i in ids]
    by_group: dict[str, list[str]] = {}
    for scan_id, group in zip(ids, memberships):
        by_group.setdefault(group, []).append(scan_id)

    n_val = int(round(len(ids) * validation_size))
    unique_groups = list(by_group)

    if n_val < len(unique_groups):
        rng = random.Random(seed)
        rng.shuffle(unique_groups)
        val_ids: list[str] = []
        remaining = list(ids)
        for group in unique_groups[:n_val]:
            choice = rng.choice(by_group[group])
            val_ids.append(choice)
            remaining.remove(choice)
        return remaining, val_ids

    # Stratification needs at least two members per class.
    counts = {g: memberships.count(g) for g in set(memberships)}
    stratify = memberships if all(c >= 2 for c in counts.values()) else None
    train_split, val_split = train_test_split(
        ids, test_size=validation_size, stratify=stratify, random_state=seed
    )
    return list(train_split), list(val_split)


class DlifDataset:
    """Loads ``(input, target)`` pairs from one condition's data root.

    The images come from ``data_root`` (a PVC-corrected tree for a corrected
    condition, the original tree for the baseline); the ground-truth arterial
    curves always come from ``aif_root``, because the ground truth is the same
    measurement regardless of how the images were processed.
    """

    def __init__(
        self,
        data_root: Path,
        scan_ids: Sequence[str],
        aif_root: Path | None = None,
        img_shape: Sequence[int] = (96, 48, 48),
        mode: str = "train",
        augment: bool = False,
        poisson_noise: bool = True,
        random_flip: bool = True,
        add_average: bool = False,
        preload: bool = False,
        seed: int = 42,
        cache: dict | None = None,
        augment_on_device: bool = False,
    ):
        self.data_root = Path(data_root)
        self.aif_root = Path(aif_root) if aif_root else self.data_root
        self.scan_ids = [str(s) for s in scan_ids]
        self.img_shape = tuple(int(v) for v in img_shape)
        self.mode = mode
        self.augment = augment and mode == "train"
        self.poisson_noise = poisson_noise
        self.random_flip = random_flip
        self.add_average = add_average
        self._rng = np.random.default_rng(seed)
        # When True the item is returned raw (single channel, no noise, no
        # flip) and the training loop applies the augmentation and the average
        # channel on the GPU - see ``train.augment_on_device``.  Poisson noise
        # over 9 M voxels per scan costs ~0.8 s in NumPy, which for 54 scans is
        # a 40 s epoch with the GPU idle; on the GPU it is milliseconds.
        self.augment_on_device = augment_on_device
        # A cache shared across the datasets of one condition: every scan is
        # read from disk once per condition, not once per epoch.  A 96x48x48x42
        # series is 74 MB as the float64 pickle and 37 MB as float32 here, so
        # the whole 70-scan set is ~2.6 GB in RAM - the same trade the group's
        # own loader makes with ``preload: True``.  Without it an epoch is
        # bounded by disk, not the GPU (28 s/epoch was measured on real data).
        self._cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = (
            cache if cache is not None else {}
        )

        missing = [
            s for s in self.scan_ids
            if not pkl_io.img_path(self.data_root, s, self.img_shape).exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} scans have no image under {self.data_root}: {missing[:5]}"
            )

        if preload:
            for scan_id in self.scan_ids:
                self._cache[scan_id] = self._read(scan_id)
            LOGGER.info("Preloaded %d scans from %s", len(self._cache), self.data_root)

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.scan_ids)

    def raw(self, index: int) -> tuple[str, np.ndarray, np.ndarray]:
        """``(scan_id, image, aif)`` without any copy - for building resident tensors."""
        scan_id = self.scan_ids[index]
        cached = self._cache.get(scan_id)
        if cached is None:
            cached = self._read(scan_id)
            self._cache[scan_id] = cached
        return scan_id, cached[0], cached[1]

    def _read(self, scan_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        image, _ = pkl_io.load_img(self.data_root, scan_id, self.img_shape)
        aif, times = pkl_io.load_aif(self.aif_root, scan_id)
        return image.astype(np.float32), aif.astype(np.float32), times.astype(np.float32)

    def __getitem__(self, index: int) -> dict[str, Any]:
        scan_id = self.scan_ids[index]
        cached = self._cache.get(scan_id)
        if cached is None:
            cached = self._read(scan_id)
            self._cache[scan_id] = cached
        image, aif, times = cached
        image = np.array(image, copy=True)

        if self.augment_on_device:
            return {"INPUT": image[None, ...], "AIF": aif, "ID": scan_id, "TIME": times}

        if self.augment:
            image = self._augment(image)

        if self.add_average:
            from .adapter import add_average_channel

            tensor = add_average_channel(image)
        else:
            tensor = image[None, ...]

        return {"INPUT": tensor, "AIF": aif, "ID": scan_id, "TIME": times}

    def _augment(self, image: np.ndarray) -> np.ndarray:
        """The group's augmentations: Poisson noise and in-plane flips.

        Reproduced exactly, including the noise formula: a random scale in
        ``[0, 1]`` sets the Poisson rate from the voxel values themselves, and
        the noise added is ``Poisson(lambda) - lambda`` so the augmentation is
        zero-mean.  Keeping this identical to the baseline protocol is what lets
        a retrained model differ from the baseline only in its inputs.
        """
        if self.poisson_noise:
            scale = float(self._rng.uniform(0.0, 1.0))
            rate = np.clip(image * scale, 0.0, None)
            image = image + (self._rng.poisson(rate) - rate).astype(np.float32)
        if self.random_flip and self._rng.random() < 0.5:
            image = np.flip(image, axis=(-1, -2))
        return np.ascontiguousarray(image, dtype=np.float32)

    # ------------------------------------------------------------------ #
    def as_torch(self):
        """Wrap this dataset for ``torch.utils.data.DataLoader``."""
        import torch
        from torch.utils.data import Dataset

        outer = self

        class _TorchDataset(Dataset):
            def __len__(self) -> int:
                return len(outer)

            def __getitem__(self, index: int):
                item = outer[index]
                return {
                    "INPUT": torch.from_numpy(np.ascontiguousarray(item["INPUT"], dtype=np.float32)),
                    "AIF": torch.from_numpy(np.ascontiguousarray(item["AIF"], dtype=np.float32)),
                    "TIME": torch.from_numpy(np.ascontiguousarray(item["TIME"], dtype=np.float32)),
                    "ID": item["ID"],
                }

        return _TorchDataset()


def WeightedMSELoss():  # noqa: N802 - mirrors the group's class name
    """The group's segmented, weighted MSE loss.

    Weights 0.4 / 0.7 / 1.0 over frames [0, 25), [25, 34), [34, 42): the late
    frames, where the curve is flattest and the count statistics best, carry the
    most weight.  Reimplemented here rather than imported so training does not
    depend on the repository being importable inside worker processes; it is
    numerically identical.
    """
    import torch
    import torch.nn as nn

    class _WeightedMSELoss(nn.Module):
        def __init__(self, indices: Sequence[int] = (0, 25, 34, 42), weights: Sequence[float] = (0.4, 0.7, 1.0)):
            super().__init__()
            self.indices = tuple(int(i) for i in indices)
            self.weights = tuple(float(w) for w in weights)

        def forward(self, predictions, targets):
            if predictions.shape != targets.shape:
                raise ValueError(
                    f"prediction shape {tuple(predictions.shape)} != target shape {tuple(targets.shape)}"
                )
            n_frames = predictions.shape[1]
            bounds = [min(i, n_frames) for i in self.indices]
            losses = []
            for i in range(len(bounds) - 1):
                start, end = bounds[i], bounds[i + 1]
                if end <= start:
                    continue
                losses.append(
                    torch.mean(self.weights[i] * (predictions[:, start:end] - targets[:, start:end]) ** 2)
                )
            if not losses:
                return torch.mean((predictions - targets) ** 2)
            return torch.sum(torch.stack(losses))

    return _WeightedMSELoss()
