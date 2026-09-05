"""Bridge to the group's DLIF implementation.

The thesis does not reimplement the network.  It uses the architecture and the
pretrained weights developed in the ML4PET group, so that the baseline
condition really is the deployed model and any difference between conditions is
attributable to the input representation rather than to a reimplementation.

This module takes care of the awkward parts of borrowing that code:

* the repository is a script tree, not an installed package, so ``src`` has to
  be put on ``sys.path`` before ``models`` and ``datahandlers`` import;
* the distributed checkpoint is a pickled *model object*, not a state dict, so
  unpickling it needs those same modules importable;
* the pretrained encoder was trained with two input channels while
  ``DLIFNet_MAX`` takes one, so the first convolution has to be sliced.

Everything here is read-only with respect to the group's repository.
"""

from __future__ import annotations

import contextlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from ..logging_utils import get_logger

LOGGER = get_logger(__name__)

__all__ = [
    "DlifRepo", "ModelInputSpec", "build_model", "load_pretrained_module",
    "load_pretrained_model", "load_checkpoint", "infer_input_spec",
    "predict_series", "count_parameters",
]


@dataclass
class DlifRepo:
    """A checked-out copy of the group's DLIF repository."""

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        if not self.src.exists():
            raise FileNotFoundError(
                f"{self.root} does not look like the DLIF repository (no src/ directory)"
            )

    @property
    def src(self) -> Path:
        return self.root / "src"

    @property
    def default_weights(self) -> Path:
        return self.src / "models" / "pretrained_weigths" / "DLIFNet.pt"

    @contextlib.contextmanager
    def on_path(self) -> Iterator[None]:
        """Temporarily put ``src`` first on ``sys.path``."""
        entry = str(self.src.resolve())
        added = entry not in sys.path
        if added:
            sys.path.insert(0, entry)
        try:
            yield
        finally:
            if added:
                with contextlib.suppress(ValueError):
                    sys.path.remove(entry)

    def import_models(self):
        with self.on_path():
            import models.models as models_module  # type: ignore

            return models_module

    def import_shared_groups(self) -> dict[str, list[str]]:
        with self.on_path():
            from datahandlers.shared_dicts import groups  # type: ignore

            return {str(k): [str(v) for v in vals] for k, vals in groups.items()}

    def available_models(self) -> list[str]:
        import torch.nn as nn

        module = self.import_models()
        return sorted(
            name for name, obj in vars(module).items()
            if isinstance(obj, type) and issubclass(obj, nn.Module) and obj.__module__ == module.__name__
        )


def build_model(repo: DlifRepo, name: str = "DLIFNet_MAX", in_channels: int = 1):
    """Instantiate one of the group's architectures by name."""
    module = repo.import_models()
    if not hasattr(module, name):
        raise AttributeError(
            f"Model {name!r} not found in the DLIF repository. Available: {repo.available_models()}"
        )
    factory = getattr(module, name)
    try:
        return factory(in_channels=in_channels)
    except TypeError:
        # Some variants take no arguments.
        return factory()


def _state_dict_from(obj: Any) -> dict[str, Any]:
    """Accept a pickled model, a state dict, or a training checkpoint."""
    if hasattr(obj, "state_dict") and callable(obj.state_dict):
        return dict(obj.state_dict())
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in obj:
                inner = obj[key]
                if hasattr(inner, "state_dict"):
                    return dict(inner.state_dict())
                if isinstance(inner, dict):
                    return dict(inner)
        return dict(obj)
    raise TypeError(f"Cannot extract a state dict from {type(obj).__name__}")


def _adapt_first_conv(state: dict[str, Any], in_channels: int) -> dict[str, Any]:
    """Slice the first convolution when the checkpoint has more input channels.

    The published weights were trained with an extra averaged-image channel;
    ``DLIFNet_MAX`` uses one.  This mirrors what the group's own
    ``load_pretrained`` does, and it is done explicitly here so it appears in
    the record rather than happening silently.
    """
    adapted = dict(state)
    for key in ("encoder.Conv1.conv.0.weight", "encoder.Conv1.match_dim.0.weight"):
        if key in adapted:
            weight = adapted[key]
            if weight.ndim == 5 and weight.shape[1] > in_channels:
                LOGGER.info(
                    "Adapting %s from %d to %d input channels", key, weight.shape[1], in_channels
                )
                adapted[key] = weight[:, :in_channels, :, :, :].contiguous()
    return adapted


def load_pretrained_model(
    repo: DlifRepo,
    weights: Path | str | None = None,
    name: str = "DLIFNet_MAX",
    in_channels: int = 1,
    device: str = "cpu",
    strict: bool = False,
):
    """Build the architecture and load the group's pretrained weights into it.

    Returns ``(model, info)`` where ``info`` records exactly which parameters
    were transferred and which were left at their initial values -- that record
    belongs in the thesis, because a baseline that quietly failed to load half
    its weights would look like a PVC effect.
    """
    import torch

    weights_path = Path(weights) if weights else repo.default_weights
    if not weights_path.is_absolute():
        weights_path = repo.root / weights_path
    if not weights_path.exists():
        raise FileNotFoundError(f"Pretrained weights not found: {weights_path}")

    model = build_model(repo, name, in_channels)

    with repo.on_path():
        # weights_only=False is required: the file is a pickled nn.Module, not
        # a state dict.  It is the group's own artefact, from a trusted path.
        payload = torch.load(str(weights_path), map_location=device, weights_only=False)

    state = _adapt_first_conv(_state_dict_from(payload), in_channels)
    model_state = model.state_dict()

    transferable = {
        k: v for k, v in state.items()
        if k in model_state and tuple(model_state[k].shape) == tuple(v.shape)
    }
    shape_mismatch = {
        k: (tuple(v.shape), tuple(model_state[k].shape))
        for k, v in state.items() if k in model_state and tuple(model_state[k].shape) != tuple(v.shape)
    }
    missing = sorted(set(model_state) - set(transferable))
    unexpected = sorted(set(state) - set(model_state))

    model_state.update(transferable)
    model.load_state_dict(model_state, strict=strict)
    model.to(device)
    model.eval()

    info = {
        "weights": str(weights_path),
        "architecture": name,
        "in_channels": in_channels,
        "n_parameters": count_parameters(model),
        "transferred": len(transferable),
        "total_in_model": len(model_state),
        "missing_from_checkpoint": missing,
        "unexpected_in_checkpoint": unexpected,
        "shape_mismatch": shape_mismatch,
    }
    if missing:
        LOGGER.warning(
            "%d/%d parameter tensors were not present in the checkpoint and keep their "
            "initial values: %s", len(missing), len(model_state), missing[:8],
        )
    if shape_mismatch:
        LOGGER.warning("Shape mismatches skipped: %s", shape_mismatch)
    if len(transferable) < 0.9 * len(model_state):
        LOGGER.error(
            "Only %d of %d tensors were transferred. The architecture built from the "
            "repository's current models.py does not match this checkpoint, so this is NOT "
            "the deployed model -- most of it is randomly initialised. Use "
            "load_pretrained_module() instead, which loads the checkpoint as the pickled "
            "module it is and therefore carries its own architecture.",
            len(transferable), len(model_state),
        )
    LOGGER.info(
        "Loaded %s from %s (%d/%d tensors, %d parameters)",
        name, weights_path.name, len(transferable), len(model_state), info["n_parameters"],
    )
    return model, info


@dataclass(frozen=True)
class ModelInputSpec:
    """What a model expects to be fed.

    Recovered from the weights rather than assumed, because the two are easy to
    get out of step: the distributed checkpoint takes two channels and a
    64 x 48 x 48 volume, while the current repository code defaults to one
    channel and the data directory that ships with it is 96 x 48 x 48.  Feeding
    the wrong one either crashes or, worse, silently produces a baseline that is
    not the deployed model.
    """

    in_channels: int
    spatial_shape: tuple[int, int, int] | None
    add_average: bool
    fixed_spatial: bool
    note: str = ""
    min_spatial_shape: tuple[int, int, int] | None = None

    def check_shape(self, shape: Sequence[int]) -> str | None:
        """Why ``shape`` is unusable for this model, or ``None`` if it is fine.

        A model with adaptive pooling accepts a *range* of sizes, not any size:
        its bottleneck convolution still needs a minimum extent after the
        encoder's pooling stages. Feeding it something smaller raises a bare
        shape error from inside torch, which is a poor way to discover that a
        configured input shape is wrong.
        """
        shape = tuple(int(v) for v in shape)
        if self.fixed_spatial and self.spatial_shape is not None:
            if shape != tuple(self.spatial_shape):
                return (
                    f"this model needs exactly {tuple(self.spatial_shape)} volumes, not {shape}. "
                    f"{self.note}"
                )
            return None
        if self.min_spatial_shape is not None:
            too_small = [
                f"{'zyx'[a]} needs at least {self.min_spatial_shape[a]}, got {shape[a]}"
                for a in range(3) if shape[a] < self.min_spatial_shape[a]
            ]
            if too_small:
                return (
                    f"input shape {shape} is too small for this model: "
                    + "; ".join(too_small) + f". {self.note}"
                )
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "in_channels": self.in_channels,
            "spatial_shape": list(self.spatial_shape) if self.spatial_shape else None,
            "min_spatial_shape": list(self.min_spatial_shape) if self.min_spatial_shape else None,
            "add_average": self.add_average,
            "fixed_spatial": self.fixed_spatial,
            "note": self.note,
        }


def infer_input_spec(model) -> ModelInputSpec:
    """Work out a model's expected input from its own weights.

    The number of input channels comes from the first 3D convolution.  Whether
    the spatial size is fixed comes from the bottleneck: a plain ``Conv3d`` with
    no adaptive pooling after it only collapses to a single voxel for one input
    size, which is then solved for from the kernel and the number of pooling
    stages.  A model with an ``AdaptiveAvgPool3d`` bottleneck accepts any size.
    """
    import torch.nn as nn

    first_conv = next((m for m in model.modules() if isinstance(m, nn.Conv3d)), None)
    in_channels = int(first_conv.in_channels) if first_conv is not None else 1

    has_adaptive = any(isinstance(m, nn.AdaptiveAvgPool3d) for m in model.modules())
    n_pools = sum(1 for m in model.modules() if isinstance(m, (nn.MaxPool3d, nn.AvgPool3d)))

    bottleneck = getattr(model, "conv3d", None)
    kernel: tuple[int, int, int] | None = None
    if isinstance(bottleneck, nn.Conv3d):
        kernel = tuple(int(k) for k in bottleneck.kernel_size)  # type: ignore[assignment]
    elif isinstance(bottleneck, nn.Sequential):
        inner = next((m for m in bottleneck if isinstance(m, nn.Conv3d)), None)
        if inner is not None:
            kernel = tuple(int(k) for k in inner.kernel_size)  # type: ignore[assignment]

    encoder = getattr(model, "encoder", None)
    stages = (
        sum(1 for name, _ in encoder.named_children() if name.lower().startswith("conv"))
        if encoder is not None else max(n_pools, 1)
    )
    downsample = 2 ** max(stages - 1, 0)

    if has_adaptive or kernel is None:
        minimum = (
            tuple(int(k * downsample) for k in kernel) if kernel is not None else None
        )
        return ModelInputSpec(
            in_channels=in_channels,
            spatial_shape=None,
            add_average=in_channels > 1,
            fixed_spatial=False,
            min_spatial_shape=minimum,  # type: ignore[arg-type]
            note=(
                "adaptive pooling bottleneck: any input size at or above "
                f"{minimum} is accepted" if minimum else
                "adaptive pooling bottleneck: any input size is accepted"
            ),
        )

    # The encoder halves each spatial axis once per pooling stage; the
    # bottleneck convolution then has to consume exactly the kernel extent.
    spatial = tuple(int(k * downsample) for k in kernel)

    return ModelInputSpec(
        in_channels=in_channels,
        spatial_shape=spatial,      # type: ignore[arg-type]
        min_spatial_shape=spatial,  # type: ignore[arg-type]
        add_average=in_channels > 1,
        fixed_spatial=True,
        note=(
            f"bottleneck Conv3d kernel {kernel} after {stages - 1} pooling stages implies a "
            f"{spatial[0]}x{spatial[1]}x{spatial[2]} input"
        ),
    )


def load_pretrained_module(
    repo: DlifRepo,
    weights: Path | str | None = None,
    device: str = "cpu",
) -> tuple[Any, ModelInputSpec, dict[str, Any]]:
    """Load the deployed model as the pickled module it is.

    This is the right way to get the baseline.  The checkpoint is a pickled
    ``nn.Module``, so it carries the architecture it was trained with --
    including layers the current repository code no longer builds.  Rebuilding
    the architecture from today's ``models.py`` and pouring the weights in gives
    a different network with most tensors left at their initial values, which
    would make the "pretrained baseline" meaningless.

    Returns the model, the input specification recovered from its weights, and a
    record for provenance.
    """
    import torch

    weights_path = Path(weights) if weights else repo.default_weights
    if not weights_path.is_absolute():
        weights_path = repo.root / weights_path
    if not weights_path.exists():
        raise FileNotFoundError(f"Pretrained weights not found: {weights_path}")

    with repo.on_path():
        # weights_only=False is required: the file is a pickled nn.Module from
        # the group's own artefact directory, not a state dict.
        model = torch.load(str(weights_path), map_location=device, weights_only=False)

    if not hasattr(model, "forward"):
        raise TypeError(
            f"{weights_path} does not contain a model object; use load_pretrained_model() "
            "to pour a state dict into an architecture instead."
        )

    model.to(device)
    model.eval()
    spec = infer_input_spec(model)

    info = {
        "weights": str(weights_path),
        "architecture": type(model).__name__,
        "loaded_as": "pickled_module",
        "n_parameters": count_parameters(model),
        "input_spec": spec.to_dict(),
    }
    LOGGER.info(
        "Loaded the deployed %s (%d parameters); it expects %d channel(s) and %s input",
        info["architecture"], info["n_parameters"], spec.in_channels,
        "x".join(str(v) for v in spec.spatial_shape) if spec.spatial_shape else "any-size",
    )
    return model, spec, info


def load_checkpoint(path: Path | str, repo: DlifRepo | None = None, device: str = "cpu"):
    """Load a checkpoint written either by this pipeline or by the group's trainer."""
    import torch

    context = repo.on_path() if repo else contextlib.nullcontext()
    with context:
        payload = torch.load(str(path), map_location=device, weights_only=False)

    if hasattr(payload, "eval"):
        payload.to(device)
        payload.eval()
        return payload, {"format": "pickled_module", "path": str(path)}
    return payload, {"format": "state_dict", "path": str(path)}


def count_parameters(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


#: First frame of the late-frame window averaged into the second input channel.
#: Matches the group's ``datahandlers.augmentation.add_average``, which uses
#: ``img[35:]`` -- the last frames of the scan, where the distribution is
#: settled and the count statistics are best.
LATE_AVERAGE_START_FRAME = 35


def add_average_channel(image: np.ndarray, start_frame: int = LATE_AVERAGE_START_FRAME) -> np.ndarray:
    """Stack the late-frame average as a second channel.

    Reproduces the group's ``add_average`` exactly, including the fact that the
    average is taken over the *late* frames rather than the whole series.  Using
    the whole-series mean instead would change what the second channel means and
    silently shift every prediction.
    """
    array = np.asarray(image, dtype=np.float32)
    start = min(start_frame, array.shape[0] - 1)
    average = array[start:].mean(axis=0, keepdims=True)
    return np.stack((array, np.repeat(average, array.shape[0], axis=0)))


def predict_series(
    model,
    image: np.ndarray,
    device: str = "cpu",
    add_average: bool = False,
    spec: "ModelInputSpec | None" = None,
) -> np.ndarray:
    """Predict one input function from one dynamic series.

    ``image`` is ``(T, Z, Y, X)`` in SUV, exactly what the group's dataloader
    hands the network after adding a channel axis.  The returned curve has one
    value per frame.

    When ``spec`` is given it wins over ``add_average`` and the spatial shape is
    checked against it, so a mismatch is an error message rather than a silently
    wrong baseline.
    """
    import torch

    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 4:
        raise ValueError(f"expected (T, Z, Y, X); got {array.shape}")

    if spec is not None:
        add_average = spec.add_average
        problem = spec.check_shape(array.shape[1:])
        if problem:
            raise ValueError(
                problem + " Build the input tree at a supported shape rather than feeding "
                "it another size."
            )

    stacked = add_average_channel(array) if add_average else array[None, ...]
    tensor = torch.from_numpy(np.ascontiguousarray(stacked[None, ...], dtype=np.float32)).to(device)

    model.eval()
    with torch.no_grad():
        output = model(tensor)
    prediction = output["REG"] if isinstance(output, dict) else output
    return prediction.detach().cpu().numpy().reshape(-1).astype(np.float64)
