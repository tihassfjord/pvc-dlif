"""Retraining and fine-tuning DLIF on PVC-corrected data.

Deconvolution changes the noise, the edge sharpness and the high-frequency
content of the images.  A network trained on uncorrected images and tested on
corrected ones is tested under a train-test distribution shift, so a negative
result in that setting cannot be separated from a failure to generalise.
Retraining is therefore part of the main hypothesis, not an optional extra.

The protocol is deliberately copied from the baseline: same folds, same number
of repeats, same optimiser, schedule, loss and stopping rule.  Only the input
representation changes, so the comparison isolates it.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..logging_utils import get_logger
from .dataset import DlifDataset, Fold, WeightedMSELoss, make_folds, stratified_val_split

LOGGER = get_logger(__name__)

__all__ = ["TrainSettings", "RunResult", "train_one_run", "train_condition", "set_seed"]


def set_seed(seed: int) -> None:
    """Seed every generator that affects training."""
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TrainSettings:
    """Training hyper-parameters, mirroring the baseline configuration."""

    epochs: int = 1000          # upstream config.yaml / EJNMMI Res. 2026
    min_epochs: int = 20
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    optimizer: str = "Adam"
    loss: str = "WeightedMSELoss"
    scheduler: str = "CosineAnnealingLR"
    eta_min: float = 1e-4
    t_max: int = 100
    # The group's config has ``training.scheduler: False``: the scheduler section
    # exists but is switched off, so the published runs used a constant learning
    # rate.  Off by default here for the same reason.
    use_scheduler: bool = False
    early_stopping: bool = False
    patience: int = 30
    monitor: str = "val_loss"
    device: str = "cuda"
    amp: bool = True
    num_workers: int = 0

    @classmethod
    def from_config(cls, config, finetune: bool = False) -> "TrainSettings":
        train = dict(config.get("dlif.train", {}))
        settings = cls(
            epochs=int(train.get("epochs", 1000)),
            min_epochs=int(train.get("min_epochs", 20)),
            batch_size=int(train.get("batch_size", 8)),
            learning_rate=float(train.get("learning_rate", 1e-4)),
            weight_decay=float(train.get("weight_decay", 0.0)),
            optimizer=str(train.get("optimizer", "Adam")),
            loss=str(train.get("loss", "WeightedMSELoss")),
            scheduler=str(train.get("scheduler", "CosineAnnealingLR")),
            use_scheduler=bool(train.get("use_scheduler", False)),
            eta_min=float(train.get("eta_min", 1e-4)),
            t_max=int(train.get("t_max", 100)),
            early_stopping=bool(train.get("early_stopping", False)),
            patience=int(train.get("patience", 30)),
            monitor=str(train.get("monitor", "val_loss")),
            device=str(train.get("device", "cuda")),
            amp=bool(train.get("amp", True)),
        )
        if finetune:
            ft = dict(config.get("dlif.finetune", {}))
            settings.learning_rate = float(ft.get("learning_rate", settings.learning_rate / 10))
            settings.epochs = int(ft.get("epochs", 100))
        return settings

    def resolve_device(self) -> str:
        import torch

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            LOGGER.warning("CUDA requested but not available; training on CPU")
            return "cpu"
        return self.device


@dataclass
class RunResult:
    """Outcome of one (fold, run) training job."""

    fold: int
    run: int
    checkpoint: str
    best_val_loss: float
    best_epoch: int
    epochs_trained: int
    seconds: float
    train_ids: list[str]
    val_ids: list[str]
    test_ids: list[str]
    history: dict[str, list[float]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build_optimizer(model, settings: TrainSettings):
    import torch

    name = settings.optimizer.lower()
    params = model.parameters()
    if name == "adam":
        return torch.optim.Adam(params, lr=settings.learning_rate, weight_decay=settings.weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=settings.learning_rate, weight_decay=settings.weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=settings.learning_rate, momentum=0.9,
                               weight_decay=settings.weight_decay)
    raise ValueError(f"Unsupported optimizer {settings.optimizer!r}")


def _build_scheduler(optimizer, settings: TrainSettings):
    import torch

    if not settings.use_scheduler:
        return None
    name = settings.scheduler.lower()
    if name == "cosineannealinglr":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=settings.t_max, eta_min=settings.eta_min
        )
    if name == "reducelronplateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.9, patience=10, min_lr=settings.eta_min
        )
    raise ValueError(f"Unsupported scheduler {settings.scheduler!r}")


def _build_loss(settings: TrainSettings):
    import torch.nn as nn

    name = settings.loss.lower()
    if name == "weightedmseloss":
        return WeightedMSELoss()
    if name == "mse":
        return nn.MSELoss()
    if name == "l1":
        return nn.L1Loss()
    if name == "smoothl1":
        return nn.SmoothL1Loss()
    raise ValueError(f"Unsupported loss {settings.loss!r}")


def augment_on_device(inputs, generator, poisson_noise: bool, random_flip: bool,
                      add_average: bool, start_frame: int = 35):
    """The group's augmentation, applied to a batch already on the GPU.

    ``inputs`` is ``(B, 1, T, Z, Y, X)``.  Per sample: a scale drawn from
    U(0, 1) sets the Poisson rate from the voxel values, the noise added is
    ``Poisson(rate) - rate`` (zero mean), then an in-plane flip with p = 0.5;
    finally the late-frame average is stacked as a second channel if the model
    takes one.  Same distribution as the NumPy path in ``DlifDataset._augment``
    and the group's ``add_poisson_noise`` / ``apply_augmentations``, just not
    on the CPU: this is what turns a 40 s epoch into a 2 s one.
    """
    import torch

    x = inputs
    if poisson_noise:
        batch = x.shape[0]
        scale = torch.rand(batch, 1, 1, 1, 1, 1, device=x.device, generator=generator)
        rate = (x * scale).clamp_min_(0.0)
        x = x + (torch.poisson(rate, generator=generator) - rate)
    if random_flip:
        flip = torch.rand(x.shape[0], device=x.device, generator=generator) < 0.5
        if bool(flip.any()):
            x = x.clone()
            x[flip] = torch.flip(x[flip], dims=(-1, -2))
    if add_average:
        late = x[:, :1, start_frame:].mean(dim=2, keepdim=True).expand(-1, -1, x.shape[2], -1, -1, -1)
        x = torch.cat([x, late], dim=1)
    return x


class ResidentBatches:
    """The whole set as one tensor, batches gathered by index - no host copies.

    Every epoch of the DataLoader path copies each scan four or five times on
    the CPU (item copy, contiguous copy, collate, pin, transfer): ~2 GB of
    memcpy plus a PCIe transfer per epoch, several seconds with the GPU idle.
    Here the stacked tensor lives on the training device when it fits (checked
    against free memory with a margin for activations), else half precision on
    the device, else pinned host memory; a batch is then a gather on the device
    or one asynchronous transfer.  Shuffling uses the run's generator.
    """

    def __init__(self, dataset, device: str, generator, batch_size: int, shuffle: bool,
                 activation_margin_gb: float = 3.0):
        import torch

        # Fill a preallocated tensor straight from the dataset cache: one copy
        # of the data, not three (item copy + stack + tensor) - the peak memory
        # of a 70-scan condition is then ~2.6 GB, not ~8 GB.
        n = len(dataset)
        first_id, first_img, first_aif = dataset.raw(0)
        x = torch.empty((n, 1, *first_img.shape), dtype=torch.float32)
        y = torch.empty((n, *first_aif.shape), dtype=torch.float32)
        self.ids = []
        for i in range(n):
            scan_id, img, aif = dataset.raw(i)
            x[i, 0].copy_(torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32)))
            y[i].copy_(torch.from_numpy(np.asarray(aif, dtype=np.float32)))
            self.ids.append(scan_id)
        self.n = n
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.generator = generator
        self.device = device
        self.where = "host"

        if device.startswith("cuda"):
            free, _total = torch.cuda.mem_get_info(torch.device(device))
            need32 = x.numel() * 4 + activation_margin_gb * 1024 ** 3
            need16 = x.numel() * 2 + activation_margin_gb * 1024 ** 3
            if free > need32:
                x = x.to(device); self.where = "device float32"
            elif free > need16:
                x = x.to(device, dtype=torch.float16); self.where = "device float16"
            else:
                x = x.pin_memory(); self.where = "pinned host"
            y = y.to(device)
        self.x, self.y = x, y

    def to_host(self) -> None:
        """Move the data off the device (after an out-of-memory during training)."""
        if self.x.is_cuda:
            self.x = self.x.float().cpu().pin_memory()
            self.where = "pinned host (after OOM)"

    def __len__(self) -> int:
        return (self.n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        import torch

        if self.shuffle:
            order = torch.randperm(self.n, generator=self.generator, device=self.x.device
                                   if self.x.is_cuda else "cpu")
        else:
            order = torch.arange(self.n, device=self.x.device if self.x.is_cuda else "cpu")
        for start in range(0, self.n, self.batch_size):
            idx = order[start:start + self.batch_size]
            xb = self.x[idx]
            if not xb.is_cuda:
                xb = xb.to(self.device, non_blocking=True)
            yield {"INPUT": xb.float(), "AIF": self.y[idx.to(self.y.device)]}


def _epoch(model, loader, criterion, device, optimizer=None, scaler=None, amp: bool = False,
           device_aug: dict | None = None, generator=None) -> float:
    import torch

    training = optimizer is not None
    model.train(training)
    total, n_batches = 0.0, 0

    for batch in loader:
        inputs = batch["INPUT"].to(device, dtype=torch.float32, non_blocking=True)
        targets = batch["AIF"].to(device, dtype=torch.float32, non_blocking=True)
        if device_aug is not None:
            # Validation gets the average channel but no noise or flips.
            inputs = augment_on_device(
                inputs, generator,
                poisson_noise=training and device_aug["poisson_noise"],
                random_flip=training and device_aug["random_flip"],
                add_average=device_aug["add_average"],
            )

        with torch.set_grad_enabled(training):
            if amp and device.startswith("cuda"):
                with torch.autocast("cuda", dtype=torch.float16):
                    output = model(inputs)
                    prediction = output["REG"] if isinstance(output, dict) else output
                    loss = criterion(prediction.float(), targets)
            else:
                output = model(inputs)
                prediction = output["REG"] if isinstance(output, dict) else output
                loss = criterion(prediction, targets)

            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and device.startswith("cuda"):
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        total += float(loss.detach().cpu())
        n_batches += 1

    return total / max(n_batches, 1)


def train_one_run(
    model,
    train_dataset: DlifDataset,
    val_dataset: DlifDataset,
    settings: TrainSettings,
    checkpoint_path: Path,
    seed: int = 42,
) -> tuple[Any, dict[str, Any]]:
    """Train one model, keeping the checkpoint with the best validation loss."""
    import torch
    from torch.utils.data import DataLoader

    set_seed(seed)
    device = settings.resolve_device()
    model = model.to(device)

    resident = getattr(train_dataset, "augment_on_device", False)
    if resident:
        # Everything on the device; the generator is created below and shared.
        generator = torch.Generator(device=device if device.startswith("cuda") else "cpu")
        generator.manual_seed(int(seed))
        train_loader = ResidentBatches(train_dataset, device, generator, settings.batch_size, shuffle=True)
        val_loader = ResidentBatches(val_dataset, device, generator, settings.batch_size, shuffle=False)
        LOGGER.info("  data resident: train %s, val %s", train_loader.where, val_loader.where)
    else:
        train_loader = DataLoader(
            train_dataset.as_torch(), batch_size=settings.batch_size, shuffle=True,
            num_workers=settings.num_workers, pin_memory=device.startswith("cuda"), drop_last=False,
        )
        val_loader = DataLoader(
            val_dataset.as_torch(), batch_size=settings.batch_size, shuffle=False,
            num_workers=settings.num_workers, pin_memory=device.startswith("cuda"),
        )

    optimizer = _build_optimizer(model, settings)
    scheduler = _build_scheduler(optimizer, settings)
    criterion = _build_loss(settings)
    scaler = torch.amp.GradScaler("cuda") if (settings.amp and device.startswith("cuda")) else None

    # Augmentation on the GPU when there is one (see augment_on_device); the
    # datasets were told to hand out raw images in that case.
    device_aug = None
    if resident:
        device_aug = {"poisson_noise": train_dataset.poisson_noise,
                      "random_flip": train_dataset.random_flip,
                      "add_average": train_dataset.add_average}
    else:
        generator = None

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "lr": []}
    best_loss, best_epoch, stale = float("inf"), -1, 0
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    epoch = 0
    progress_path = checkpoint_path.with_name("progress.json")

    for epoch in range(settings.epochs):
        try:
            train_loss = _epoch(model, train_loader, criterion, device, optimizer, scaler, settings.amp,
                                device_aug=device_aug, generator=generator)
        except torch.cuda.OutOfMemoryError:
            # The resident data plus activations did not fit after all: move
            # the data back to host memory and retry this epoch once.
            if not (resident and getattr(train_loader, "x", None) is not None and train_loader.x.is_cuda):
                raise
            LOGGER.warning("GPU out of memory with the data resident; moving data to host and retrying")
            train_loader.to_host(); val_loader.to_host()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            train_loss = _epoch(model, train_loader, criterion, device, optimizer, scaler, settings.amp,
                                device_aug=device_aug, generator=generator)
        val_loss = _epoch(model, val_loader, criterion, device, device_aug=device_aug, generator=generator)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        if scheduler is not None:
            if settings.scheduler.lower() == "reducelronplateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

        monitored = val_loss if settings.monitor == "val_loss" else train_loss
        if monitored < best_loss:
            best_loss, best_epoch, stale = monitored, epoch, 0
            torch.save(
                {"state_dict": model.state_dict(), "epoch": epoch, "val_loss": val_loss},
                str(checkpoint_path),
            )
        else:
            stale += 1
            if settings.early_stopping and stale >= settings.patience and epoch >= settings.min_epochs:
                LOGGER.info("Early stopping at epoch %d (best %d)", epoch, best_epoch)
                break

        # Heartbeat: a small JSON the GUI and `status.py` read for a live
        # "epoch e/N, s/epoch, ETA" line, rewritten every epoch.
        elapsed = time.perf_counter() - started
        per_epoch = elapsed / (epoch + 1)
        _write_progress(progress_path, {
            "epoch": epoch + 1, "epochs": settings.epochs,
            "train_loss": float(train_loss), "val_loss": float(val_loss),
            "best_val_loss": float(best_loss), "best_epoch": int(best_epoch),
            "seconds_per_epoch": per_epoch,
            "eta_seconds": per_epoch * (settings.epochs - epoch - 1),
            "updated": time.time(),
        })

        # Log the first few epochs every time (is it running at all, and how
        # fast), then every 10th - a run of 1000 epochs should not be silent.
        if epoch < 3 or (epoch + 1) % 10 == 0 or epoch == settings.epochs - 1:
            LOGGER.info("  epoch %4d/%d  train %.5f  val %.5f  best %.5f  (%.1f s/epoch, ETA %s)",
                        epoch + 1, settings.epochs, train_loss, val_loss, best_loss,
                        per_epoch, _fmt_eta(per_epoch * (settings.epochs - epoch - 1)))

    # Restore the best weights so the returned model is the one that is scored.
    payload = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    return model, {
        "best_val_loss": best_loss,
        "best_epoch": best_epoch,
        "epochs_trained": epoch + 1,
        "seconds": time.perf_counter() - started,
        "history": history,
    }


def _write_progress(path: Path, payload: dict) -> None:
    """Atomic-enough rewrite of the heartbeat file (write, then rename)."""
    try:
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(path)
    except OSError:                       # never let the heartbeat stop a run
        pass


def _fmt_eta(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def _run_is_complete(summary_path: Path, settings: "TrainSettings") -> tuple[bool, str]:
    """Whether an existing run may be reused under the current settings.

    A pilot leaves 5-epoch checkpoints behind; the full run must not accept
    them as finished.  A run counts as complete when it trained the configured
    number of epochs, or stopped early under a protocol that allows it.
    """
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, "unreadable summary"
    trained = int(summary.get("epochs_trained", 0))
    if trained >= settings.epochs:
        return True, ""
    if settings.early_stopping and trained >= settings.min_epochs:
        return True, ""
    return False, f"incomplete ({trained} of {settings.epochs} epochs)"


def train_condition(
    model_factory,
    data_root: Path,
    aif_root: Path,
    scan_ids: Sequence[str],
    group_of: Mapping[str, str | None],
    out_dir: Path,
    settings: TrainSettings,
    n_folds: int = 10,
    n_runs: int = 10,
    validation_size: float = 0.15,
    img_shape: Sequence[int] = (96, 48, 48),
    augmentation: Mapping[str, Any] | None = None,
    seed: int = 42,
    resume: bool = True,
    folds: Sequence[Fold] | None = None,
    augment_on_device: bool | None = None,
) -> list[RunResult]:
    """Run the full cross-validated training protocol for one condition.

    ``augment_on_device`` moves the Poisson-noise/flip augmentation (and the
    average channel) onto the training device.  ``None`` means "when that
    device is a GPU", which is the only case where it is faster.

    ``model_factory`` is called for every run and returns a fresh model, either
    randomly initialised (retraining) or preloaded with the pretrained weights
    (fine-tuning).

    ``folds`` can be passed in so that every condition uses the *same* folds;
    otherwise they are derived from the scan IDs and the seed, which gives the
    same result as long as the ID list matches.
    """
    augmentation = dict(augmentation or {})
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    folds = list(folds) if folds is not None else make_folds(scan_ids, n_folds, seed)
    (out_dir / "folds.json").write_text(
        json.dumps([f.as_dict() for f in folds], indent=2), encoding="utf-8"
    )

    # One in-memory copy of every scan for this condition, shared by all runs.
    shared_cache: dict = {}
    if augment_on_device is None:
        augment_on_device = settings.resolve_device().startswith("cuda")
    if augment_on_device:
        LOGGER.info("augmentation runs on the training device")

    results: list[RunResult] = []
    total_jobs = len(folds) * n_runs
    job = 0
    trained_jobs, trained_seconds = 0, 0.0        # for the ETA line
    for fold in folds:
        for run in range(1, n_runs + 1):
            job += 1
            run_dir = out_dir / f"fold_{fold.index:02d}" / f"run_{run:02d}"
            checkpoint = run_dir / "model.pt"
            summary_path = run_dir / "summary.json"

            if resume and checkpoint.exists() and summary_path.exists():
                complete, why = _run_is_complete(summary_path, settings)
                if complete:
                    LOGGER.info("skip fold %d run %d (complete)", fold.index, run)
                    results.append(RunResult(**json.loads(summary_path.read_text(encoding="utf-8"))))
                    continue
                LOGGER.warning("fold %d run %d exists but is %s - retraining", fold.index, run, why)

            run_seed = seed + 1000 * fold.index + run
            train_ids, val_ids = stratified_val_split(
                fold.train_ids, group_of, validation_size, seed=run_seed
            )

            train_dataset = DlifDataset(
                data_root, train_ids, aif_root, img_shape, mode="train", augment=True,
                poisson_noise=bool(augmentation.get("poisson_noise", True)),
                random_flip=bool(augmentation.get("random_flip", True)),
                add_average=bool(augmentation.get("add_average", False)),
                seed=run_seed, cache=shared_cache, augment_on_device=augment_on_device,
            )
            val_dataset = DlifDataset(
                data_root, val_ids, aif_root, img_shape, mode="val", augment=False,
                add_average=bool(augmentation.get("add_average", False)), seed=run_seed,
                cache=shared_cache, augment_on_device=augment_on_device,
            )

            eta = ""
            if trained_jobs:
                per_job = trained_seconds / trained_jobs
                eta = f", ETA for this condition {_fmt_eta(per_job * (total_jobs - job + 1))}"
            LOGGER.info(
                "fold %d/%d run %d/%d (job %d of %d%s) -- %d train, %d val, %d test",
                fold.index, len(folds), run, n_runs, job, total_jobs, eta,
                len(train_ids), len(val_ids), len(fold.test_ids),
            )
            model = model_factory()
            model, info = train_one_run(model, train_dataset, val_dataset, settings, checkpoint, run_seed)
            trained_jobs += 1
            trained_seconds += float(info["seconds"])

            result = RunResult(
                fold=fold.index,
                run=run,
                checkpoint=str(checkpoint),
                best_val_loss=float(info["best_val_loss"]),
                best_epoch=int(info["best_epoch"]),
                epochs_trained=int(info["epochs_trained"]),
                seconds=float(info["seconds"]),
                train_ids=list(train_ids),
                val_ids=list(val_ids),
                test_ids=list(fold.test_ids),
                history=info["history"],
            )
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
            results.append(result)

    return results
