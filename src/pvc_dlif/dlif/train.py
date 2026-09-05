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

    epochs: int = 300
    min_epochs: int = 20
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    optimizer: str = "Adam"
    loss: str = "WeightedMSELoss"
    scheduler: str = "CosineAnnealingLR"
    eta_min: float = 1e-4
    t_max: int = 100
    use_scheduler: bool = True
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
            epochs=int(train.get("epochs", 300)),
            min_epochs=int(train.get("min_epochs", 20)),
            batch_size=int(train.get("batch_size", 16)),
            learning_rate=float(train.get("learning_rate", 1e-3)),
            weight_decay=float(train.get("weight_decay", 0.0)),
            optimizer=str(train.get("optimizer", "Adam")),
            loss=str(train.get("loss", "WeightedMSELoss")),
            scheduler=str(train.get("scheduler", "CosineAnnealingLR")),
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


def _epoch(model, loader, criterion, device, optimizer=None, scaler=None, amp: bool = False) -> float:
    import torch

    training = optimizer is not None
    model.train(training)
    total, n_batches = 0.0, 0

    for batch in loader:
        inputs = batch["INPUT"].to(device, dtype=torch.float32, non_blocking=True)
        targets = batch["AIF"].to(device, dtype=torch.float32, non_blocking=True)

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

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "lr": []}
    best_loss, best_epoch, stale = float("inf"), -1, 0
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    epoch = 0

    for epoch in range(settings.epochs):
        train_loss = _epoch(model, train_loader, criterion, device, optimizer, scaler, settings.amp)
        val_loss = _epoch(model, val_loader, criterion, device)

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

        if epoch % 25 == 0 or epoch == settings.epochs - 1:
            LOGGER.info("  epoch %3d  train %.5f  val %.5f  best %.5f",
                        epoch, train_loss, val_loss, best_loss)

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
) -> list[RunResult]:
    """Run the full cross-validated training protocol for one condition.

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

    results: list[RunResult] = []
    for fold in folds:
        for run in range(1, n_runs + 1):
            run_dir = out_dir / f"fold_{fold.index:02d}" / f"run_{run:02d}"
            checkpoint = run_dir / "model.pt"
            summary_path = run_dir / "summary.json"

            if resume and checkpoint.exists() and summary_path.exists():
                LOGGER.info("skip fold %d run %d (exists)", fold.index, run)
                results.append(RunResult(**json.loads(summary_path.read_text(encoding="utf-8"))))
                continue

            run_seed = seed + 1000 * fold.index + run
            train_ids, val_ids = stratified_val_split(
                fold.train_ids, group_of, validation_size, seed=run_seed
            )

            train_dataset = DlifDataset(
                data_root, train_ids, aif_root, img_shape, mode="train", augment=True,
                poisson_noise=bool(augmentation.get("poisson_noise", True)),
                random_flip=bool(augmentation.get("random_flip", True)),
                add_average=bool(augmentation.get("add_average", False)),
                seed=run_seed,
            )
            val_dataset = DlifDataset(
                data_root, val_ids, aif_root, img_shape, mode="val", augment=False,
                add_average=bool(augmentation.get("add_average", False)), seed=run_seed,
            )

            LOGGER.info(
                "fold %d/%d run %d/%d -- %d train, %d val, %d test",
                fold.index, len(folds), run, n_runs, len(train_ids), len(val_ids), len(fold.test_ids),
            )
            model = model_factory()
            model, info = train_one_run(model, train_dataset, val_dataset, settings, checkpoint, run_seed)

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
