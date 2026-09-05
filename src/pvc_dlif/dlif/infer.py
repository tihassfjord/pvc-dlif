"""Producing one predicted input function per scan, per condition.

Two prediction modes, and the difference between them is the point of the
study:

``pretrained``
    One deployed model is applied to every scan under every input
    representation.  Comparing corrected against uncorrected inputs here
    measures the robustness of the deployed model to the distribution shift
    that PVC introduces -- it is not a test of the main hypothesis.

``retrained``
    Each scan is predicted only by models whose training fold excluded it, so
    there is no leakage, and each of the repeats for that fold gives one
    prediction.  Keeping every repeat rather than only their mean is what makes
    the bias/variance decomposition possible later.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..logging_utils import get_logger
from ..data import pkl_io
from .adapter import predict_series
from .dataset import Fold

LOGGER = get_logger(__name__)

__all__ = ["Prediction", "predict_pretrained", "predict_retrained", "predictions_to_frame", "save_predictions"]


@dataclass
class Prediction:
    """One predicted input function, with the ground truth it is scored against."""

    scan_id: str
    condition: str
    model: str
    fold: int | None
    run: int | None
    time_min: np.ndarray
    predicted: np.ndarray
    truth: np.ndarray

    def to_records(self) -> list[dict[str, Any]]:
        return [
            {
                "scan_id": self.scan_id,
                "condition": self.condition,
                "model": self.model,
                "fold": self.fold,
                "run": self.run,
                "frame": int(i),
                "time_min": float(self.time_min[i]),
                "predicted": float(self.predicted[i]),
                "truth": float(self.truth[i]),
            }
            for i in range(len(self.predicted))
        ]


def _load_pair(
    data_root: Path,
    aif_root: Path,
    scan_id: str,
    img_shape: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image, _ = pkl_io.load_img(data_root, scan_id, img_shape)
    truth, times = pkl_io.load_aif(aif_root, scan_id)
    return image, truth, times


def _align(predicted: np.ndarray, truth: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trim to the common length; a mismatch is a data problem worth surfacing."""
    n = min(len(predicted), len(truth), len(times))
    if not (len(predicted) == len(truth) == len(times)):
        LOGGER.warning(
            "Length mismatch (pred %d, truth %d, time %d); using the first %d frames",
            len(predicted), len(truth), len(times), n,
        )
    return predicted[:n], truth[:n], times[:n]


def predict_pretrained(
    model,
    data_root: Path,
    aif_root: Path,
    scan_ids: Sequence[str],
    condition: str,
    img_shape: Sequence[int] = (96, 48, 48),
    device: str = "cpu",
    add_average: bool = False,
    spec: Any = None,
) -> list[Prediction]:
    """Apply one fixed model to every scan under one input representation.

    ``spec`` is the model's own input requirement, recovered from its weights;
    when given it decides the channel layout and checks the spatial shape, so a
    mismatch between the model and the input tree is an error rather than a
    quietly wrong baseline.
    """
    predictions: list[Prediction] = []
    for scan_id in scan_ids:
        image, truth, times = _load_pair(data_root, aif_root, scan_id, img_shape)
        curve = predict_series(model, image, device=device, add_average=add_average, spec=spec)
        curve, truth, times = _align(curve, truth, times)
        predictions.append(
            Prediction(
                scan_id=scan_id, condition=condition, model="pretrained",
                fold=None, run=None, time_min=times, predicted=curve, truth=truth,
            )
        )
    LOGGER.info("%s: %d predictions from the pretrained model", condition, len(predictions))
    return predictions


def predict_retrained(
    model_factory,
    checkpoint_root: Path,
    data_root: Path,
    aif_root: Path,
    folds: Sequence[Fold],
    condition: str,
    img_shape: Sequence[int] = (96, 48, 48),
    device: str = "cpu",
    add_average: bool = False,
    n_runs: int | None = None,
    spec: Any = None,
) -> list[Prediction]:
    """Predict every held-out scan with the models trained without it.

    ``checkpoint_root`` is the directory ``train_condition`` wrote, containing
    ``fold_XX/run_YY/model.pt``.
    """
    import torch

    checkpoint_root = Path(checkpoint_root)
    predictions: list[Prediction] = []

    for fold in folds:
        fold_dir = checkpoint_root / f"fold_{fold.index:02d}"
        if not fold_dir.exists():
            LOGGER.warning("No checkpoints for fold %d under %s", fold.index, checkpoint_root)
            continue

        run_dirs = sorted(d for d in fold_dir.iterdir() if d.is_dir() and d.name.startswith("run_"))
        if n_runs is not None:
            run_dirs = run_dirs[:n_runs]

        # Load the held-out scans once and reuse them across the repeats.
        held_out = {
            scan_id: _load_pair(data_root, aif_root, scan_id, img_shape)
            for scan_id in fold.test_ids
        }

        for run_dir in run_dirs:
            checkpoint = run_dir / "model.pt"
            if not checkpoint.exists():
                continue
            payload = torch.load(str(checkpoint), map_location=device, weights_only=False)
            model = model_factory()
            model.load_state_dict(payload["state_dict"] if "state_dict" in payload else payload)
            model.to(device)
            model.eval()
            run_index = int(run_dir.name.split("_")[-1])

            for scan_id, (image, truth, times) in held_out.items():
                curve = predict_series(model, image, device=device, add_average=add_average, spec=spec)
                curve, truth_a, times_a = _align(curve, truth, times)
                predictions.append(
                    Prediction(
                        scan_id=scan_id, condition=condition, model="retrained",
                        fold=fold.index, run=run_index,
                        time_min=times_a, predicted=curve, truth=truth_a,
                    )
                )

    LOGGER.info(
        "%s: %d predictions from %d folds", condition, len(predictions), len(folds)
    )
    return predictions


def predictions_to_frame(predictions: Iterable[Prediction]):
    """Flatten predictions into a tidy, frame-level DataFrame."""
    import pandas as pd

    records: list[dict[str, Any]] = []
    for prediction in predictions:
        records.extend(prediction.to_records())
    return pd.DataFrame.from_records(records)


def save_predictions(predictions: Iterable[Prediction], path: Path) -> Path:
    """Write the tidy prediction table, parquet when available, else CSV."""
    frame = predictions_to_frame(predictions)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_parquet(path, index=False)
    except Exception:  # noqa: BLE001 - pyarrow is optional
        path = path.with_suffix(".csv")
        frame.to_csv(path, index=False)
    LOGGER.info("Wrote %d prediction rows to %s", len(frame), path)
    return path
