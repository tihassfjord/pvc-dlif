"""Curve-level error metrics for predicted input functions.

The metrics are the ones a kinetic-modelling reader cares about: how far the
whole curve is from the arterial ground truth (RMSE), how much tracer the curve
delivers (area under the curve), and whether the early bolus is right (peak
height, time to peak, peak width).  The peak matters out of proportion to its
duration -- it is short, noisy, and it is what a two-tissue-compartment fit is
most sensitive to.

Everything is computed on the sampled time grid with trapezoidal integration,
using the real frame times rather than frame indices, because the frames are
far from uniformly spaced (8 s early, 5 min late).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = ["CurveMetrics", "compute_metrics", "auc", "peak_split_index", "metrics_frame"]


def _as_array(values: Sequence[float] | np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).ravel()


def auc(curve: Sequence[float], time_min: Sequence[float], start: int = 0, stop: int | None = None) -> float:
    """Trapezoidal area under a curve segment, in SUV*minutes."""
    y = _as_array(curve)
    t = _as_array(time_min)
    stop = len(y) if stop is None else int(stop)
    start = max(0, int(start))
    stop = min(len(y), stop)
    if stop - start < 2:
        return 0.0
    return float(np.trapezoid(y[start:stop], t[start:stop]))


def peak_split_index(curve: Sequence[float], time_min: Sequence[float]) -> int:
    """Where the bolus peak ends and the tail begins.

    Uses the first minimum of the first derivative after the maximum: the point
    where the washout stops steepening.  This gives a reproducible split for the
    peak/tail area ratio without hand-picking a time.
    """
    y = _as_array(curve)
    if y.size < 4:
        return y.size
    peak = int(np.argmax(y))
    if peak >= y.size - 2:
        return y.size
    gradient = np.gradient(y, _as_array(time_min))
    after = gradient[peak:]
    # First index after the steepest descent where the slope stops decreasing.
    steepest = int(np.argmin(after))
    tail_start = peak + steepest
    return int(min(max(tail_start + 1, peak + 1), y.size))


def _fwhm(curve: Sequence[float], time_min: Sequence[float]) -> float:
    """Full width at half maximum of the bolus peak, in minutes.

    Interpolates the half-maximum crossings rather than snapping to the nearest
    frame, because the early frames are 5-8 s apart and rounding to a frame
    would quantise the width into a handful of values.
    """
    y = _as_array(curve)
    t = _as_array(time_min)
    if y.size < 3:
        return float("nan")
    peak = int(np.argmax(y))
    half = y[peak] / 2.0
    if not np.isfinite(half) or half <= 0:
        return float("nan")

    left = np.nan
    for i in range(peak, 0, -1):
        if y[i - 1] <= half <= y[i]:
            span = y[i] - y[i - 1]
            frac = 0.5 if abs(span) < 1e-12 else (half - y[i - 1]) / span
            left = t[i - 1] + frac * (t[i] - t[i - 1])
            break

    right = np.nan
    for i in range(peak, y.size - 1):
        if y[i + 1] <= half <= y[i]:
            span = y[i] - y[i + 1]
            frac = 0.5 if abs(span) < 1e-12 else (y[i] - half) / span
            right = t[i] + frac * (t[i + 1] - t[i])
            break

    if not np.isfinite(left) or not np.isfinite(right):
        return float("nan")
    return float(right - left)


def _ccc(a: np.ndarray, b: np.ndarray) -> float:
    """Lin's concordance correlation coefficient."""
    if a.size < 2:
        return float("nan")
    va, vb = a.var(), b.var()
    ma, mb = a.mean(), b.mean()
    cov = float(np.mean((a - ma) * (b - mb)))
    denom = va + vb + (ma - mb) ** 2
    return float(2 * cov / denom) if denom > 1e-12 else float("nan")


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


@dataclass
class CurveMetrics:
    """Error metrics for one predicted curve against one ground-truth curve."""

    rmse: float
    nrmse: float
    mae: float
    max_error: float
    r2: float
    pearson_r: float
    ccc: float
    bias: float                # mean signed error: positive = over-prediction
    auc_pred: float
    auc_truth: float
    auc_ratio: float
    auc_peak_pred: float
    auc_peak_truth: float
    auc_tail_pred: float
    auc_tail_truth: float
    peak_height_pred: float
    peak_height_truth: float
    peak_height_ratio: float
    peak_time_pred: float
    peak_time_truth: float
    peak_time_error: float
    fwhm_pred: float
    fwhm_truth: float
    rmse_early: float
    rmse_late: float
    n_frames: int

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def compute_metrics(
    predicted: Sequence[float],
    truth: Sequence[float],
    time_min: Sequence[float],
    early_late_split_min: float = 2.5,
) -> CurveMetrics:
    """Score one predicted input function against arterial sampling.

    ``early_late_split_min`` separates the low-count bolus frames from the
    high-count later frames.  The recovery-noise trade-off is expected to differ
    between the two, so the split RMSE is reported alongside the aggregate one.
    """
    y = _as_array(predicted)
    g = _as_array(truth)
    t = _as_array(time_min)

    n = min(y.size, g.size, t.size)
    y, g, t = y[:n], g[:n], t[:n]
    if n == 0:
        raise ValueError("empty curve")

    residual = y - g
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    truth_std = float(g.std())
    ss_tot = float(np.sum((g - g.mean()) ** 2))

    split = peak_split_index(g, t)
    early = t < early_late_split_min
    late = ~early

    auc_pred_total = auc(y, t)
    auc_truth_total = auc(g, t)

    return CurveMetrics(
        rmse=rmse,
        nrmse=float(rmse / truth_std) if truth_std > 1e-12 else float("nan"),
        mae=float(np.mean(np.abs(residual))),
        max_error=float(np.max(np.abs(residual))),
        r2=float(1.0 - np.sum(residual ** 2) / ss_tot) if ss_tot > 1e-12 else float("nan"),
        pearson_r=_pearson(y, g),
        ccc=_ccc(y, g),
        bias=float(np.mean(residual)),
        auc_pred=auc_pred_total,
        auc_truth=auc_truth_total,
        auc_ratio=float(auc_pred_total / auc_truth_total) if abs(auc_truth_total) > 1e-12 else float("nan"),
        auc_peak_pred=auc(y, t, 0, split),
        auc_peak_truth=auc(g, t, 0, split),
        auc_tail_pred=auc(y, t, split, None),
        auc_tail_truth=auc(g, t, split, None),
        peak_height_pred=float(np.max(y)),
        peak_height_truth=float(np.max(g)),
        peak_height_ratio=(
            float(np.max(y) / np.max(g)) if abs(float(np.max(g))) > 1e-12 else float("nan")
        ),
        peak_time_pred=float(t[int(np.argmax(y))]),
        peak_time_truth=float(t[int(np.argmax(g))]),
        peak_time_error=float(t[int(np.argmax(y))] - t[int(np.argmax(g))]),
        fwhm_pred=_fwhm(y, t),
        fwhm_truth=_fwhm(g, t),
        rmse_early=float(np.sqrt(np.mean(residual[early] ** 2))) if early.any() else float("nan"),
        rmse_late=float(np.sqrt(np.mean(residual[late] ** 2))) if late.any() else float("nan"),
        n_frames=int(n),
    )


def metrics_frame(predictions: Any, early_late_split_min: float = 2.5):
    """Score a whole collection of predictions into a tidy DataFrame.

    Accepts either an iterable of ``Prediction`` objects or the frame-level
    table produced by :func:`pvc_dlif.dlif.infer.predictions_to_frame`.
    """
    import pandas as pd

    rows: list[dict[str, Any]] = []

    if isinstance(predictions, pd.DataFrame):
        keys = ["scan_id", "condition", "model", "fold", "run"]
        for key_values, group in predictions.groupby(keys, dropna=False):
            group = group.sort_values("frame")
            metrics = compute_metrics(
                group["predicted"].to_numpy(),
                group["truth"].to_numpy(),
                group["time_min"].to_numpy(),
                early_late_split_min,
            )
            rows.append({**dict(zip(keys, key_values)), **metrics.to_dict()})
    else:
        for prediction in predictions:
            metrics = compute_metrics(
                prediction.predicted, prediction.truth, prediction.time_min, early_late_split_min
            )
            rows.append(
                {
                    "scan_id": prediction.scan_id,
                    "condition": prediction.condition,
                    "model": prediction.model,
                    "fold": prediction.fold,
                    "run": prediction.run,
                    **metrics.to_dict(),
                }
            )

    return pd.DataFrame(rows)
