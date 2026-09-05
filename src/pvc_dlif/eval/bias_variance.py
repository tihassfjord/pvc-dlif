"""Decomposing squared error into bias and variance.

The project description makes a specific prediction: RL and RVC should reduce
bias (spill-out is undone, so the blood-pool signal recovers) while raising
variance (deconvolution amplifies noise).  If both move, an aggregate RMSE can
stay flat and hide a substantial change in each -- which is exactly the kind of
null result that would be misread.

For one scan and one frame, over the ``R`` repeats of a condition::

    E[(y - g)^2]  =  (mean_r y_r - g)^2   +   Var_r(y_r)
                     \\_____ bias^2 _____/     \\__ variance __/

The variance term needs repeats, so it is only defined for the retrained
conditions, which are trained ``n_runs`` times per fold.  For a single
deterministic model the variance is zero by construction and the decomposition
degenerates to bias^2; the functions below say so rather than reporting a
misleading zero.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Sequence

import numpy as np

__all__ = ["BiasVariance", "decompose", "decompose_frame", "decompose_by_condition"]


@dataclass
class BiasVariance:
    """Bias/variance decomposition of mean squared error."""

    mse: float
    bias_squared: float
    variance: float
    signed_bias: float
    n_repeats: int
    per_frame_bias_squared: list[float]
    per_frame_variance: list[float]

    @property
    def is_estimable(self) -> bool:
        """Whether the variance term is estimated from real repeats."""
        return self.n_repeats >= 2

    @property
    def variance_share(self) -> float:
        """Fraction of MSE attributable to variance."""
        return float(self.variance / self.mse) if self.mse > 1e-15 else float("nan")

    def to_dict(self, with_per_frame: bool = False) -> dict[str, Any]:
        payload = asdict(self)
        if not with_per_frame:
            payload.pop("per_frame_bias_squared")
            payload.pop("per_frame_variance")
        payload["variance_share"] = self.variance_share
        payload["estimable"] = self.is_estimable
        return payload


def decompose(predictions: Sequence[Sequence[float]] | np.ndarray, truth: Sequence[float]) -> BiasVariance:
    """Decompose the error of ``R`` repeated predictions of one curve.

    Parameters
    ----------
    predictions:
        Array of shape ``(R, T)`` -- one row per repeat.
    truth:
        The arterial ground-truth curve, length ``T``.
    """
    y = np.atleast_2d(np.asarray(predictions, dtype=np.float64))
    g = np.asarray(truth, dtype=np.float64).ravel()

    n_repeats, n_frames = y.shape
    if n_frames != g.size:
        n = min(n_frames, g.size)
        y, g = y[:, :n], g[:n]

    mean_prediction = y.mean(axis=0)
    per_frame_bias = mean_prediction - g
    per_frame_bias_sq = per_frame_bias ** 2
    # Population variance across repeats: it is the variance term of the
    # decomposition, not a sample estimate of a wider population.
    per_frame_var = y.var(axis=0, ddof=0) if n_repeats > 1 else np.zeros_like(g)

    mse = float(np.mean((y - g[None, :]) ** 2))

    return BiasVariance(
        mse=mse,
        bias_squared=float(np.mean(per_frame_bias_sq)),
        variance=float(np.mean(per_frame_var)),
        signed_bias=float(np.mean(per_frame_bias)),
        n_repeats=int(n_repeats),
        per_frame_bias_squared=[float(v) for v in per_frame_bias_sq],
        per_frame_variance=[float(v) for v in per_frame_var],
    )


def decompose_frame(prediction_table: Any, with_per_frame: bool = False):
    """Decompose every ``(condition, scan)`` group of a tidy prediction table.

    ``prediction_table`` is the frame-level table from
    :func:`pvc_dlif.dlif.infer.predictions_to_frame`, with one row per
    ``(scan, condition, run, frame)``.
    """
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for (condition, scan_id), group in prediction_table.groupby(["condition", "scan_id"], dropna=False):
        wide = group.pivot_table(index="run", columns="frame", values="predicted", dropna=False)
        # A pretrained condition has no runs; pivot_table then yields one row.
        if wide.empty:
            ordered = group.sort_values("frame")
            matrix = ordered["predicted"].to_numpy()[None, :]
            truth = ordered["truth"].to_numpy()
        else:
            matrix = wide.to_numpy()
            truth = (
                group.sort_values("frame")
                .drop_duplicates("frame")["truth"]
                .to_numpy()
            )
        result = decompose(matrix, truth)
        rows.append(
            {
                "condition": condition,
                "scan_id": scan_id,
                **result.to_dict(with_per_frame=with_per_frame),
            }
        )
    return pd.DataFrame(rows)


def decompose_by_condition(prediction_table: Any):
    """Aggregate the decomposition to one row per condition.

    Averaging over scans is the right aggregation here: each scan is one paired
    observation, and every scan contributes to every condition.
    """
    import pandas as pd

    per_scan = decompose_frame(prediction_table)
    if per_scan.empty:
        return per_scan

    aggregated = (
        per_scan.groupby("condition")
        .agg(
            n_scans=("scan_id", "nunique"),
            n_repeats=("n_repeats", "median"),
            mse=("mse", "mean"),
            bias_squared=("bias_squared", "mean"),
            variance=("variance", "mean"),
            signed_bias=("signed_bias", "mean"),
        )
        .reset_index()
    )
    aggregated["variance_share"] = aggregated["variance"] / aggregated["mse"].where(aggregated["mse"] > 0)
    aggregated["estimable"] = aggregated["n_repeats"] >= 2
    aggregated["residual"] = aggregated["mse"] - (aggregated["bias_squared"] + aggregated["variance"])
    return aggregated
