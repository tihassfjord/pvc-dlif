"""Frame-level analysis: where along the curve does PVC help, and where does it hurt?

Aggregate curve metrics can hide the mechanism.  Count statistics vary by more
than an order of magnitude across a scan -- 5 s frames during the bolus, 5 min
frames at the end -- and deconvolution behaves very differently at the two
extremes.  The phantom work characterised the recovery-noise trade-off at a
single, high count level; this module is where that trade-off is traced across
the time-activity curve.

Two outputs:

* :func:`frame_error_table` -- per-frame signed and absolute error for every
  condition, with the frame's timing and a count proxy attached, so error can
  be regressed on count level rather than eyeballed.
* :func:`failure_modes` -- the scans and frames where a corrected condition is
  *worse* than the reference, ranked, so the failure modes named in the project
  description (early low-count frames, blood-pool structures near the
  resolution limit) can be confirmed or ruled out rather than assumed.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ..logging_utils import get_logger

LOGGER = get_logger(__name__)

__all__ = ["frame_error_table", "bin_frames", "summarise_by_bin", "failure_modes", "attach_diagnostics"]


def bin_frames(time_min: Sequence[float], edges: Sequence[float]) -> list[str]:
    """Label each frame with the time bin it falls into."""
    t = np.asarray(time_min, dtype=float)
    bounds = list(edges)
    labels: list[str] = []
    for value in t:
        label = f">={bounds[-1]:g}"
        for low, high in zip(bounds[:-1], bounds[1:]):
            if low <= value < high:
                label = f"{low:g}-{high:g} min"
                break
        labels.append(label)
    return labels


def frame_error_table(prediction_table: Any, frame_bins_min: Sequence[float] | None = None):
    """Per-frame errors from a tidy prediction table.

    Adds the signed error (which carries the direction of the bias), the
    absolute error, the squared error, and a time-bin label.
    """
    import pandas as pd

    table = prediction_table.copy()
    table["error"] = table["predicted"] - table["truth"]
    table["abs_error"] = table["error"].abs()
    table["squared_error"] = table["error"] ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        table["relative_error"] = np.where(
            table["truth"].abs() > 1e-9, table["error"] / table["truth"], np.nan
        )

    if frame_bins_min:
        table["time_bin"] = bin_frames(table["time_min"].to_numpy(), frame_bins_min)

    return table


def summarise_by_bin(frame_table: Any, by: Sequence[str] = ("condition", "time_bin")):
    """Average per-frame error within each time bin and condition."""
    import pandas as pd

    columns = [c for c in by if c in frame_table.columns]
    if not columns:
        raise ValueError(f"none of {by} are columns of the frame table")

    grouped = (
        frame_table.groupby(list(columns), dropna=False)
        .agg(
            n_frames=("squared_error", "size"),
            n_scans=("scan_id", "nunique"),
            rmse=("squared_error", lambda s: float(np.sqrt(np.mean(s)))),
            mean_signed_error=("error", "mean"),
            median_signed_error=("error", "median"),
            mean_abs_error=("abs_error", "mean"),
            mean_truth=("truth", "mean"),
        )
        .reset_index()
    )
    return grouped


def attach_diagnostics(frame_table: Any, diagnostics: Mapping[str, Any], condition: str | None = None):
    """Join the PVC per-frame diagnostics onto the per-frame error table.

    ``diagnostics`` maps ``scan_id`` to the ``frames`` list written by the PVC
    runner.  The join is what lets noise amplification and peak recovery -- the
    two halves of the recovery-noise trade-off -- be plotted against the error
    they produce, instead of being reported separately and left to the reader.
    """
    import pandas as pd

    records: list[dict[str, Any]] = []
    for scan_id, payload in diagnostics.items():
        frames = payload.get("frames", payload) if isinstance(payload, Mapping) else payload
        for row in frames:
            records.append(
                {
                    "scan_id": scan_id,
                    "frame": int(row["frame"]),
                    "counts_proxy": row.get("counts_proxy"),
                    "duration_s": row.get("duration_s"),
                    "noise_pct_std_before": row.get("noise_pct_std_before"),
                    "noise_pct_std_after": row.get("noise_pct_std_after"),
                    "blood_peak_before": row.get("blood_peak_before"),
                    "blood_peak_after": row.get("blood_peak_after"),
                    "negative_fraction_after": row.get("negative_fraction_after"),
                }
            )
    if not records:
        return frame_table

    diag = pd.DataFrame.from_records(records)
    diag["noise_amplification"] = diag["noise_pct_std_after"] / diag["noise_pct_std_before"].where(
        diag["noise_pct_std_before"] > 0
    )
    diag["peak_recovery"] = diag["blood_peak_after"] / diag["blood_peak_before"].where(
        diag["blood_peak_before"] > 0
    )

    subset = frame_table if condition is None else frame_table[frame_table["condition"] == condition]
    return subset.merge(diag, on=["scan_id", "frame"], how="left")


def failure_modes(
    metrics_table: Any,
    reference: str,
    metric: str = "rmse",
    aggregate_runs: str = "mean",
    top_n: int = 15,
):
    """Rank the scans where a condition does worse than the reference.

    Returns one row per ``(condition, scan)`` with the change in the metric and
    a ``degraded`` flag, sorted worst first.  The point is to make the failure
    cases nameable -- which animals, which groups -- rather than leaving them
    inside an average.
    """
    import pandas as pd

    collapsed = (
        metrics_table.groupby(["condition", "scan_id"], dropna=False)[metric]
        .agg(aggregate_runs)
        .reset_index()
    )
    wide = collapsed.pivot(index="scan_id", columns="condition", values=metric)
    if reference not in wide.columns:
        raise ValueError(f"reference {reference!r} not in the table")

    rows: list[dict[str, Any]] = []
    for condition in wide.columns:
        if condition == reference:
            continue
        delta = wide[condition] - wide[reference]
        for scan_id, change in delta.items():
            rows.append(
                {
                    "condition": condition,
                    "scan_id": scan_id,
                    "metric": metric,
                    "reference_value": float(wide.loc[scan_id, reference]),
                    "condition_value": float(wide.loc[scan_id, condition]),
                    "delta": float(change),
                    "relative_delta": (
                        float(change / wide.loc[scan_id, reference])
                        if abs(wide.loc[scan_id, reference]) > 1e-12 else float("nan")
                    ),
                    "degraded": bool(change > 0),
                }
            )

    frame = pd.DataFrame(rows).sort_values("delta", ascending=False)
    if top_n:
        return frame.head(top_n * max(1, len(wide.columns) - 1))
    return frame
