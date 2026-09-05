"""Turning pipeline outputs into the tables the figures take.

Stage 08 and the GUI both draw the same figures, so the code that gathers the
inputs for them lives here and both call it.  If a figure's input changes, it
changes in one place.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "read_table",
    "frame_diagnostics_table",
    "iteration_sweep_table",
    "curves_for_scan",
    "load_predictions",
]


def read_table(stem: Path) -> pd.DataFrame | None:
    """``<stem>.parquet`` if present, else ``<stem>.csv``, else None."""
    stem = Path(stem)
    parquet = stem.with_suffix(".parquet")
    if parquet.exists():
        try:
            return pd.read_parquet(parquet)
        except Exception:                    # noqa: BLE001 - pyarrow optional
            pass
    csv = stem.with_suffix(".csv")
    return pd.read_csv(csv) if csv.exists() else None


def frame_diagnostics_table(pvc_dir: Path) -> pd.DataFrame:
    """One row per (tag, scan, frame) from every ``*.diagnostics.json`` under pvc/.

    Ratios are computed here rather than stored, so a diagnostics file written
    by an older run is still readable.
    """
    rows: list[dict[str, Any]] = []
    for diag_path in sorted(Path(pvc_dir).rglob("*.diagnostics.json")):
        payload = json.loads(diag_path.read_text(encoding="utf-8"))
        tag = diag_path.parent.name
        for row in payload.get("frames", []):
            noise_before = row.get("noise_pct_std_before") or np.nan
            peak_before = row.get("blood_peak_before") or np.nan
            rows.append({
                "tag": tag,
                "scan_id": payload.get("scan_id"),
                "frame": row.get("frame"),
                "time_min": row.get("time_min"),
                "counts_proxy": row.get("counts_proxy"),
                "noise_amplification": (row.get("noise_pct_std_after") / noise_before
                                        if noise_before else np.nan),
                "peak_recovery": (row.get("blood_peak_after") / peak_before
                                  if peak_before else np.nan),
                "negative_fraction_after": row.get("negative_fraction_after"),
            })
    columns = ["tag", "scan_id", "frame", "time_min", "counts_proxy",
               "noise_amplification", "peak_recovery", "negative_fraction_after"]
    return pd.DataFrame(rows, columns=columns)


def iteration_sweep_table(metrics: pd.DataFrame, conditions: list, metric: str = "rmse") -> pd.DataFrame:
    """Median metric per corrected condition, keyed by method and iteration count.

    ``conditions`` is ``config.conditions``; only those with a PVC method appear.
    """
    rows: list[dict[str, Any]] = []
    by_name = {c.name: c for c in conditions}
    for name in metrics["condition"].unique():
        condition = by_name.get(name)
        if condition is None or condition.pvc_method is None:
            continue
        subset = metrics[metrics["condition"] == name]
        rows.append({
            "method": condition.pvc_method,
            "iterations": condition.pvc_iterations,
            "condition": name,
            metric: float(subset[metric].median()),
        })
    return pd.DataFrame(rows, columns=["method", "iterations", "condition", metric])


def load_predictions(predictions_dir: Path) -> pd.DataFrame | None:
    """Pretrained and retrained prediction tables stacked, or None if neither exists."""
    frames = [t for t in (read_table(Path(predictions_dir) / "pretrained"),
                          read_table(Path(predictions_dir) / "retrained")) if t is not None]
    return pd.concat(frames, ignore_index=True) if frames else None


def curves_for_scan(predictions: pd.DataFrame, scan_id: str):
    """``(curves, truth)`` in the form :func:`figures.plot_curve_overlay` takes.

    Repeats (runs) of the same condition are averaged frame by frame, which is
    the same collapse stage 06 uses before scoring.
    """
    subset = predictions[predictions["scan_id"].astype(str) == str(scan_id)]
    if subset.empty:
        return {}, None
    collapsed = (subset.groupby(["condition", "frame"])
                 .agg(predicted=("predicted", "mean"), truth=("truth", "first"),
                      time_min=("time_min", "first"))
                 .reset_index())
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    truth = None
    for condition, group in collapsed.groupby("condition"):
        group = group.sort_values("frame")
        curves[str(condition)] = (group["time_min"].to_numpy(), group["predicted"].to_numpy())
        if truth is None:
            truth = (group["time_min"].to_numpy(), group["truth"].to_numpy())
    return curves, truth
