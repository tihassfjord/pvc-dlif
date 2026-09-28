"""Kinetic parameters per training run, then averaged within scan.

The order matters.  Averaging the predicted curves over runs first and fitting
the average scores an ensemble, not a model: the mean curve is smoother and
closer to the truth than a typical single run, and the gain grows with the
number of runs, so conditions with more runs look better for no reason that
belongs to the condition.  Here every run's curve is fitted on its own, the
error is computed per run, and only then averaged within the scan -- the same
order as the curve metrics (metric per run -> mean over runs -> pair).

Both the signed relative error and its absolute value are averaged per run.
The mean of |error| is not |mean error|: a scan whose runs err in both
directions has a small mean error and a large typical one.

The fits with the true input do not depend on the condition, so they are done
once per scan and reused.  The Patlak break point can be

* ``"truth"`` -- fitted with the true input and used unchanged for every
  predicted input of that (scan, VOI).  A per-curve refit lets the Patlak fit
  absorb part of the input-function difference; this does not.
* ``None`` -- fitted separately for every curve (the reference implementation's
  behaviour).
* a number -- fixed for every curve.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .kinetics import fit_two_tissue, patlak

__all__ = ["kinetics_for_scan", "kinetics_per_run", "collapse_runs"]


def _relative(value: float, reference: float) -> float:
    if not np.isfinite(value) or not np.isfinite(reference) or abs(reference) < 1e-12:
        return float("nan")
    return float((value - reference) / reference)


def kinetics_for_scan(task: Mapping[str, Any]) -> list[dict[str, Any]]:
    """All conditions and runs of one scan.

    ``task`` keys: ``scan_id``, ``tissue`` ({voi: curve}), ``time_min``,
    ``truth`` (true input), ``runs`` (list of dicts with ``condition, fold, run,
    predicted``), ``models``, ``t_star`` ("truth", None or float),
    ``irreversible``.
    """
    tissue: Mapping[str, np.ndarray] = task["tissue"]
    t = np.asarray(task["time_min"], float)
    truth = np.asarray(task["truth"], float)
    models = list(task["models"])
    t_star_mode = task["t_star"]
    irreversible = bool(task["irreversible"])

    truth_fits = {}
    for voi, curve in tissue.items():
        pt = patlak(curve, truth, t, None if t_star_mode == "truth" else t_star_mode)
        tt = fit_two_tissue(curve, truth, t, irreversible=irreversible) if "two_tissue" in models else None
        truth_fits[voi] = (pt, tt)

    params = ("k1", "k2", "k3", "vb", "ki") if irreversible else ("k1", "k2", "k3", "k4", "vb", "ki")
    rows: list[dict[str, Any]] = []
    for r in task["runs"]:
        pred = np.asarray(r["predicted"], float)
        key = {"condition": r["condition"], "scan_id": task["scan_id"],
               "fold": r["fold"], "run": r["run"]}
        for voi, curve in tissue.items():
            pt, tt = truth_fits[voi]
            if "patlak" in models:
                t_star = pt.t_star_min if t_star_mode == "truth" else t_star_mode
                pp = patlak(curve, pred, t, t_star)
                rows.append({**key, "voi": voi, "model": "patlak", "parameter": "ki",
                             "value_predicted": pp.ki, "value_truth": pt.ki,
                             "relative_error": _relative(pp.ki, pt.ki),
                             "r_squared_predicted": pp.r_squared, "r_squared_truth": pt.r_squared,
                             "t_star_predicted": pp.t_star_min, "t_star_truth": pt.t_star_min})
            if tt is not None:
                tp = fit_two_tissue(curve, pred, t, irreversible=irreversible)
                for parameter in params:
                    rows.append({**key, "voi": voi, "model": "two_tissue", "parameter": parameter,
                                 "value_predicted": getattr(tp, parameter),
                                 "value_truth": getattr(tt, parameter),
                                 "relative_error": _relative(getattr(tp, parameter), getattr(tt, parameter)),
                                 "r_squared_predicted": tp.r_squared, "r_squared_truth": tt.r_squared,
                                 "converged_predicted": tp.converged, "converged_truth": tt.converged,
                                 "bounds_hit_predicted": tp.bounds_hit, "bounds_hit_truth": tt.bounds_hit,
                                 "irreversible": irreversible})
    return rows


def kinetics_per_run(tasks: Iterable[Mapping[str, Any]], jobs: int = 1) -> pd.DataFrame:
    tasks = list(tasks)
    if jobs <= 1:
        rows = [row for task in tasks for row in kinetics_for_scan(task)]
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            rows = [row for chunk in pool.map(kinetics_for_scan, tasks, chunksize=1) for row in chunk]
    return pd.DataFrame(rows)


def collapse_runs(per_run: pd.DataFrame) -> pd.DataFrame:
    """One row per (condition, scan, VOI, model, parameter): the mean over runs.

    ``relative_error`` is the mean of the per-run signed errors and
    ``abs_relative_error`` the mean of the per-run absolute errors.
    ``value_predicted`` is the mean of the per-run values.  For the two-tissue
    model, ``n_bounds_hit_predicted`` counts the runs whose fit ended on a
    bound; ``bounds_hit_predicted`` is true when any did.
    """
    keys = ["condition", "scan_id", "voi", "model", "parameter"]
    df = per_run.assign(abs_relative_error=per_run["relative_error"].abs())
    agg: dict[str, tuple[str, str]] = {
        "value_predicted": ("value_predicted", "mean"),
        "value_truth": ("value_truth", "first"),
        "relative_error": ("relative_error", "mean"),
        "abs_relative_error": ("abs_relative_error", "mean"),
        "r_squared_predicted": ("r_squared_predicted", "mean"),
        "r_squared_truth": ("r_squared_truth", "first"),
        "n_runs": ("relative_error", "size"),
    }
    for col in ("t_star_predicted", "t_star_truth"):
        if col in df:
            agg[col] = (col, "mean" if col == "t_star_predicted" else "first")
    if "bounds_hit_predicted" in df:
        df["bounds_hit_predicted"] = df["bounds_hit_predicted"].astype("boolean")
        agg["n_bounds_hit_predicted"] = ("bounds_hit_predicted", "sum")
        agg["bounds_hit_truth"] = ("bounds_hit_truth", "first")
        agg["converged_truth"] = ("converged_truth", "first")
        agg["irreversible"] = ("irreversible", "first")
    out = df.groupby(keys, dropna=False).agg(**agg).reset_index()
    if "n_bounds_hit_predicted" in out:
        is_tcm = out["model"] == "two_tissue"
        out["n_bounds_hit_predicted"] = out["n_bounds_hit_predicted"].where(is_tcm)
        out["bounds_hit_predicted"] = (out["n_bounds_hit_predicted"] > 0).where(is_tcm)
    return out


def build_tasks(predictions: pd.DataFrame, load_voi, data_root, vois: Sequence[str] | None,
                models: Sequence[str], t_star, irreversible: bool, skip_missing=True):
    """One task per scan from a frame-level prediction table."""
    for scan_id, scan in predictions.groupby("scan_id"):
        try:
            curves, voi_time = load_voi(data_root, scan_id)
        except FileNotFoundError:
            if skip_missing:
                continue
            raise
        tissue = {k: v for k, v in curves.items() if not vois or k in vois}
        runs, truth = [], None
        for (condition, fold, run), g in scan.groupby(["condition", "fold", "run"], dropna=False):
            g = g.sort_values("frame")
            this_truth = g["truth"].to_numpy()
            if truth is None:
                truth = this_truth
            elif this_truth.shape != truth.shape or not np.allclose(this_truth, truth, rtol=0, atol=1e-9):
                raise ValueError(f"{scan_id}: truth curve differs between conditions")
            runs.append({"condition": condition, "fold": fold, "run": run,
                         "predicted": g["predicted"].to_numpy()})
        yield {"scan_id": scan_id, "tissue": tissue, "time_min": voi_time, "truth": truth,
               "runs": runs, "models": list(models), "t_star": t_star,
               "irreversible": irreversible}
