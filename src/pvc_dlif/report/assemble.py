"""Turning pipeline outputs into the tables the figures take.

Stage 08 and the GUI both draw the same figures, so the code that gathers the
inputs for them lives here and both call it.  If a figure's input changes, it
changes in one place.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

__all__ = [
    "read_table",
    "frame_diagnostics_table",
    "iteration_sweep_table",
    "psf_sweep_table",
    "paired_sweep_table",
    "condition_arms",
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


def condition_arms(conditions: list, reference: str,
                   labels: Mapping[str, str] | None = None) -> list[dict[str, Any]]:
    """Split the conditions into groups that may share a figure axis.

    Three questions are being asked at once in this study, and putting their
    conditions on one axis against one reference invites a comparison that is
    not supported.  The deployed model and the retrained model differ in
    architecture, parameter count, input channels and training pipeline, so a
    deployed condition drawn beside a retrained one against a retrained
    reference shows that architectural difference as if it were an effect of
    the correction.

    The groups, each with the only reference it can legitimately be read
    against:

    ``deployed``
        The published model applied to each input representation, against the
        same model on uncorrected input.  Answers: how does the model
        currently in use behave on corrected images?
    ``retrained``
        Models fitted on their own input representation, against the retrained
        baseline.  Answers the main hypothesis, and is the primary endpoint.
    ``motion``
        Models fitted on a motion-corrected representation, against the same
        retrained baseline.  Registration is a different intervention from
        deconvolution, so it is a separate question and a separate correction
        family.  Were it folded into ``retrained``, adding a motion condition
        would move the main hypothesis' adjusted p-value without anything about
        the PVC experiment having changed -- which is the failure the arms
        exist to prevent, seen once already when PSF conditions moved it from
        0.26 to 0.36.
    ``fixed_weights`` (one group per set of borrowed weights)
        One set of retrained weights applied to several inputs, against the
        condition those weights were fitted by.  Answers: how much does the
        input alone move the prediction?

    ``labels`` maps ``"pretrained"`` and ``"retrained"`` to the name of the
    network each arm uses, and those names go into the question string and so
    into every figure title.  Without them a figure says only "deployed" and
    "retrained", which leaves a reader unable to tell which of two different
    networks produced it -- and the difference between them is precisely what
    must not be read across arms.

    Returns one entry per non-empty group, as ``{"name", "reference",
    "conditions", "question"}``.
    """
    by_name = {c.name: c for c in conditions}
    names = dict(labels or {})
    deployed_label = names.get("pretrained", "the published model")
    retrained_label = names.get("retrained", "the retrained model")

    def pick_reference(candidates: list) -> str | None:
        """The uncorrected, unregistered condition among ``candidates``."""
        for c in candidates:
            if c.pvc_method is None and not c.motion:
                return c.name
        return None

    deployed = [c for c in conditions if c.model != "retrained"]
    borrowed = [c for c in conditions if c.model == "retrained" and c.borrows_checkpoints]
    fitted = [c for c in conditions
              if c.model == "retrained" and not c.borrows_checkpoints]
    # Registration and deconvolution are different interventions, so the
    # conditions trained on a registered representation form their own family.
    # Both read against the same unregistered, uncorrected baseline: the
    # question each answers is "does this preprocessing step help", and they
    # answer it about different steps.
    retrained = [c for c in fitted if not c.motion]
    motion = [c for c in fitted if c.motion]

    main_reference = reference if reference in by_name else pick_reference(retrained)
    groups = [
        ("deployed", deployed, pick_reference(deployed),
         f"{deployed_label} on each input representation"),
        ("retrained", retrained, main_reference,
         f"{retrained_label}, each fitted on its own input"),
        # The reference is the unregistered baseline, which is not a member of
        # this group; ``condition_arms`` drops it from the target list below,
        # so the family is the motion conditions alone.
        ("motion", [*motion, *( [by_name[main_reference]] if main_reference in by_name else [] )],
         main_reference,
         f"{retrained_label}, fitted on motion-corrected input"),
    ]

    # A borrowed condition is read against the condition it borrowed from, not
    # against the study's primary reference.  Those are the same thing when the
    # weights come from the baseline, but a condition that borrows a
    # PVC-trained model and is shown uncorrected images has to be compared with
    # that PVC-trained model: same weights, different input.  Comparing it with
    # the baseline instead would vary the weights and the input at once.
    by_source: dict[str, list] = {}
    for c in borrowed:
        by_source.setdefault(str(c.checkpoints_from), []).append(c)
    for source, members in by_source.items():
        if source not in by_name:
            continue
        name = "fixed_weights" if source == reference else f"fixed_weights_{source}"
        groups.append((name, members, source,
                       f"{retrained_label}: the weights of {source}, on several inputs"))

    out: list[dict[str, Any]] = []
    for name, members, ref, question in groups:
        if ref is None:
            continue
        targets = [c.name for c in members if c.name != ref]
        if not targets:
            continue
        out.append({"name": name, "reference": ref,
                    "conditions": targets, "question": question})
    return out


def _sweep_key(condition) -> tuple:
    """What has to be equal for two conditions to lie on the same sweep line.

    Everything about a condition except the one axis being swept: the method,
    the model, whether the input was motion-corrected, and whether the model
    was fitted on this input or borrowed from another condition.  Conditions
    that differ in any of these are different curves, not further points on
    one, and connecting them would draw a line through unrelated results.
    """
    return (condition.pvc_method, condition.model, bool(condition.motion),
            getattr(condition, "motion_order", "mc_then_pvc") if condition.motion else "",
            getattr(condition, "checkpoints_from", None) is not None)


def _label(condition) -> str:
    parts = [str(condition.model)]
    if condition.motion:
        order = getattr(condition, "motion_order", "mc_then_pvc")
        parts.append("mc" if order == "mc_then_pvc" else "pvc-first")
    if getattr(condition, "checkpoints_from", None) is not None:
        parts.append("borrowed")
    return parts[0] if len(parts) == 1 else f"{parts[0]} ({', '.join(parts[1:])})"


def iteration_sweep_table(metrics: pd.DataFrame, conditions: list, metric: str = "rmse") -> pd.DataFrame:
    """Median metric per corrected condition, against the iteration count.

    PSF-scaled conditions are left out: they sit at the primary iteration count
    but differ in the assumed resolution, so they would stack on one x.  They
    belong to :func:`psf_sweep_table`.  Everything else that distinguishes a
    condition goes into ``model``, which is the series label, so that separate
    curves stay separate.
    """
    rows: list[dict[str, Any]] = []
    by_name = {c.name: c for c in conditions}
    for name in metrics["condition"].unique():
        condition = by_name.get(name)
        if condition is None or condition.pvc_method is None:
            continue
        if getattr(condition, "psf_scale", None) is not None:
            continue
        subset = metrics[metrics["condition"] == name]
        rows.append({
            "method": condition.pvc_method,
            "model": _label(condition),
            "iterations": condition.pvc_iterations,
            "condition": name,
            metric: float(subset[metric].median()),
        })
    table = pd.DataFrame(rows, columns=["method", "model", "iterations", "condition", metric])
    if table.empty:
        return table
    return table.sort_values(["method", "model", "iterations"]).reset_index(drop=True)


def psf_sweep_table(metrics: pd.DataFrame, conditions: list, metric: str = "rmse") -> pd.DataFrame:
    """Median metric against the PSF scale the correction assumed.

    The measured PSF is one number with uncertainty and with assumptions behind
    it -- static point source, centre of the field, spatially invariant.  This
    table carries that uncertainty through to the metric: the spread across
    scales is how much the result depends on the PSF being right.

    The unscaled condition that a scaled one perturbs is its 1.0 point, so it
    appears here as well as in the iteration sweep.  Unscaled conditions that
    nothing perturbs -- the rest of the iteration grid -- do not.
    """
    by_name = {c.name: c for c in conditions}
    scored = [by_name[n] for n in metrics["condition"].unique()
              if by_name.get(n) is not None and by_name[n].pvc_method is not None]

    # Which (method, model, motion, borrowed, iterations) combinations actually
    # have a scaled variant; only those get their unscaled anchor point.
    perturbed = {(*_sweep_key(c), c.pvc_iterations) for c in scored
                 if getattr(c, "psf_scale", None) is not None}

    rows: list[dict[str, Any]] = []
    for condition in scored:
        scale = getattr(condition, "psf_scale", None)
        if scale is None:
            if (*_sweep_key(condition), condition.pvc_iterations) not in perturbed:
                continue
            scale = 1.0
        subset = metrics[metrics["condition"] == condition.name]
        rows.append({
            "method": condition.pvc_method,
            "model": _label(condition),
            "psf_scale": float(scale),
            "condition": condition.name,
            metric: float(subset[metric].median()),
        })
    table = pd.DataFrame(rows, columns=["method", "model", "psf_scale", "condition", metric])
    if table.empty:
        return table
    return table.sort_values(["method", "model", "psf_scale"]).reset_index(drop=True)


def paired_sweep_table(metrics: pd.DataFrame, conditions: list, metric: str,
                       axis: str = "iterations") -> pd.DataFrame:
    """A sweep of *paired* change, anchored at each series' first point.

    A sweep of marginal medians answers "what is the median value at each
    setting", which is not the question a paired design asks and can hide a
    real effect entirely.  Peak height across the iteration grid is the example
    that forced this function into existence: the marginal medians are 0.890 at
    every iteration count, while the paired difference between the highest and
    lowest count is negative in 58 of 70 scans at p = 1e-8.  Rounding, not
    absence, made the first version look flat.

    Here each scan is compared against *its own* value at the lowest setting of
    the axis, and the median of those differences is what is plotted.  Zero is
    therefore "no change from the first setting" and the figure shows the
    quantity the statistics test.
    """
    base = (iteration_sweep_table if axis == "iterations" else psf_sweep_table)(
        metrics, conditions, metric=metric)
    if not len(base):
        return base

    per_scan = metrics.groupby(["condition", "scan_id"])[metric].mean()
    rows: list[dict[str, Any]] = []
    for (method, model), group in base.groupby(["method", "model"]):
        group = group.sort_values(axis)
        anchor = group["condition"].iloc[0]
        anchor_values = per_scan.loc[anchor] if anchor in per_scan.index.levels[0] else None
        if anchor_values is None:
            continue
        for _, row in group.iterrows():
            values = per_scan.loc[row["condition"]]
            shared = anchor_values.index.intersection(values.index)
            if not len(shared):
                continue
            delta = (values.loc[shared] - anchor_values.loc[shared]).median()
            rows.append({"method": method, "model": model, axis: row[axis],
                         "condition": row["condition"], "anchor": anchor,
                         metric: float(delta)})
    return pd.DataFrame(rows, columns=["method", "model", axis, "condition", "anchor", metric])


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
