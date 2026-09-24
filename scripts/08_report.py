"""Stage 08 -- build the thesis figures and LaTeX tables.

Reads the result tables from stage 06 and writes everything into
``<work>/report``.  The LaTeX fragments are meant to be ``\\input{}`` into the
thesis, so the numbers in the text and the numbers in the results directory
cannot drift apart.

    python scripts/08_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from _common import base_parser, start

from pvc_dlif.logging_utils import get_logger, write_provenance
from pvc_dlif.report import assemble, figures, tables

LOGGER = get_logger("stage.report")


def _read(path: Path) -> pd.DataFrame | None:
    for candidate in (path.with_suffix(".parquet"), path.with_suffix(".csv")):
        if candidate.exists():
            return pd.read_parquet(candidate) if candidate.suffix == ".parquet" else pd.read_csv(candidate)
    LOGGER.warning("Missing result table: %s", path)
    return None


def _condition_labels(conditions) -> dict[str, str]:
    """A reader's name for each condition, for figure legends.

    The internal key encodes the model as well as the input, which is right
    for a table but wrong for a legend on a per-arm figure, where every curve
    already comes from the same model and the suffix is noise.  What the
    reader needs there is the input: uncorrected, or which correction at which
    setting.
    """
    labels: dict[str, str] = {}
    for condition in conditions:
        if condition.pvc_method is None:
            name = "Uncorrected input"
        else:
            name = condition.pvc_method.upper()
            if condition.pvc_iterations is not None:
                name += f", $k={condition.pvc_iterations}$"
            if condition.psf_scale is not None:
                name += f", PSF $\\times{condition.psf_scale:g}$"
        if condition.motion:
            order = "PVC first" if condition.motion_order == "pvc_then_mc" else "MC first"
            name += f" + motion ({order})"
        labels[condition.name] = name
    return labels


def _strength_ladder(arm: dict, conditions) -> dict | None:
    """Order an arm's conditions by how strongly the correction was applied.

    Only the iteration grid qualifies.  The PSF variants are a different axis
    and belong on their own figure; the shift and motion conditions are not a
    strength axis at all.  Returns ``None`` unless at least one method has more
    than one setting, because a ladder with one rung per method is a bar chart
    with extra steps.
    """
    by_name = {c.name: c for c in conditions}
    members = [n for n in arm["conditions"] if n in by_name]

    groups: dict[str, list[tuple[int, str]]] = {}
    for name in members:
        condition = by_name[name]
        if condition.pvc_method is None or condition.psf_scale is not None:
            continue
        if condition.motion or condition.subset:
            continue
        if condition.pvc_iterations is None:
            continue
        groups.setdefault(condition.pvc_method, []).append(
            (int(condition.pvc_iterations), name)
        )

    ordered = {m: [n for _, n in sorted(v)] for m, v in sorted(groups.items())}
    if not any(len(v) > 1 for v in ordered.values()):
        return None

    order = [arm["reference"]]
    edges, within = [], []
    for members_in_method in ordered.values():
        edges.append(len(order))
        order.extend(members_in_method)
        if len(members_in_method) > 1:
            within.append(tuple(members_in_method))
    # A rule before each method block, including the one that separates the
    # reference from the first corrected condition: the reference is not a
    # point on a strength axis, it is where the axis starts from.
    return {"order": order, "group_edges": tuple(edges),
            "within_groups": tuple(within)}


def _overlay_members(arm: dict, curves) -> list[str]:
    """The conditions an example-curve figure should draw for one arm.

    Taking the first three conditions in configuration order puts whichever
    sensitivity variants happen to be declared first on the figure, and those
    differ from the main setting by a few tenths of a per cent at the peak --
    three lines that land on top of each other.  The main settings of each
    method differ visibly, so those are what the figure shows; the sensitivity
    axes are reported in the sweep figures, where their size is legible.
    """
    variant = ("_i10", "_i20", "_psf", "_shift", "_reverse", "_pvcfirst")
    available = [c for c in arm["conditions"] if c in curves and c != arm["reference"]]
    primary = [c for c in available if not any(token in c for token in variant)]
    return (primary or available)[:3]


def main() -> int:
    parser = base_parser(__doc__ or "")
    parser.add_argument("--example-scans", nargs="*", default=None,
                        help="scan IDs to draw curve overlays for (default: best, median and worst)")
    args = parser.parse_args()
    config = start(args, "08_report")

    results = config.dir_results
    report_dir = config.dir_report
    report_dir.mkdir(parents=True, exist_ok=True)
    formats = ("pdf", "png") if str(config.get("report.figure_format", "pdf")) == "pdf" else ("png",)

    metrics = _read(results / "curve_metrics")
    comparisons = _read(results / "comparisons")
    bias_variance = _read(results / "bias_variance")
    binned = _read(results / "error_by_time_bin")

    if metrics is None:
        LOGGER.error("No curve metrics; run stage 06 first")
        return 1

    reference = config.reference_condition.name
    if reference not in set(metrics["condition"]):
        reference = sorted(metrics["condition"].unique())[0]

    written: list[str] = []

    if args.dry_run:
        print(json.dumps({"metrics_rows": len(metrics), "reference": reference}, indent=2))
        return 0

    # -- one set of figures per arm --------------------------------------- #
    # The deployed model and the retrained model are different networks, so a
    # figure that draws their conditions on one axis against one reference
    # shows an architectural difference as though it were an effect of the
    # correction.  Each arm is therefore drawn separately, against the only
    # reference it can be read against.  See assemble.condition_arms.
    arms = assemble.condition_arms(config.conditions, reference,
                                   config.get("dlif.model_labels"))
    condition_labels = _condition_labels(config.conditions)
    scored = set(metrics["condition"].unique())

    # A ratio is right at 1.0, not at zero, so "lower" is not "better" for it.
    # Every figure that asserts a direction has to be told which kind of metric
    # it is holding.
    TARGET = {"auc_ratio": 1.0, "peak_height_ratio": 1.0}
    FIGURE_METRICS = ("rmse", "auc_ratio", "peak_height_ratio")

    # -- the whole study on one axis -------------------------------------- #
    # Drawn first because it is the figure the rest of them support: every
    # comparison, same scale, so their sizes can be read against each other.
    if comparisons is not None and len(comparisons):
        for metric in FIGURE_METRICS:
            subset = comparisons[comparisons.get("metric") == metric] \
                if "metric" in comparisons.columns else comparisons
            if not len(subset):
                continue
            fig = figures.plot_effects_forest(
                subset, metric=metric, lower_is_better=metric not in TARGET,
                title=f"Paired difference in {metric.upper()}, every condition against its reference",
            )
            written += [str(p) for p in figures.save_figure(
                fig, report_dir / f"effects_{metric}", formats)]

    # One vertical scale for every arm's paired plot, so a small effect on a
    # narrow axis cannot look larger than a big one on a wide axis.
    shared_ylim: dict[str, tuple[float, float]] = {}
    for metric in FIGURE_METRICS:
        if metric not in metrics.columns:
            continue
        per_scan = metrics.groupby(["condition", "scan_id"])[metric].mean()
        low, high = float(per_scan.min()), float(per_scan.max())
        pad = 0.05 * (high - low or 1.0)
        shared_ylim[metric] = (low - pad, high + pad)
    for arm in arms:
        members = [c for c in arm["conditions"] if c in scored]
        if not members or arm["reference"] not in scored:
            LOGGER.info("arm %s: nothing scored yet, skipping", arm["name"])
            continue
        keep = [arm["reference"], *members]
        subset = metrics[metrics["condition"].isin(keep)]
        tag = arm["name"]

        for metric in FIGURE_METRICS:
            if metric not in metrics.columns:
                continue
            if metric in TARGET:
                # A ratio is read as distance from its target, not as a rise or
                # a fall, so it gets the agreement figure rather than the
                # paired one: seventy crossing lines say far less than a
                # centre and a spread against a marked target.
                fig = figures.plot_ratio_agreement(
                    subset, metric=metric, reference=arm["reference"],
                    conditions=members, target=TARGET[metric],
                    title=(f"{metric.replace('_', ' ')} against the arterial truth"
                           f"  -  {arm['question']}"),
                )
                written += [str(p) for p in figures.save_figure(
                    fig, report_dir / f"agreement_{metric}_{tag}", formats)]
                continue
            fig = figures.plot_paired_metric(
                subset, metric=metric, reference=arm["reference"], conditions=members,
                ylim=shared_ylim.get(metric),
                title=(f"{metric.upper()} per scan against {arm['reference']}"
                       f"  -  {arm['question']}"),
            )
            written += [str(p) for p in figures.save_figure(
                fig, report_dir / f"paired_{metric}_{tag}", formats)]

        # The strength ladder.  The agreement figure above shows where each
        # condition sits; this one shows that they sit in an order, and that
        # the order is the strength of the correction.  The step from
        # uncorrected to corrected is large enough to read on a single curve;
        # the step from one iteration count to the next is a few tenths of a
        # per cent of the peak and is legible only as a paired difference over
        # the cohort, which is the third panel.
        ladder = _strength_ladder(arm, config.conditions)
        if ladder is not None and "peak_height_ratio" in metrics.columns:
            fig = figures.plot_condition_ladder(
                subset, metric="peak_height_ratio", order=ladder["order"],
                reference=arm["reference"], labels=condition_labels, target=1.0,
                group_edges=ladder["group_edges"],
                within_groups=ladder["within_groups"],
                title=f"Peak recovery against correction strength  -  {arm['question']}",
            )
            written += [str(p) for p in figures.save_figure(
                fig, report_dir / f"strength_ladder_peak_height_ratio_{tag}", formats)]

        if bias_variance is not None and len(bias_variance):
            bv = bias_variance[bias_variance["condition"].isin(keep)]
            if len(bv):
                fig = figures.plot_bias_variance(
                    bv, title=f"Squared error decomposed  -  {arm['question']}",
                    reference=arm["reference"],
                )
                written += [str(p) for p in figures.save_figure(
                    fig, report_dir / f"bias_variance_{tag}", formats)]

        if binned is not None and len(binned):
            bb = binned[binned["condition"].isin(keep)]
            if len(bb):
                fig = figures.plot_error_by_time_bin(
                    bb, metric="rmse", reference=arm["reference"],
                    title=f"Error across the time-activity curve  -  {arm['question']}",
                )
                written += [str(p) for p in figures.save_figure(
                    fig, report_dir / f"error_by_time_bin_{tag}", formats)]

        subset.to_csv(results / f"arm_{tag}_metrics.csv", index=False)

    # The arm assignment itself is recorded, so a figure can be traced back to
    # the set of conditions it was drawn from without re-reading the config.
    (results / "condition_arms.json").write_text(
        json.dumps(arms, indent=2), encoding="utf-8")

    # -- recovery-noise trade-off from the PVC diagnostics ---------------- #
    # Assembled by the shared helper so the GUI draws exactly the same figure.
    diag_frame = assemble.frame_diagnostics_table(config.dir_pvc)
    if len(diag_frame):
        diag_frame.to_csv(results / "pvc_frame_diagnostics.csv", index=False)
        for tag, group in diag_frame.groupby("tag"):
            fig = figures.plot_recovery_noise(
                group, title=f"Recovery against noise, per frame ({tag})"
            )
            written += [str(p) for p in figures.save_figure(
                fig, report_dir / f"recovery_noise_{tag}", formats
            )]

    # -- sensitivity sweeps ------------------------------------------------ #
    # RMSE is the primary endpoint, but the peak metrics are what make the
    # deployed arm's behaviour legible.  Both axes lower the predicted peak and
    # both broaden it; the PSF axis does so several times more strongly than
    # the iteration axis.
    #
    # The peak sweeps are drawn as *paired* change from each series' first
    # point, not as marginal medians.  The iteration effect on peak height is
    # about 0.003 -- invisible in a median rounded to three decimals, and
    # negative in 58 of 70 scans at p = 1e-8.  A sweep of medians would have
    # drawn that as a flat line.  RMSE keeps the marginal form, where the
    # effects are large enough that the absolute level is the useful reading
    # and a rule at the uncorrected value gives the comparison a reader wants.
    SWEEP_METRICS = ("rmse", "peak_height_ratio", "fwhm_pred")

    def _reference_level(metric: str) -> float | None:
        """The uncorrected deployed condition's median, for the rule."""
        deployed = next((a for a in arms if a["name"] == "deployed"), None)
        if deployed is None or metric not in metrics.columns:
            return None
        rows = metrics[metrics["condition"] == deployed["reference"]]
        if not len(rows):
            return None
        per_scan = rows.groupby("scan_id")[metric].mean()
        return float(per_scan.median())

    for metric in SWEEP_METRICS:
        if metric not in metrics.columns:
            continue
        suffix = "" if metric == "rmse" else f"_{metric}"
        level = None if metric == "rmse" else _reference_level(metric)

        paired = metric != "rmse"
        sweep = (assemble.paired_sweep_table(metrics, config.conditions, metric, "iterations")
                 if paired else
                 assemble.iteration_sweep_table(metrics, config.conditions, metric=metric))
        if len(sweep) and sweep["iterations"].nunique() > 1:
            sweep.to_csv(results / f"iteration_sweep{suffix}.csv", index=False)
            fig = figures.plot_iteration_sweep(
                sweep, metric=metric, reference_level=None if paired else level,
                title=(f"change in {metric.replace('_', ' ')} from the lowest iteration count"
                       if paired else
                       f"{metric.replace('_', ' ')} against the deconvolution iteration count"),
            )
            written += [str(p) for p in figures.save_figure(
                fig, report_dir / f"iteration_sweep{suffix}", formats)]

        psf_sweep = (assemble.paired_sweep_table(metrics, config.conditions, metric, "psf_scale")
                     if paired else
                     assemble.psf_sweep_table(metrics, config.conditions, metric=metric))
        if len(psf_sweep):
            psf_sweep.to_csv(results / f"psf_sweep{suffix}.csv", index=False)
            fig = figures.plot_psf_sweep(
                psf_sweep, metric=metric, reference_level=None if paired else level,
                title=(f"change in {metric.replace('_', ' ')} from the narrowest PSF"
                       if paired else
                       f"{metric.replace('_', ' ')} against the assumed PSF width"),
            )
            written += [str(p) for p in figures.save_figure(
                fig, report_dir / f"psf_sweep{suffix}", formats)]

    # -- curve overlays for example scans ---------------------------------- #
    predictions = assemble.load_predictions(config.dir_predictions)

    if predictions is not None:
        reference_metrics = metrics[metrics["condition"] == reference].sort_values("rmse")
        if args.example_scans:
            examples = list(args.example_scans)
        elif len(reference_metrics) >= 3:
            examples = [
                reference_metrics.iloc[0]["scan_id"],
                reference_metrics.iloc[len(reference_metrics) // 2]["scan_id"],
                reference_metrics.iloc[-1]["scan_id"],
            ]
        else:
            examples = reference_metrics["scan_id"].tolist()

        # One overlay per arm rather than one overlay of everything: nineteen
        # predicted curves of the same scan differ by a few per cent at the
        # peak, and drawn together they are a band rather than a comparison.
        for scan_id in examples:
            curves, truth = assemble.curves_for_scan(predictions, scan_id)
            if truth is None:
                continue
            for arm in arms:
                members = _overlay_members(arm, curves)
                if not members or arm["reference"] not in curves:
                    continue
                fig = figures.plot_curve_overlay(
                    curves, truth, conditions=[arm["reference"], *members],
                    labels=condition_labels,
                    title=f"Scan {scan_id}  -  {arm['question']}",
                )
                written += [str(p) for p in figures.save_figure(
                    fig, report_dir / f"curves_{scan_id}_{arm['name']}", formats
                )]

    # -- motion traces ------------------------------------------------------ #
    qc_path = config.dir_motion / "motion_qc.csv"
    if qc_path.exists():
        qc = pd.read_csv(qc_path)
        qc.to_csv(report_dir / "motion_qc.csv", index=False)

    # -- LaTeX tables -------------------------------------------------------- #
    if bool(config.get("report.latex_tables", True)):
        tables.write_table(
            tables.metrics_summary_table(metrics), report_dir / "table_metrics_summary.tex"
        )
        written.append(str(report_dir / "table_metrics_summary.tex"))

        if comparisons is not None and len(comparisons):
            tables.write_table(
                tables.comparison_table(
                    comparisons,
                    metrics=["rmse", "nrmse", "auc_ratio", "peak_height_ratio", "peak_time_error"],
                    alpha=float(config.get("evaluation.statistics.alpha", 0.05)),
                    n_boot=int(config.get("evaluation.statistics.n_boot", 10000)),
                ),
                report_dir / "table_comparisons.tex",
            )
            written.append(str(report_dir / "table_comparisons.tex"))

        if bias_variance is not None and len(bias_variance):
            tables.write_table(
                tables.bias_variance_table(bias_variance), report_dir / "table_bias_variance.tex"
            )
            written.append(str(report_dir / "table_bias_variance.tex"))

    index = report_dir / "report_index.json"
    index.write_text(json.dumps({"outputs": sorted(written)}, indent=2), encoding="utf-8")
    write_provenance(index, "08_report", config)

    print(json.dumps({"n_outputs": len(written), "directory": str(report_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
