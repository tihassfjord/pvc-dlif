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

    # -- paired dot plots ------------------------------------------------ #
    for metric in ("rmse", "auc_ratio", "peak_height_ratio"):
        if metric not in metrics.columns:
            continue
        fig = figures.plot_paired_metric(
            metrics, metric=metric, reference=reference,
            title=f"{metric.upper()} per scan, each condition against {reference}",
        )
        written += [str(p) for p in figures.save_figure(fig, report_dir / f"paired_{metric}", formats)]

    # -- bias / variance -------------------------------------------------- #
    if bias_variance is not None and len(bias_variance):
        fig = figures.plot_bias_variance(
            bias_variance, title="Squared error decomposed into bias and variance"
        )
        written += [str(p) for p in figures.save_figure(fig, report_dir / "bias_variance", formats)]

    # -- error against frame timing --------------------------------------- #
    if binned is not None and len(binned):
        fig = figures.plot_error_by_time_bin(
            binned, metric="rmse", title="Prediction error across the time-activity curve"
        )
        written += [str(p) for p in figures.save_figure(fig, report_dir / "error_by_time_bin", formats)]

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

    # -- iteration sweep --------------------------------------------------- #
    sweep = assemble.iteration_sweep_table(metrics, config.conditions, metric="rmse")
    if sweep["iterations"].nunique() > 1:
        fig = figures.plot_iteration_sweep(
            sweep, metric="rmse",
            title="Sensitivity of the result to the deconvolution iteration count",
        )
        written += [str(p) for p in figures.save_figure(fig, report_dir / "iteration_sweep", formats)]

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

        for scan_id in examples:
            curves, truth = assemble.curves_for_scan(predictions, scan_id)
            if truth is None:
                continue
            fig = figures.plot_curve_overlay(curves, truth, title=f"Scan {scan_id}")
            written += [str(p) for p in figures.save_figure(
                fig, report_dir / f"curves_{scan_id}", formats
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
