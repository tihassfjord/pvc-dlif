"""Stage 06 -- score every condition and compare them.

Collects predictions from the pretrained and retrained models, then produces
the whole evaluation in one pass:

* curve-level metrics per scan and condition;
* the bias/variance decomposition, so a flat RMSE with a large trade underneath
  it is visible rather than reported as "no effect";
* paired Wilcoxon tests against the reference condition, with effect sizes,
  bootstrap intervals and multiple-comparison correction;
* per-frame errors binned by frame timing, which is where the recovery-noise
  trade-off across the time-activity curve shows up;
* the scans where a corrected condition is worse, named rather than averaged;
* Patlak and two-tissue parameters computed with each predicted input function
  against the same tissue curves.

    python scripts/06_evaluate.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from _common import base_parser, start

from pvc_dlif.data import pkl_io
from pvc_dlif.data.manifest import load_manifest
from pvc_dlif.dlif.adapter import DlifRepo, build_model
from pvc_dlif.dlif.dataset import load_folds, make_folds
from pvc_dlif.dlif.infer import predict_retrained, predictions_to_frame, save_predictions
from pvc_dlif.eval.bias_variance import decompose_by_condition, decompose_frame
from pvc_dlif.eval.curve_metrics import metrics_frame
from pvc_dlif.eval.frames import failure_modes, frame_error_table, summarise_by_bin
from pvc_dlif.eval.kinetics import compare_kinetics
from pvc_dlif.eval.stats import compare_conditions
from pvc_dlif.logging_utils import get_logger, write_provenance

LOGGER = get_logger("stage.evaluate")


def _read_table(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_parquet(path)
    csv = path.with_suffix(".csv")
    if csv.exists():
        return pd.read_csv(csv)
    return None


def _write(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path.with_suffix(".csv"), index=False)
    try:
        frame.to_parquet(path.with_suffix(".parquet"), index=False)
    except Exception:  # noqa: BLE001 - pyarrow optional
        pass
    LOGGER.info("Wrote %s (%d rows)", path.with_suffix(".csv").name, len(frame))
    return path.with_suffix(".csv")


def main() -> int:
    parser = base_parser(__doc__ or "")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-retrained", action="store_true",
                        help="score only the pretrained predictions")
    parser.add_argument("--skip-kinetics", action="store_true")
    args = parser.parse_args()
    config = start(args, "06_evaluate")

    manifest = load_manifest(config.work / "manifest.json")
    results_dir = config.dir_results

    # ---------------------------------------------------------------- #
    # Gather predictions
    # ---------------------------------------------------------------- #
    tables: list[pd.DataFrame] = []

    pretrained = _read_table(config.dir_predictions / "pretrained.parquet")
    if pretrained is not None:
        tables.append(pretrained)
    else:
        LOGGER.warning("No pretrained predictions; run stage 04")

    if not args.skip_retrained:
        retrained_path = config.dir_predictions / "retrained.parquet"
        retrained = _read_table(retrained_path)
        if retrained is None:
            repo = DlifRepo(config.dlif_repo)
            base_ids = [e.scan_id for e in manifest if e.usable]
            default_folds = make_folds(base_ids, int(config.get("dlif.cv.n_folds", 10)), config.seed)

            def model_factory():
                return build_model(repo, config.retrain_model_name, config.retrain_in_channels)

            collected = []
            for condition in config.conditions:
                if condition.model != "retrained":
                    continue
                checkpoint_root = config.dir_models / condition.name
                data_root = config.dir_dlif_inputs / condition.input_tag
                if not checkpoint_root.exists() or not data_root.exists():
                    LOGGER.warning("%s: missing checkpoints or inputs; skipping", condition.name)
                    continue
                # Score with the partition the checkpoints were trained on.
                folds_file = checkpoint_root / "folds.json"
                if folds_file.exists():
                    folds = load_folds(folds_file)
                    if [f.test_ids for f in folds] != [f.test_ids for f in default_folds]:
                        LOGGER.warning(
                            "%s was trained on a different partition than the full study "
                            "(%d folds over %d scans) - fine for a pilot, not for results",
                            condition.name, len(folds), sum(len(f.test_ids) for f in folds),
                        )
                else:
                    folds = default_folds
                collected.extend(
                    predict_retrained(
                        model_factory=model_factory,
                        checkpoint_root=checkpoint_root,
                        data_root=data_root,
                        aif_root=config.dlif_data_root,
                        folds=folds,
                        condition=condition.name,
                        img_shape=config.retrain_shape,
                        device=args.device,
                        add_average=config.retrain_add_average,
                    )
                )
            if collected:
                save_predictions(collected, retrained_path)
                retrained = predictions_to_frame(collected)
        if retrained is not None:
            tables.append(retrained)

    if not tables:
        LOGGER.error("No predictions to evaluate")
        return 1

    predictions = pd.concat(tables, ignore_index=True)
    LOGGER.info(
        "%d prediction rows over %d conditions and %d scans",
        len(predictions), predictions["condition"].nunique(), predictions["scan_id"].nunique(),
    )

    if args.dry_run:
        print(predictions.groupby("condition")["scan_id"].nunique().to_string())
        return 0

    # ---------------------------------------------------------------- #
    # Curve metrics
    # ---------------------------------------------------------------- #
    split = float(config.get("evaluation.frame_bins_min", [0, 2.5])[2]
                  if len(config.get("evaluation.frame_bins_min", [])) > 2 else 2.5)
    metrics = metrics_frame(predictions, early_late_split_min=split)
    _write(metrics, results_dir / "curve_metrics")

    # ---------------------------------------------------------------- #
    # Bias / variance
    # ---------------------------------------------------------------- #
    per_scan = decompose_frame(predictions)
    _write(per_scan, results_dir / "bias_variance_per_scan")
    by_condition = decompose_by_condition(predictions)
    _write(by_condition, results_dir / "bias_variance")

    # ---------------------------------------------------------------- #
    # Paired statistics
    # ---------------------------------------------------------------- #
    reference = config.reference_condition.name
    available = set(metrics["condition"].unique())
    if reference not in available:
        reference = sorted(available)[0]
        LOGGER.warning("Configured reference is absent; using %s", reference)

    comparisons = compare_conditions(
        metrics,
        reference=reference,
        metrics=[m for m in config.get("evaluation.metrics", []) if m in metrics.columns] or None,
        test=str(config.get("evaluation.statistics.test", "wilcoxon")),
        alternative=str(config.get("evaluation.statistics.alternative", "two-sided")),
        correction=str(config.get("evaluation.statistics.multiple_comparison", "holm")),
        n_boot=int(config.get("evaluation.statistics.n_boot", 10000)),
        alpha=float(config.get("evaluation.statistics.alpha", 0.05)),
        seed=config.seed,
    )
    _write(comparisons, results_dir / "comparisons")

    # ---------------------------------------------------------------- #
    # Frame-level analysis
    # ---------------------------------------------------------------- #
    frame_table = frame_error_table(predictions, config.get("evaluation.frame_bins_min"))
    binned = summarise_by_bin(frame_table)
    _write(binned, results_dir / "error_by_time_bin")

    modes = failure_modes(metrics, reference=reference, metric="rmse")
    _write(modes, results_dir / "failure_modes")

    # Group and tracer stratification: does PVC help uniformly?
    group_of = {e.scan_id: (e.group or "ungrouped") for e in manifest}
    tracer_of = {e.scan_id: e.tracer for e in manifest}
    stratified = metrics.copy()
    stratified["group"] = stratified["scan_id"].map(group_of)
    stratified["tracer"] = stratified["scan_id"].map(tracer_of)
    by_group = (
        stratified.groupby(["condition", "group"])["rmse"]
        .agg(["count", "median", "mean", "std"]).reset_index()
    )
    _write(by_group, results_dir / "rmse_by_group")

    # ---------------------------------------------------------------- #
    # Kinetic modelling
    # ---------------------------------------------------------------- #
    if not args.skip_kinetics and bool(config.get("evaluation.kinetics.enabled", True)):
        target_vois = list(config.get("evaluation.kinetics.target_vois", []))
        rows: list[dict] = []
        collapsed = (
            predictions.groupby(["condition", "scan_id", "frame"], dropna=False)
            .agg(predicted=("predicted", "mean"), truth=("truth", "first"),
                 time_min=("time_min", "first"))
            .reset_index()
        )
        for (condition, scan_id), group in collapsed.groupby(["condition", "scan_id"]):
            if not pkl_io.voi_path(config.dlif_data_root, scan_id).exists():
                continue
            curves, voi_time = pkl_io.load_voi(config.dlif_data_root, scan_id)
            selected = {k: v for k, v in curves.items() if not target_vois or k in target_vois}
            group = group.sort_values("frame")
            try:
                for row in compare_kinetics(
                    group["predicted"].to_numpy(),
                    group["truth"].to_numpy(),
                    selected,
                    voi_time,
                    models=list(config.get("evaluation.kinetics.models", ["patlak"])),
                    t_star_min=float(config.get("evaluation.kinetics.patlak_t_star_min", 5.0)),
                ):
                    rows.append({"condition": condition, "scan_id": scan_id, **row})
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("kinetics failed for %s / %s: %s", condition, scan_id, exc)
        if rows:
            _write(pd.DataFrame(rows), results_dir / "kinetics")

    # ---------------------------------------------------------------- #
    # Headline summary
    # ---------------------------------------------------------------- #
    headline = comparisons[comparisons["metric"] == "rmse"] if "metric" in comparisons.columns else comparisons
    summary = {
        "reference": reference,
        "n_scans": int(metrics["scan_id"].nunique()),
        "conditions": sorted(metrics["condition"].unique()),
        "rmse_median": metrics.groupby("condition")["rmse"].median().round(4).to_dict(),
        "rmse_vs_reference": [
            {
                "condition": row["condition"],
                "delta_median": round(float(row["median_difference"]), 4),
                "ci": [round(float(row["ci_low"]), 4), round(float(row["ci_high"]), 4)],
                "effect_size": round(float(row["effect_size"]), 3),
                "p_adjusted": (None if not np.isfinite(row["p_adjusted"]) else round(float(row["p_adjusted"]), 4)),
            }
            for _, row in headline.iterrows()
        ],
        "bias_variance": by_condition.round(5).to_dict(orient="records"),
    }
    summary_path = results_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_provenance(summary_path, "06_evaluate", config)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
