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
from pvc_dlif.dlif.dataset import load_folds, make_folds
from pvc_dlif.eval.bias_variance import decompose_by_condition, decompose_frame
from pvc_dlif.eval.curve_metrics import metrics_frame
from pvc_dlif.report import assemble
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
    parser.add_argument("--reuse-predictions", action="store_true",
                        help="reuse predictions/retrained.* wholesale, without checking it "
                             "against the checkpoints on disk")
    parser.add_argument("--repredict", action="store_true",
                        help="ignore the cache and re-predict every condition")
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
        # Imported here, not at module scope: --skip-retrained scores the
        # pretrained predictions alone and must not need torch for it.  On one
        # Windows/conda environment importing torch alongside numpy aborted the
        # process inside numpy's BLAS, which took down a run that never
        # intended to touch a GPU.
        from pvc_dlif.dlif.adapter import DlifRepo, build_model
        from pvc_dlif.dlif.infer import (
            predict_retrained, predictions_to_frame,
        )

        retrained_path = config.dir_predictions / "retrained.parquet"
        signature_path = config.dir_predictions / "retrained.signature.json"

        # Per-condition cache.  Predicting a condition costs minutes, and
        # adding one condition to the config used to re-predict all of them.
        # What makes reuse safe is not trusting the stored table but checking
        # it: a cached condition is kept only when the input tree it names and
        # the exact set of (fold, run) checkpoints behind it are unchanged.
        # Anything else -- a recollected run, an edited pvc setting, a
        # different borrow source -- fails the check and is predicted again.
        cached = None if (args.repredict or args.reuse_predictions) else _read_table(retrained_path)
        stored: dict = {}
        if cached is not None and signature_path.exists():
            try:
                stored = json.loads(signature_path.read_text(encoding="utf-8"))
            except Exception:                               # noqa: BLE001 - treat as no cache
                stored = {}
        if not stored:
            cached = None

        def _signature(condition, checkpoint_root: Path, source_name: str) -> dict:
            # Quarantined trees (a leading "_", as used for superseded runs)
            # are excluded, because they are deliberately outside the analysis
            # and inference never reads them.  Counting them would make moving
            # an already-ignored folder invalidate a cache it cannot affect.
            names = sorted(
                str(p.relative_to(checkpoint_root).as_posix())
                for p in checkpoint_root.rglob("*.pt")
                if not any(part.startswith("_")
                           for part in p.relative_to(checkpoint_root).parts)
            )
            return {"input_tag": condition.input_tag, "source": source_name,
                    "checkpoints": names}

        retrained = _read_table(retrained_path) if args.reuse_predictions else None
        if retrained is None:
            repo = DlifRepo(config.dlif_repo)
            base_ids = [e.scan_id for e in manifest if e.usable]
            default_folds = make_folds(base_ids, int(config.get("dlif.cv.n_folds", 10)), config.seed)

            def model_factory():
                return build_model(repo, config.retrain_model_name, config.retrain_in_channels)

            collected = []
            reused_frames = []
            signatures: dict[str, dict] = {}
            for condition in config.conditions:
                if condition.model != "retrained":
                    continue
                # A condition may borrow another's checkpoints.  The weights
                # then come from the condition it names, while the input tree
                # is still its own - which is exactly the distribution shift
                # the pretrained conditions measure, applied to the retrained
                # models.  No training, inference only.
                source_name = condition.checkpoints_from or condition.name
                checkpoint_root = config.dir_models / source_name
                data_root = config.dir_dlif_inputs / condition.input_tag
                if not checkpoint_root.exists() or not data_root.exists():
                    LOGGER.warning("%s: missing checkpoints or inputs; skipping", condition.name)
                    continue
                signature = _signature(condition, checkpoint_root, source_name)
                signatures[condition.name] = signature
                if (cached is not None and stored.get(condition.name) == signature
                        and condition.name in set(cached["condition"])):
                    rows = cached[cached["condition"] == condition.name]
                    LOGGER.info("%s: unchanged, reusing %d cached prediction rows",
                                condition.name, len(rows))
                    reused_frames.append(rows)
                    continue

                if condition.borrows_checkpoints:
                    LOGGER.info(
                        "%s: models trained on '%s' applied to the '%s' inputs",
                        condition.name, source_name, condition.input_tag,
                    )
                # Score with the partition the checkpoints were trained on.
                folds_file = checkpoint_root / "folds.json"
                if folds_file.exists():
                    folds = load_folds(folds_file)
                    # The file must account for every fold that has checkpoints
                    # on disk.  One-fold-per-job training has each job write
                    # this file, so a job that recorded only its own fold
                    # leaves a partition narrower than the tree it sits in --
                    # and trusting it would score a fraction of the scans while
                    # silently ignoring the rest of the checkpoints.  The
                    # partition is a deterministic function of the scan IDs and
                    # the seed, so recomputing it is exact, not a guess.
                    on_disk = {int(d.name.split("_")[-1])
                               for d in checkpoint_root.glob("fold_*") if d.is_dir()}
                    recorded = {f.index for f in folds}
                    if not on_disk <= recorded:
                        LOGGER.warning(
                            "%s: folds.json records %d fold(s) but %d have checkpoints; "
                            "the file was written per-job and is incomplete. Using the "
                            "partition recomputed from the scan IDs and seed instead.",
                            source_name, len(recorded), len(on_disk),
                        )
                        folds = default_folds
                    elif [f.test_ids for f in folds] != [f.test_ids for f in default_folds]:
                        LOGGER.warning(
                            "%s was trained on a different partition than the full study "
                            "(%d folds over %d scans) - fine for a pilot, not for results",
                            source_name, len(folds), sum(len(f.test_ids) for f in folds),
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
            frames = list(reused_frames)
            if collected:
                frames.append(predictions_to_frame(collected))
            if frames:
                retrained = pd.concat(frames, ignore_index=True)
                retrained_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    retrained.to_parquet(retrained_path, index=False)
                except Exception:                           # noqa: BLE001 - pyarrow optional
                    retrained_path = retrained_path.with_suffix(".csv")
                    retrained.to_csv(retrained_path, index=False)
                LOGGER.info("Wrote %d prediction rows to %s (%d condition(s) reused, %d predicted)",
                            len(retrained), retrained_path.name,
                            len(reused_frames), len(signatures) - len(reused_frames))
                # The signature file is written only alongside a table that
                # matches it, so a crash between the two cannot leave a cache
                # that claims to describe predictions it does not.
                signature_path.write_text(json.dumps(signatures, indent=2), encoding="utf-8")
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

    # Each arm asks a different question and is read against its own
    # reference, so each is also its own family for the multiple-comparison
    # correction.  Correcting across arms would make the main hypothesis'
    # p-value depend on how many sensitivity conditions happened to be run on
    # the deployed model, which is a decision no reader of that hypothesis
    # would take into account.  See report.assemble.condition_arms.
    arms = assemble.condition_arms(config.conditions, reference,
                                   config.get("dlif.model_labels"))
    family_of = {name: arm["name"] for arm in arms for name in arm["conditions"]}

    stat_kwargs = dict(
        metrics=[m for m in config.get("evaluation.metrics", []) if m in metrics.columns] or None,
        test=str(config.get("evaluation.statistics.test", "wilcoxon")),
        alternative=str(config.get("evaluation.statistics.alternative", "two-sided")),
        correction=str(config.get("evaluation.statistics.multiple_comparison", "holm")),
        n_boot=int(config.get("evaluation.statistics.n_boot", 10000)),
        alpha=float(config.get("evaluation.statistics.alpha", 0.05)),
        seed=config.seed,
    )

    frames = []
    for arm in arms:
        members = [c for c in arm["conditions"] if c in available]
        if not members or arm["reference"] not in available:
            LOGGER.info("arm %s: nothing scored, skipping", arm["name"])
            continue
        LOGGER.info("arm %s: %d condition(s) against %s",
                    arm["name"], len(members), arm["reference"])
        frames.append(compare_conditions(
            metrics[metrics["condition"].isin([arm["reference"], *members])],
            reference=arm["reference"], conditions=members,
            families={name: arm["name"] for name in members},
            **stat_kwargs,
        ))

    comparisons = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
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
        # ``null`` in the config means the Patlak break point is fitted per
        # curve, which is the reference implementation's behaviour and the
        # default here; a number pins it, for the sensitivity check.
        t_star = config.get("evaluation.kinetics.patlak_t_star_min", None)
        t_star = None if t_star is None else float(t_star)
        # The two-tissue model is reversible unless the config asks otherwise.
        # k4 = 0 is the usual FDG shorthand, but a model that cannot represent
        # washout absorbs any washout present into the other rate constants.
        irreversible = bool(config.get("evaluation.kinetics.irreversible", False))
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
                    t_star_min=t_star,
                    # The VOI filter has already been applied above, from the
                    # config, so the selection is not narrowed a second time
                    # here -- a liver row appears exactly when the config asks
                    # for one.
                    vois=None,
                    irreversible=irreversible,
                ):
                    rows.append({"condition": condition, "scan_id": scan_id, **row})
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("kinetics failed for %s / %s: %s", condition, scan_id, exc)
        if rows:
            _write(pd.DataFrame(rows), results_dir / "kinetics")

    # ---------------------------------------------------------------- #
    # Headline summary
    # ---------------------------------------------------------------- #
    # The summary carries one reference, so only the rows measured against it
    # go in; the deployed arm has its own reference and lives in comparisons.csv.
    headline = comparisons
    if "metric" in comparisons.columns:
        headline = comparisons[(comparisons["metric"] == "rmse")
                               & (comparisons["reference"] == reference)]
    summary = {
        "reference": reference,
        "n_scans": int(metrics["scan_id"].nunique()),
        "conditions": sorted(metrics["condition"].unique()),
        "rmse_median": metrics.groupby("condition")["rmse"].median().round(4).to_dict(),
        "rmse_vs_reference": [
            {
                "condition": row["condition"],
                "arm": row.get("family"),
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
