"""Stage 05 -- retrain or fine-tune DLIF on each input representation.

This is the core of the main hypothesis.  Deconvolution changes the statistics
of the images, so a model trained on uncorrected data and applied to corrected
data is operating under a train-test distribution shift.  Retraining removes
that confound: comparing the retrained model against the pretrained one on
*identical* corrected inputs separates the effect of the corrected images from
the effect of the shift.

The protocol is copied from the baseline -- same folds, same repeats, same
optimiser, schedule, loss and stopping rule -- and the folds are derived from
the scan IDs and the seed alone, so every condition trains and tests on exactly
the same partitions.

    python scripts/05_retrain.py --conditions baseline_retrained rl_retrained
    python scripts/05_retrain.py --folds 1 2 --runs 3      # a cheaper pilot (folds are 1-based)
"""

from __future__ import annotations

import json
import sys

from _common import base_parser, start

from pvc_dlif.data import pkl_io
from pvc_dlif.data.manifest import load_manifest
from pvc_dlif.dlif.adapter import (
    DlifRepo, build_model, infer_input_spec, load_pretrained_module,
)
from pvc_dlif.dlif.dataset import make_folds
from pvc_dlif.dlif.train import TrainSettings, train_condition
from pvc_dlif.logging_utils import get_logger, write_provenance

LOGGER = get_logger("stage.retrain")


def main() -> int:
    parser = base_parser(__doc__ or "")
    parser.add_argument("--conditions", nargs="*", default=None,
                        help="condition names to train (default: all retrained conditions)")
    parser.add_argument("--mode", choices=["retrain", "finetune"], default=None,
                        help="override dlif.finetune.enabled")
    parser.add_argument("--folds", nargs="*", type=int, default=None, help="only these fold indices")
    parser.add_argument("--runs", type=int, default=None, help="override dlif.cv.n_runs")
    parser.add_argument("--epochs", type=int, default=None, help="override dlif.train.epochs")
    parser.add_argument("--device", default=None, help="override dlif.train.device")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    config = start(args, "05_retrain")

    manifest = load_manifest(config.work / "manifest.json")
    repo = DlifRepo(config.dlif_repo)

    # Check the configured input shape against what the architecture can take,
    # before spending hours discovering it inside a convolution.
    probe = build_model(repo, config.retrain_model_name, config.retrain_in_channels)
    problem = infer_input_spec(probe).check_shape(config.retrain_shape)
    if problem:
        LOGGER.error(
            "dlif.retrain_model.input_shape is %s, but %s cannot use it: %s",
            list(config.retrain_shape), config.retrain_model_name, problem,
        )
        return 1
    del probe

    finetune = (args.mode == "finetune") if args.mode else bool(config.get("dlif.finetune.enabled", False))
    settings = TrainSettings.from_config(config, finetune=finetune)
    if args.epochs:
        settings.epochs = args.epochs
    if args.device:
        settings.device = args.device

    conditions = [
        c for c in config.conditions
        if c.model == "retrained" and (args.conditions is None or c.name in set(args.conditions))
    ]
    if not conditions:
        raise SystemExit("No retrained conditions selected")

    n_folds = int(config.get("dlif.cv.n_folds", 10))
    n_runs = args.runs if args.runs is not None else int(config.get("dlif.cv.n_runs", 10))
    validation_size = float(config.get("dlif.cv.validation_size", 0.15))
    group_of = {e.scan_id: e.group for e in manifest}

    # Folds come from the scan IDs and the seed only, so they are identical for
    # every condition -- the pairing depends on it.
    base_ids = [e.scan_id for e in manifest if e.usable]
    if args.ids:
        base_ids = [s for s in base_ids if s in set(args.ids)]
    if args.limit:
        base_ids = base_ids[: args.limit]

    all_folds = make_folds(base_ids, n_folds, config.seed)
    folds = [f for f in all_folds if args.folds is None or f.index in set(args.folds)]
    if not folds:
        # Folds are numbered 1..n_folds (the group's convention). Selecting none
        # must be an error, not a run that trains nothing and reports success.
        raise SystemExit(
            f"--folds {args.folds} selects no fold; valid indices are 1..{len(all_folds)}"
        )

    LOGGER.info(
        "%s %d conditions on %d scans, %d folds x %d runs, %d epochs, device %s",
        "Fine-tuning" if finetune else "Retraining",
        len(conditions), len(base_ids), len(folds), n_runs, settings.epochs, settings.resolve_device(),
    )

    if args.dry_run:
        print(json.dumps(
            {
                "conditions": [c.name for c in conditions],
                "n_scans": len(base_ids),
                "folds": [f.index for f in folds],
                "runs": n_runs,
                "jobs": len(conditions) * len(folds) * n_runs,
                "settings": settings.__dict__,
            },
            indent=2, default=str,
        ))
        return 0

    summary: dict[str, object] = {}
    for condition in conditions:
        data_root = config.dir_dlif_inputs / condition.input_tag
        if not data_root.exists():
            LOGGER.error("%s: no input tree at %s; run stage 03", condition.name, data_root)
            continue

        available = set(pkl_io.available_ids(data_root, config.retrain_shape, require_aif=False))
        usable_here = [s for s in base_ids if s in available]
        if len(usable_here) != len(base_ids):
            LOGGER.warning(
                "%s: %d of %d scans present; folds are still taken from the full ID list so "
                "the partition matches the other conditions",
                condition.name, len(usable_here), len(base_ids),
            )

        if finetune:
            def model_factory(_condition=condition):
                model, _, _ = load_pretrained_module(
                    repo, weights=config.pretrained_weights, device="cpu"
                )
                if bool(config.get("dlif.finetune.freeze_encoder", False)):
                    for name, parameter in model.named_parameters():
                        if name.startswith("encoder"):
                            parameter.requires_grad_(False)
                return model
        else:
            def model_factory():
                return build_model(repo, config.retrain_model_name, config.retrain_in_channels)

        out_dir = config.dir_models / condition.name
        results = train_condition(
            model_factory=model_factory,
            data_root=data_root,
            aif_root=config.dlif_data_root,
            scan_ids=usable_here,
            group_of=group_of,
            out_dir=out_dir,
            settings=settings,
            n_folds=n_folds,
            n_runs=n_runs,
            validation_size=validation_size,
            img_shape=config.retrain_shape,
            augmentation={**config.get("dlif.augmentation", {}),
                          "add_average": config.retrain_add_average},
            seed=config.seed,
            resume=not args.no_resume,
            folds=folds,
        )
        summary[condition.name] = {
            "runs": len(results),
            "median_best_val_loss": float(
                sorted(r.best_val_loss for r in results)[len(results) // 2]
            ) if results else None,
            "total_hours": round(sum(r.seconds for r in results) / 3600, 2),
            "checkpoints": str(out_dir),
        }
        write_provenance(out_dir / "training.json", "05_retrain", config,
                         condition=condition.name, mode="finetune" if finetune else "retrain",
                         settings=settings.__dict__)

    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
