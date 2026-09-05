"""Stage 04 -- predictions from the deployed, pretrained model.

Applies the group's pretrained DLIF model to every input representation.  What
this measures is the robustness of the deployed model to the distribution shift
that deconvolution introduces -- corrected images have different noise, sharper
edges and more high-frequency content than anything the model saw in training.

That is reported as its own result.  It is not a test of the main hypothesis:
a drop here cannot be separated from the model simply failing to generalise,
which is what stage 05 exists to settle.

    python scripts/04_infer_baseline.py
"""

from __future__ import annotations

import json
import sys

from _common import base_parser, start

from pvc_dlif.data import pkl_io
from pvc_dlif.data.manifest import load_manifest
from pvc_dlif.dlif.adapter import DlifRepo, load_pretrained_module
from pvc_dlif.dlif.infer import predict_pretrained, save_predictions
from pvc_dlif.logging_utils import get_logger, write_provenance

LOGGER = get_logger("stage.infer")


def main() -> int:
    parser = base_parser(__doc__ or "")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--conditions", nargs="*", default=None,
                        help="condition names to predict (default: all pretrained conditions)")
    args = parser.parse_args()
    config = start(args, "04_infer")

    manifest = load_manifest(config.work / "manifest.json")
    repo = DlifRepo(config.dlif_repo)

    # The checkpoint is a pickled module and carries its own architecture, which
    # the repository's current models.py no longer builds. Loading it as a
    # module is the only way to get the deployed network rather than a mostly
    # randomly initialised look-alike.
    model, spec, info = load_pretrained_module(
        repo, weights=config.pretrained_weights, device=args.device
    )
    LOGGER.info("Loaded %s: %s", info["architecture"], json.dumps(info))

    shape = config.pretrained_shape
    if spec.fixed_spatial and spec.spatial_shape and tuple(spec.spatial_shape) != tuple(shape):
        LOGGER.error(
            "The model needs %s volumes but dlif.pretrained.input_shape is %s. %s",
            spec.spatial_shape, list(shape), spec.note,
        )
        return 1
    if spec.add_average != config.pretrained_add_average:
        LOGGER.warning(
            "The model takes %d input channels, so add_average should be %s; the config says %s. "
            "Using the model's own requirement.",
            spec.in_channels, spec.add_average, config.pretrained_add_average,
        )

    conditions = [
        c for c in config.conditions
        if c.model == "pretrained" and (args.conditions is None or c.name in set(args.conditions))
    ]
    if not conditions:
        raise SystemExit("No pretrained conditions selected")

    scan_ids = [e.scan_id for e in manifest if e.usable]
    if args.ids:
        scan_ids = [s for s in scan_ids if s in set(args.ids)]
    if args.limit:
        scan_ids = scan_ids[: args.limit]

    all_predictions = []

    for condition in conditions:
        data_root = config.dir_dlif_inputs / condition.input_tag
        if not data_root.exists():
            LOGGER.warning("%s: no input tree at %s; run stage 03", condition.name, data_root)
            continue

        available = {
            s for s in scan_ids if pkl_io.img_path(data_root, s, shape).exists()
        }
        missing = set(scan_ids) - available
        if missing:
            LOGGER.warning("%s: %d scans missing from this tag: %s",
                           condition.name, len(missing), sorted(missing)[:8])

        if args.dry_run:
            LOGGER.info("would predict %d scans for %s", len(available), condition.name)
            continue

        predictions = predict_pretrained(
            model=model,
            data_root=data_root,
            aif_root=config.dlif_data_root,
            scan_ids=sorted(available),
            condition=condition.name,
            img_shape=shape,
            device=args.device,
            add_average=spec.add_average,
            spec=spec,
        )
        all_predictions.extend(predictions)

    if args.dry_run:
        return 0
    if not all_predictions:
        LOGGER.error("No predictions produced")
        return 1

    path = save_predictions(all_predictions, config.dir_predictions / "pretrained.parquet")
    write_provenance(path, "04_infer", config, model_info=info, dlif_repo=repo.describe(),
                     input_shape=list(shape), add_average=spec.add_average,
                     conditions=[c.name for c in conditions])

    # A paired design needs every condition to cover the same scans.
    by_condition: dict[str, set[str]] = {}
    for prediction in all_predictions:
        by_condition.setdefault(prediction.condition, set()).add(prediction.scan_id)
    common = set.intersection(*by_condition.values()) if by_condition else set()
    print(json.dumps(
        {"conditions": {k: len(v) for k, v in by_condition.items()}, "paired_scans": len(common)},
        indent=2,
    ))
    if any(len(v) != len(common) for v in by_condition.values()):
        LOGGER.warning(
            "Conditions do not cover the same scans; the paired comparison will use the "
            "%d scans common to all of them.", len(common),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
