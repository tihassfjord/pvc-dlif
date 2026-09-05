"""Stage 03 -- turn corrected series into DLIF-ready data roots.

For every input tag (uncorrected, ``rl_i15``, ``rvc_i15``, ...) this writes a
data root laid out exactly like the group's::

    <work>/dlif_inputs/<tag>/IMG_SUV_96x48x48/IMG_<ID>.pkl

so the group's dataloader, trainer and evaluator run against it unchanged.
The arterial ground truth is *not* copied: it is the same measurement for every
condition, and every stage reads it from the original data root.

Every tag -- the uncorrected one included -- is built through the preprocessing
window fixed in stage 01, never by copying the distributed pickles.  That is
what makes the comparison clean: if the baseline came from the distributed files
and the corrected conditions from regenerated ones, part of the difference
between them would be preprocessing rather than PVC.

    python scripts/03_make_dlif_inputs.py
"""

from __future__ import annotations

import json
import sys

from _common import base_parser, start

from pvc_dlif.data import pkl_io
from pvc_dlif.data.dicom_io import read_nifti_4d
from pvc_dlif.data.manifest import load_manifest
from pvc_dlif.data.preprocess import load_plan
from pvc_dlif.logging_utils import get_logger, write_provenance

LOGGER = get_logger("stage.inputs")


def main() -> int:
    parser = base_parser(__doc__ or "")
    parser.add_argument("--tags", nargs="*", default=None,
                        help="input tags to build (default: all tags implied by the conditions)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = start(args, "03_inputs")

    manifest = load_manifest(config.work / "manifest.json")
    plan_path = config.work / "preprocessing_plan.json"
    if not plan_path.exists():
        raise SystemExit("Run scripts/01_prepare_data.py first")
    plan = load_plan(plan_path)

    fov = plan.to_dict()["fov_mm"]
    LOGGER.info(
        "Preprocessing: pure %s voxel crop (%.1f x %.1f x %.1f mm) at native resolution; "
        "%d scan windows fixed on the uncorrected data and reused for every condition",
        "x".join(str(v) for v in plan.crop_shape), *fov, len(plan.windows),
    )

    tags = args.tags or sorted({c.input_tag for c in config.conditions})
    shapes = config.dlif_shapes
    scan_ids = [e.scan_id for e in manifest if e.usable]
    if args.ids:
        scan_ids = [s for s in scan_ids if s in set(args.ids)]
    if args.limit:
        scan_ids = scan_ids[: args.limit]

    LOGGER.info(
        "Building %d tags x %d shapes for %d scans: tags=%s shapes=%s",
        len(tags), len(shapes), len(scan_ids), tags, shapes,
    )
    written: dict[str, int] = {}

    for tag in tags:
        out_root = config.dir_dlif_inputs / tag

        for shape in shapes:
            key = f"{tag}@{'x'.join(str(v) for v in shape)}"
            count = 0

            for scan_id in scan_ids:
                target = pkl_io.img_path(out_root, scan_id, shape)
                if target.exists() and not args.force:
                    count += 1
                    continue

                if tag == "orig":
                    source = config.dir_native / f"{scan_id}.nii.gz"
                elif tag.startswith("mc_"):
                    source = config.dir_motion / tag[3:] / f"{scan_id}.nii.gz"
                else:
                    source = config.dir_pvc / tag / f"{scan_id}.nii.gz"

                if not source.exists():
                    LOGGER.warning("%s: missing %s", scan_id, source)
                    continue
                if args.dry_run:
                    LOGGER.info("would build %s / %s from %s", key, scan_id, source.name)
                    continue

                # The window was fixed on the uncorrected data in stage 01 and
                # is reused unchanged here, so the crop is identical across
                # every condition and cannot itself carry a PVC effect.
                shaped = plan.transform(scan_id, shape)
                scan = read_nifti_4d(source, scan_id)
                resampled = shaped.apply_4d(scan.data)
                pkl_io.save_img(
                    out_root, scan_id, resampled, scan.frame_mid_min, shape,
                    extra={"source": str(source), "grid_transform": shaped.to_dict()},
                )
                count += 1

            written[key] = count
            LOGGER.info("%s: %d/%d scans", key, count, len(scan_ids))

    if args.dry_run:
        return 0

    report_path = config.dir_results / "dlif_inputs_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(written, indent=2), encoding="utf-8")
    write_provenance(
        report_path, "03_inputs", config,
        crop_shape=list(plan.crop_shape), fov_mm=fov, n_windows=len(plan.windows),
    )

    print(json.dumps(written, indent=2))
    incomplete = {t: n for t, n in written.items() if n < len(scan_ids)}
    if incomplete:
        LOGGER.warning(
            "These tags are incomplete, so the design is not fully paired for them: %s", incomplete
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
