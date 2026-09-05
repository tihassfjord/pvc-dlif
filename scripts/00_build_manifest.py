"""Stage 00 -- inventory the dataset.

Writes ``<work>/manifest.json`` and a CSV alongside it, listing every animal
and what exists for it.  Run this first: every later stage reads the manifest,
and its summary is the number that goes into the thesis when the dataset is
described.

    python scripts/00_build_manifest.py --read-geometry
"""

from __future__ import annotations

import json
import sys

from _common import base_parser, start

from pvc_dlif.data.manifest import build_manifest
from pvc_dlif.logging_utils import get_logger, write_provenance

LOGGER = get_logger("stage.manifest")


def main() -> int:
    parser = base_parser(__doc__ or "")
    parser.add_argument(
        "--no-geometry", action="store_true",
        help="skip the DICOM headers (faster, but no grid or frame-schedule checks)",
    )
    parser.add_argument(
        "--verify-inputs", action="store_true",
        help="also open every DLIF input pickle to check its shape (reads the whole dataset)",
    )
    args = parser.parse_args()
    config = start(args, "00_manifest")

    manifest = build_manifest(
        dicom_root=config.dicom_root,
        dlif_data_root=config.dlif_data_root,
        dlif_repo=config.dlif_repo if config.dlif_repo.exists() else None,
        img_shape=config.dlif_shape,
        ignore_ids=config.ignore_ids,
        motion_affected=config.motion_affected_ids,
        read_geometry=not args.no_geometry,
        verify_inputs=args.verify_inputs,
    )

    if args.dry_run:
        print(json.dumps(manifest.summary(), indent=2))
        return 0

    path = manifest.save(config.work / "manifest.json")
    manifest.to_dataframe().to_csv(config.work / "manifest.csv", index=False)
    write_provenance(path, "00_manifest", config, summary=manifest.summary())

    summary = manifest.summary()
    print(json.dumps(summary, indent=2))

    if summary["n_usable"] == 0:
        LOGGER.error(
            "No usable scans. A scan needs a native DICOM, a DLIF input and an arterial "
            "ground truth to contribute to the paired design; check the paths in the config."
        )
        return 1

    missing_dicom = [e.scan_id for e in manifest if e.has_dlif_img and not e.has_dicom and not e.excluded]
    if missing_dicom:
        LOGGER.warning(
            "%d scans have a DLIF input but no native DICOM, so they cannot be PVC-corrected "
            "on the reconstructed grid: %s",
            len(missing_dicom), missing_dicom[:10],
        )

    LOGGER.info("Manifest written to %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
