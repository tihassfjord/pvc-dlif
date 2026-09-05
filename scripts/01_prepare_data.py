"""Stage 01 -- convert the DICOM exports and fix the preprocessing.

Three jobs:

1. Write every dynamic scan as a 4D NIfTI at its reconstructed resolution
   (``<work>/native/<ID>.nii.gz`` plus a timing sidecar).  PVC runs on those,
   because that is the grid the measured point-source PSF refers to.

2. Fix the preprocessing window for each scan and store it in
   ``<work>/preprocessing_plan.json``.  The group's preprocessing turns out to
   be a pure 96 x 48 x 48 voxel crop at native resolution -- no resampling --
   so the window is three integers, recovered per scan by matching against
   their distributed input.  It is derived here from the *uncorrected* series
   and reused verbatim for every condition, which is what guarantees the
   corrected and uncorrected conditions differ only by PVC and not also by a
   crop.

3. Report how closely the reconstructed windows reproduce the distributed
   ``IMG_*.pkl`` files.  Anything short of r = 1.0 on a scan means that scan's
   distributed input did not come from this reconstruction, and is worth
   knowing before it is used.

    python scripts/01_prepare_data.py                  # convert + plan + check
    python scripts/01_prepare_data.py --skip-convert   # re-derive the plan only
"""

from __future__ import annotations

import json
import sys

import numpy as np

from _common import base_parser, start

from pvc_dlif.data import pkl_io
from pvc_dlif.data.dicom_io import read_dynamic_dicom, read_nifti_4d, write_nifti_4d
from pvc_dlif.data.manifest import load_manifest
from pvc_dlif.data.preprocess import derive_windows
from pvc_dlif.logging_utils import get_logger, write_provenance

LOGGER = get_logger("stage.prepare")


def main() -> int:
    parser = base_parser(__doc__ or "")
    parser.add_argument("--skip-convert", action="store_true",
                        help="reuse the native NIfTI files already written")
    parser.add_argument("--force", action="store_true", help="rewrite existing NIfTI files")
    parser.add_argument("--no-match", action="store_true",
                        help="centre every window on the animal instead of matching the "
                             "distributed inputs")
    parser.add_argument("--diagnose-scans", type=int, default=8,
                        help="how many scans to verify against the distributed inputs")
    args = parser.parse_args()
    config = start(args, "01_prepare")

    manifest_path = config.work / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit("Run scripts/00_build_manifest.py first")
    manifest = load_manifest(manifest_path)

    entries = [e for e in manifest if e.usable]
    if args.ids:
        entries = [e for e in entries if e.scan_id in set(args.ids)]
    if args.limit:
        entries = entries[: args.limit]
    if not entries:
        raise SystemExit("No usable scans to process")

    native_dir = config.dir_native
    native_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- #
    # 1. Convert to NIfTI
    # ---------------------------------------------------------------- #
    available: list[str] = []
    for entry in entries:
        target = native_dir / f"{entry.scan_id}.nii.gz"
        if target.exists() and not args.force:
            available.append(entry.scan_id)
            continue
        if args.skip_convert:
            LOGGER.warning("%s: no native NIfTI and --skip-convert was given", entry.scan_id)
            continue
        if args.dry_run:
            LOGGER.info("would convert %s", entry.scan_id)
            continue
        try:
            scan = read_dynamic_dicom(entry.dicom_path, entry.scan_id)  # type: ignore[arg-type]
            write_nifti_4d(scan, target)
            available.append(entry.scan_id)
            LOGGER.info("converted %s (%d/%d)", entry.scan_id, len(available), len(entries))
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("failed to convert %s: %s", entry.scan_id, exc)

    if args.dry_run:
        print(json.dumps({"would_convert": len(entries)}, indent=2))
        return 0

    LOGGER.info("%d/%d scans available as native NIfTI", len(available), len(entries))
    if not available:
        LOGGER.error("No native NIfTI files; nothing to derive a preprocessing plan from")
        return 1

    # ---------------------------------------------------------------- #
    # 2. Fix the preprocessing window, from the uncorrected data only
    # ---------------------------------------------------------------- #
    crop_shape = config.reference_shape

    def uncorrected_scans():
        for scan_id in available:
            scan = read_nifti_4d(native_dir / f"{scan_id}.nii.gz", scan_id)
            reference = None
            if not args.no_match:
                try:
                    reference, _ = pkl_io.load_img(config.dlif_data_root, scan_id, crop_shape)
                except Exception:  # noqa: BLE001 - not every scan has one
                    reference = None
            yield scan_id, scan.data, scan.geometry.voxel_mm, reference

    plan = derive_windows(
        uncorrected_scans(),
        crop_shape=crop_shape,
        apply_intensity_scale=bool(config.get("preprocessing.apply_intensity_scale", True)),
        late_fraction=float(config.get("preprocessing.late_fraction", 0.4)),
        threshold_fraction=float(config.get("preprocessing.threshold_fraction", 0.15)),
    )
    plan_path = plan.save(config.work / "preprocessing_plan.json")
    write_provenance(plan_path, "01_prepare", config, n_scans=len(plan.windows))

    # ---------------------------------------------------------------- #
    # 3. Diagnose: how close is this to the distributed inputs?
    # ---------------------------------------------------------------- #
    reference_shape = config.reference_shape
    rows: list[dict] = []
    for scan_id in available[: args.diagnose_scans]:
        try:
            distributed, _ = pkl_io.load_img(config.dlif_data_root, scan_id, reference_shape)
        except Exception:  # noqa: BLE001 - not every scan has a distributed input
            continue
        scan = read_nifti_4d(native_dir / f"{scan_id}.nii.gz", scan_id)
        ours = plan.transform(scan_id, reference_shape).apply_4d(scan.data)

        n = min(ours.shape[0], distributed.shape[0])
        a = ours[:n].reshape(n, -1).astype(np.float64)
        b = distributed[:n].reshape(n, -1).astype(np.float64)
        per_frame = [
            float(np.corrcoef(a[t], b[t])[0, 1])
            if a[t].std() > 1e-9 and b[t].std() > 1e-9 else np.nan
            for t in range(n)
        ]
        rows.append({
            "scan_id": scan_id,
            "volume_correlation": float(np.nanmedian(per_frame)),
            "late_frame_correlation": float(np.nanmedian(per_frame[n // 2:])),
            "mean_ratio": float(b.mean() / a.mean()) if a.mean() else np.nan,
        })

    diagnostic_path = config.dir_results / "preprocessing_agreement.csv"
    diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        import pandas as pd

        frame = pd.DataFrame(rows)
        frame.to_csv(diagnostic_path, index=False)
        median_r = float(frame["volume_correlation"].median())
        LOGGER.info(
            "Agreement with the distributed DLIF inputs: median r = %.4f over %d scans "
            "(written to %s)", median_r, len(frame), diagnostic_path.name,
        )
        if median_r >= 0.999:
            LOGGER.info(
                "The uncorrected inputs reproduce the distributed ones exactly, so the "
                "pretrained model is being used on the data it was trained on."
            )
        else:
            imperfect = frame[frame["volume_correlation"] < 0.999]["scan_id"].tolist()
            LOGGER.warning(
                "%d scan(s) do not reproduce their distributed input exactly (median r = "
                "%.4f): %s. Their distributed input probably came from a different "
                "reconstruction. The PVC comparison is still valid for them -- every "
                "condition uses the same window -- but the pretrained model is extrapolating "
                "on those scans.", len(imperfect), median_r, imperfect[:8],
            )
    else:
        median_r = float("nan")
        LOGGER.warning("No scan could be compared against a distributed input")

    summary = {
        "n_converted": len(available),
        "crop_shape": list(plan.crop_shape),
        "voxel_mm": [round(v, 5) for v in plan.voxel_mm],
        "fov_mm": plan.to_dict()["fov_mm"],
        "resampled": False,
        "n_matched_to_distributed": len(plan.fits),
        "n_exact_matches": plan.n_exact,
        "agreement_with_distributed_median_r": round(median_r, 5) if rows else None,
        "n_window_warnings": len(plan.warnings),
        "plan": str(plan_path),
    }
    (config.dir_results / "prepare_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))

    if plan.warnings:
        LOGGER.warning(
            "%d window warning(s); the first few: %s",
            len(plan.warnings), plan.warnings[:3],
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
