"""Stage 07 -- the motion-affected subset (secondary, exploratory).

Three modes:

``--screen``
    Measure frame-to-frame centre-of-mass displacement for every scan and rank
    them by how cyclic the displacement series looks.  This produces a
    candidate list to review with C. Salomonsen; it does not decide the subset.
    Write the confirmed IDs into ``motion.affected_ids`` in the config.

``--correct``
    Apply motion correction to the confirmed subset -- FALCON when it is
    installed, otherwise a translation-only rigid fallback that says so.  Both
    orders (motion correction then PVC, and PVC then motion correction) are
    produced, because the interpolation involved in resampling smooths the
    images and partly cancels the resolution recovery PVC is meant to give, so
    the order is tested rather than assumed.

``--qc``
    Registration quality before and after, plus the smoothing cost of the
    resampling. FALCON was developed for human whole-body PET; how it
    transfers to mouse data with sub-millimetre voxels and very low early-frame
    counts is checked here rather than taken on trust.

    python scripts/07_motion_subset.py --screen
    python scripts/07_motion_subset.py --correct --qc
"""

from __future__ import annotations

import json
import sys

import numpy as np

from _common import base_parser, start

from pvc_dlif.data.dicom_io import DynamicScan, read_nifti_4d, write_nifti_4d
from pvc_dlif.data.manifest import load_manifest
from pvc_dlif.logging_utils import get_logger, write_provenance
from pvc_dlif.motion.correct import (
    FalconRunner, interpolation_smoothing_cost, reference_frame_index, rigid_translation_correct,
)
from pvc_dlif.motion.detect import detect_motion, screen_dataset
from pvc_dlif.pvc.deconvolution import PVCSettings, make_backend
from pvc_dlif.pvc.psf import PSF
from pvc_dlif.pvc.runner import correct_scan

LOGGER = get_logger("stage.motion")


def main() -> int:
    parser = base_parser(__doc__ or "")
    parser.add_argument("--screen", action="store_true", help="rank scans as motion candidates")
    parser.add_argument("--correct", action="store_true", help="correct the confirmed subset")
    parser.add_argument("--qc", action="store_true", help="registration and smoothing quality report")
    parser.add_argument("--method", choices=["falcon", "rigid", "auto"], default="auto")
    parser.add_argument("--backend", choices=["petpvc", "numpy"], default="petpvc")
    args = parser.parse_args()
    config = start(args, "07_motion")

    if not (args.screen or args.correct or args.qc):
        parser.error("choose at least one of --screen, --correct, --qc")

    manifest = load_manifest(config.work / "manifest.json")
    motion_dir = config.dir_motion
    motion_dir.mkdir(parents=True, exist_ok=True)

    threshold = float(config.get("motion.detector.com_threshold_mm", 0.5))
    cyclic_threshold = float(config.get("motion.detector.cyclic_autocorr_threshold", 0.4))

    # ---------------------------------------------------------------- #
    if args.screen:
        entries = [e for e in manifest if e.usable]
        if args.ids:
            entries = [e for e in entries if e.scan_id in set(args.ids)]
        if args.limit:
            entries = entries[: args.limit]

        def source():
            for entry in entries:
                path = config.dir_native / f"{entry.scan_id}.nii.gz"
                if not path.exists():
                    LOGGER.warning("%s: no native NIfTI", entry.scan_id)
                    continue
                scan = read_nifti_4d(path, entry.scan_id)
                yield entry.scan_id, scan.data, scan.geometry.voxel_mm

        table = screen_dataset(source(), threshold, cyclic_threshold)
        out = motion_dir / "motion_screening.csv"
        table.to_csv(out, index=False)
        write_provenance(out, "07_motion_screen", config)

        candidates = table[table["flagged"]]["scan_id"].tolist()
        print(table.head(20).to_string(index=False))
        print(f"\n{len(candidates)} candidates: {candidates}")
        print(
            "\nThese are proposals from an automatic measure, not the affected subset. "
            "Review them with C. Salomonsen, then list the confirmed IDs under "
            "motion.affected_ids in the config."
        )

    # ---------------------------------------------------------------- #
    if args.correct or args.qc:
        subset = config.motion_affected_ids
        if args.ids:
            subset = [s for s in subset if s in set(args.ids)] or list(args.ids)
        if not subset:
            LOGGER.error(
                "motion.affected_ids is empty. Run --screen, confirm the subset, and list the "
                "IDs in the config before correcting."
            )
            return 1

        falcon = FalconRunner(config.falcon_exe)
        method = args.method
        if method == "auto":
            method = "falcon" if falcon.available else "rigid"
        if method == "falcon" and not falcon.available:
            LOGGER.error("FALCON requested but not found; set paths.falcon_exe or use --method rigid")
            return 1
        if method == "rigid":
            LOGGER.warning(
                "Using the translation-only rigid fallback. It does not correct rotation and is "
                "not a substitute for FALCON in the reported results."
            )

        psf = PSF.from_config(config)
        settings = PVCSettings(
            method=config.pvc_methods[0],
            iterations=config.iterations_primary,
            psf=psf,
            alpha=float(config.get("pvc.rvc_alpha", 1.5)),
        )
        backend = make_backend(args.backend, config.petpvc_exe)
        qc_rows: list[dict] = []

        for scan_id in subset:
            native_path = config.dir_native / f"{scan_id}.nii.gz"
            if not native_path.exists():
                LOGGER.warning("%s: no native NIfTI", scan_id)
                continue
            scan = read_nifti_4d(native_path, scan_id)

            if args.dry_run:
                LOGGER.info("would motion-correct %s with %s", scan_id, method)
                continue

            before = detect_motion(scan_id, scan.data, scan.geometry.voxel_mm,
                                   threshold, cyclic_threshold)

            if method == "falcon":
                out_dir = motion_dir / "falcon" / scan_id
                corrected_path = falcon.run(
                    native_path, out_dir,
                    reference_frame=reference_frame_index(
                        scan.data, str(config.get("motion.reference_frame", "late_mean"))
                    ),
                )
                corrected = read_nifti_4d(corrected_path, scan_id).data
                correction_info = {"method": "falcon", "output": str(corrected_path)}
            else:
                corrected, result = rigid_translation_correct(
                    scan.data, scan.geometry.voxel_mm,
                    reference=str(config.get("motion.reference_frame", "late_mean")),
                )
                correction_info = result.to_dict()

            after = detect_motion(scan_id, corrected, scan.geometry.voxel_mm,
                                  threshold, cyclic_threshold)
            smoothing = interpolation_smoothing_cost(scan.data, corrected, scan.geometry.voxel_mm)

            # mc_orig: motion-corrected, no PVC
            mc_scan = DynamicScan(
                scan_id=scan_id, data=corrected,
                frame_start_s=scan.frame_start_s, frame_duration_s=scan.frame_duration_s,
                geometry=scan.geometry, source=native_path,
                meta={**(scan.meta or {}), "motion_correction": correction_info},
            )
            write_nifti_4d(mc_scan, motion_dir / "orig" / f"{scan_id}.nii.gz")

            if args.correct:
                # Order A: motion correction, then PVC.
                pvc_after_mc, _, _ = correct_scan(mc_scan, settings, backend=backend, diagnostics=False)
                write_nifti_4d(
                    DynamicScan(scan_id, pvc_after_mc, scan.frame_start_s, scan.frame_duration_s,
                                scan.geometry, native_path,
                                {"order": "mc_then_pvc", **correction_info}),
                    motion_dir / settings.tag / f"{scan_id}.nii.gz",
                )

                # Order B: PVC, then motion correction on the corrected series.
                pvc_first, _, _ = correct_scan(scan, settings, backend=backend, diagnostics=False)
                pvc_scan = DynamicScan(scan_id, pvc_first, scan.frame_start_s,
                                       scan.frame_duration_s, scan.geometry, native_path)
                if method == "rigid":
                    mc_after_pvc, _ = rigid_translation_correct(
                        pvc_first, scan.geometry.voxel_mm,
                        reference=str(config.get("motion.reference_frame", "late_mean")),
                    )
                else:
                    tmp = motion_dir / "falcon_pvcfirst" / scan_id
                    tmp.mkdir(parents=True, exist_ok=True)
                    tmp_input = tmp / f"{scan_id}.nii.gz"
                    write_nifti_4d(pvc_scan, tmp_input)
                    mc_after_pvc = read_nifti_4d(falcon.run(tmp_input, tmp), scan_id).data
                write_nifti_4d(
                    DynamicScan(scan_id, mc_after_pvc, scan.frame_start_s, scan.frame_duration_s,
                                scan.geometry, native_path, {"order": "pvc_then_mc"}),
                    motion_dir / f"pvcfirst_{settings.tag}" / f"{scan_id}.nii.gz",
                )

            qc_rows.append(
                {
                    "scan_id": scan_id,
                    "method": method,
                    "max_displacement_before_mm": before.max_displacement_mm,
                    "max_displacement_after_mm": after.max_displacement_mm,
                    "mean_displacement_before_mm": before.mean_displacement_mm,
                    "mean_displacement_after_mm": after.mean_displacement_mm,
                    "cyclic_score_before": before.cyclic_score,
                    "cyclic_score_after": after.cyclic_score,
                    **smoothing,
                }
            )
            LOGGER.info(
                "%s: displacement %.2f -> %.2f mm, cyclic %.2f -> %.2f, gradient energy x %.3f",
                scan_id, before.max_displacement_mm, after.max_displacement_mm,
                before.cyclic_score, after.cyclic_score, smoothing["gradient_energy_ratio"],
            )

        if qc_rows:
            import pandas as pd

            frame = pd.DataFrame(qc_rows)
            qc_path = motion_dir / "motion_qc.csv"
            frame.to_csv(qc_path, index=False)
            write_provenance(qc_path, "07_motion_correct", config, method=method)
            print(frame.to_string(index=False))

            worse = frame[frame["max_displacement_after_mm"] >= frame["max_displacement_before_mm"]]
            if len(worse):
                LOGGER.warning(
                    "Registration did not reduce displacement for %d scan(s): %s. Inspect these "
                    "before using the corrected data.",
                    len(worse), worse["scan_id"].tolist(),
                )
            smoothed = frame[frame["gradient_energy_ratio"] < 0.9]
            if len(smoothed):
                LOGGER.warning(
                    "Resampling cost more than 10%% of gradient energy for %d scan(s): %s. That "
                    "smoothing works against the resolution recovery PVC provides.",
                    len(smoothed), smoothed["scan_id"].tolist(),
                )

    return 0


if __name__ == "__main__":
    sys.exit(main())
