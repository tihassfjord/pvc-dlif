"""Stage 07 -- the motion-affected subset (secondary, exploratory).

Three modes:

``--screen``
    Rank the scans as motion candidates, cheaply and before any correction has
    run.  Treat the output as an ordering, not a measurement: both columns it
    reports move with the tracer as well as with the animal, and on this data
    that dominates.  The number worth quoting comes from ``--correct``, which
    reports the transform the registration actually applied.  Write the
    confirmed IDs into ``motion.affected_ids`` in the config.

``--correct``
    Apply motion correction to the confirmed subset -- FALCON when it is
    installed, otherwise a translation-only rigid fallback that says so.  Both
    orders (motion correction then PVC, and PVC then motion correction) are
    produced, because the interpolation involved in resampling smooths the
    images and partly cancels the resolution recovery PVC is meant to give, so
    the order is tested rather than assumed.

``--qc``
    What the registration applied, per frame and summarised per scan, plus what
    the resampling cost in sharpness.  FALCON was developed for human
    whole-body PET; how it transfers to mouse data with sub-millimetre voxels
    and very low early-frame counts is checked here rather than taken on trust.
    Two tables come out: ``motion_qc.csv`` (one row per scan) and
    ``motion_transforms.csv`` (one row per frame).

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
    FalconRunner, interpolation_smoothing_cost, read_falcon_transforms, reference_frame_index,
    rigid_translation_correct, summarise_transforms, transforms_from_result,
)
from pvc_dlif.motion.detect import screen_dataset
from pvc_dlif.pvc.deconvolution import PVCSettings, make_backend
from pvc_dlif.pvc.psf import PSF
from pvc_dlif.pvc.runner import correct_scan

LOGGER = get_logger("stage.motion")


def main() -> int:
    parser = base_parser(__doc__ or "")
    parser.add_argument("--screen", action="store_true", help="rank scans as motion candidates")
    parser.add_argument("--correct", action="store_true", help="correct the confirmed subset")
    parser.add_argument("--qc", action="store_true", help="registration and smoothing quality report")
    parser.add_argument("--fast-screen", action="store_true",
                        help="centre-of-mass only: skips the per-frame registration, which is the "
                             "slow part and the only part that separates motion from tracer "
                             "redistribution. For a quick look, not for a number you quote.")
    parser.add_argument("--no-resume", action="store_true",
                        help="redo scans whose outputs already exist (default is to skip them)")
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
    shift_threshold = float(config.get("motion.detector.shift_threshold_mm", 0.5))
    reference_strategy = str(config.get("motion.reference_frame", "late_mean"))
    with_registration = not args.fast_screen

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

        table = screen_dataset(
            source(), threshold, cyclic_threshold,
            shift_threshold_mm=shift_threshold, reference=reference_strategy,
            with_registration=with_registration,
        )
        out = motion_dir / "motion_screening.csv"
        table.to_csv(out, index=False)
        write_provenance(out, "07_motion_screen", config)

        candidates = table[table["flagged"]]["scan_id"].tolist()
        print(table.head(20).to_string(index=False))
        print(f"\n{len(candidates)} of {len(table)} candidates: {candidates}")
        if with_registration:
            print(
                "\nTreat both columns as a ranking, not a measurement. max_displacement_mm "
                "follows the tracer from blood pool to liver and bladder and runs above 10 mm on "
                "every scan for that reason alone; p95_shift_mm is better but still registers on "
                "whatever is bright, which in the early frames is also the tracer. The number to "
                "quote comes from --correct, where the registration reports the transform it "
                "actually applied."
            )
        else:
            print(
                "\n--fast-screen: centre-of-mass only. That measure cannot separate the animal "
                "moving from the tracer redistributing, so it flags nearly everything. Re-run "
                "without --fast-screen before quoting anything."
            )
        print(
            "These are proposals from an automatic measure, not the affected subset. "
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

        # Whatever the config names for falconz; nulls are dropped so falconz
        # falls back to its own defaults rather than to ours.
        falcon_options = {
            key: config.get(f"motion.falcon.{key}")
            for key in ("registration", "multi_resolution_iterations", "start_frame", "mode",
                        "keep_intermediates")
        }
        falcon_options = {k: v for k, v in falcon_options.items() if v is not None}
        if "multi_resolution_iterations" in falcon_options:
            falcon_options["multi_resolution"] = falcon_options.pop("multi_resolution_iterations")

        rotation_suspect = float(config.get("motion.falcon.rotation_suspect_deg", 2.0))
        transform_rows: list[dict] = []
        qc_rows: list[dict] = []

        def _flush_tables() -> None:
            """Write what has been measured so far.

            Called after every scan rather than once at the end: a cohort run
            takes many hours, and a crash at scan sixty should not throw away
            the measurements from the first fifty-nine.
            """
            import pandas as pd

            if qc_rows:
                pd.DataFrame(qc_rows).to_csv(motion_dir / "motion_qc.csv", index=False)
            if transform_rows:
                pd.DataFrame(transform_rows).to_csv(
                    motion_dir / "motion_transforms.csv", index=False
                )

        psf = PSF.from_config(config)
        settings = PVCSettings(
            method=config.pvc_methods[0],
            iterations=config.iterations_primary,
            psf=psf,
            alpha=float(config.get("pvc.rvc_alpha", 1.5)),
        )
        backend = make_backend(args.backend, config.petpvc_exe)

        # A full cohort is many hours of work.  Skipping what is already on
        # disk turns an interrupted run into one that can simply be started
        # again, rather than one that has to begin from the first scan.
        def outputs_for(scan_id: str) -> list:
            paths = [motion_dir / "orig" / f"{scan_id}.nii.gz"]
            if args.correct:
                paths += [
                    motion_dir / settings.tag / f"{scan_id}.nii.gz",
                    motion_dir / f"pvcfirst_{settings.tag}" / f"{scan_id}.nii.gz",
                ]
            return paths

        for scan_id in subset:
            native_path = config.dir_native / f"{scan_id}.nii.gz"
            if not native_path.exists():
                LOGGER.warning("%s: no native NIfTI", scan_id)
                continue
            if not args.no_resume and all(p.exists() for p in outputs_for(scan_id)):
                LOGGER.info("%s: already corrected; skipping (use --no-resume to redo)", scan_id)
                continue
            scan = read_nifti_4d(native_path, scan_id)

            if args.dry_run:
                LOGGER.info("would motion-correct %s with %s", scan_id, method)
                continue

            # One reference frame per scan, chosen on the ORIGINAL series and
            # used for both orderings.  Recomputing it on the PVC-corrected
            # series picks a neighbouring frame often enough to matter: the two
            # orderings would then be aligned to different targets, and a
            # comparison meant to isolate the order of operations would carry a
            # second difference it never intended to test.
            ref_index = reference_frame_index(scan.data, reference_strategy)

            if method == "falcon":
                out_dir = motion_dir / "falcon" / scan_id
                corrected_path = falcon.run(
                    native_path, out_dir, reference_frame=ref_index, **falcon_options,
                )
                corrected = read_nifti_4d(corrected_path, scan_id).data
                per_frame = read_falcon_transforms(out_dir / "transforms")
                correction_info = {"method": "falcon", "output": str(corrected_path)}
            else:
                corrected, result = rigid_translation_correct(
                    scan.data, scan.geometry.voxel_mm, reference=ref_index,
                )
                per_frame = transforms_from_result(result)
                correction_info = result.to_dict()

            # What the registration itself did, per frame, in millimetres.
            #
            # There is deliberately no before/after image-based displacement
            # here.  Both measures that could provide one -- centre of mass, and
            # phase correlation against a late reference -- move with the tracer
            # rather than with the animal on this data, by an order of magnitude
            # more than the motion itself.  They reported 5-19 mm where the
            # registration reported 0.3-1.5 mm, and they did not change when the
            # correction was applied, because correction does not move the
            # tracer.  Reporting them beside the transforms put two contradictory
            # numbers in the same table.  The transforms are the measurement; the
            # smoothing cost below is what the correction cost to get it.
            applied = summarise_transforms(per_frame, rotation_suspect_deg=rotation_suspect)
            transform_rows.extend({"scan_id": scan_id, "method": method, **row}
                                  for row in per_frame)

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
                        pvc_first, scan.geometry.voxel_mm, reference=ref_index,
                    )
                else:
                    tmp = motion_dir / "falcon_pvcfirst" / scan_id
                    tmp.mkdir(parents=True, exist_ok=True)
                    tmp_input = tmp / f"{scan_id}.nii.gz"
                    write_nifti_4d(pvc_scan, tmp_input)
                    mc_after_pvc = read_nifti_4d(
                        falcon.run(
                            tmp_input, tmp, reference_frame=ref_index, **falcon_options,
                        ),
                        scan_id,
                    ).data
                write_nifti_4d(
                    DynamicScan(scan_id, mc_after_pvc, scan.frame_start_s, scan.frame_duration_s,
                                scan.geometry, native_path, {"order": "pvc_then_mc"}),
                    motion_dir / f"pvcfirst_{settings.tag}" / f"{scan_id}.nii.gz",
                )

            qc_rows.append(
                {
                    "scan_id": scan_id,
                    "method": method,
                    "reference_frame": ref_index,
                    # What the registration applied, in millimetres. Measured by
                    # the registration itself, not inferred from the images
                    # afterwards.
                    "applied_median_mm": applied["translation_median_mm"],
                    "applied_p95_mm": applied["translation_p95_mm"],
                    "applied_max_mm": applied["translation_max_mm"],
                    "applied_rotation_max_deg": applied["rotation_max_deg"],
                    "n_frames_registered": applied["n_frames"],
                    "n_frames_suspect": applied["n_suspect"],
                    "suspect_frames": applied["suspect_frames"],
                    "first_reliable_frame": applied["first_reliable_frame"],
                    **smoothing,
                }
            )
            _flush_tables()
            LOGGER.info(
                "%s: applied median %.2f mm, max %.2f mm, rotation max %.2f deg over %d frames; "
                "resampling kept %.0f%% of gradient energy%s",
                scan_id, applied["translation_median_mm"], applied["translation_max_mm"],
                applied["rotation_max_deg"], applied["n_frames"],
                100.0 * smoothing["gradient_energy_ratio"],
                f"; {applied['n_suspect']} frame(s) unreliable ({applied['suspect_frames']}) - "
                f"start_frame would be {applied['first_reliable_frame']}"
                if applied["n_suspect"] else "",
            )

        if qc_rows:
            import pandas as pd

            frame = pd.DataFrame(qc_rows)
            qc_path = motion_dir / "motion_qc.csv"
            frame.to_csv(qc_path, index=False)
            write_provenance(qc_path, "07_motion_correct", config, method=method)
            print(frame.to_string(index=False))

            if transform_rows:
                # One row per frame per scan: the table to plot and to quote.
                per_frame_path = motion_dir / "motion_transforms.csv"
                pd.DataFrame(transform_rows).to_csv(per_frame_path, index=False)
                write_provenance(per_frame_path, "07_motion_transforms", config, method=method)
                LOGGER.info("wrote %d per-frame transforms to %s",
                            len(transform_rows), per_frame_path)

            # No residual-motion warning: there is no measure on this data
            # that can tell leftover motion from the tracer moving, so any
            # "registration did not help" verdict here would be noise. The
            # evidence that correction worked is the transforms themselves, a
            # before/after look in a viewer, and the downstream comparison of
            # the mc_* conditions against their uncorrected counterparts.
            unreliable = frame[frame["n_frames_suspect"] > 0]
            if len(unreliable):
                LOGGER.warning(
                    "%d scan(s) had frames the registration could not fit: %s. Raise "
                    "motion.falcon.start_frame to the largest first_reliable_frame and redo those.",
                    len(unreliable), unreliable["scan_id"].tolist(),
                )
            rotated = frame[frame["applied_rotation_max_deg"] > 2.0 * rotation_suspect]
            if len(rotated):
                LOGGER.warning(
                    "%d scan(s) needed unusually large rotations: %s. Inspect these before use.",
                    len(rotated), rotated["scan_id"].tolist(),
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
