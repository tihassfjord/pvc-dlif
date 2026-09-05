"""Stage 02 -- apply RL and RVC deconvolution to every dynamic scan.

Runs on the native NIfTI series written by stage 01, with the measured PSF, a
fixed iteration count per run, and PETPVC's internal stopping criterion
disabled so the iteration count is exactly what was asked for.

By default it runs the primary iteration count for every method.  ``--sweep``
adds the rest of the iteration grid, which is the sensitivity analysis reported
alongside the main result rather than a search for the best setting.

    python scripts/02_run_pvc.py                      # primary setting only
    python scripts/02_run_pvc.py --sweep              # plus the iteration grid
    python scripts/02_run_pvc.py --psf-sensitivity    # plus the PSF-mismatch check
"""

from __future__ import annotations

import json
import sys

from _common import base_parser, start

from pvc_dlif.data.manifest import load_manifest
from pvc_dlif.logging_utils import get_logger, write_provenance
from pvc_dlif.pvc.deconvolution import PVCSettings, PetpvcBackend
from pvc_dlif.pvc.psf import PSF
from pvc_dlif.pvc.runner import run_batch

LOGGER = get_logger("stage.pvc")


def main() -> int:
    parser = base_parser(__doc__ or "")
    parser.add_argument("--sweep", action="store_true", help="run the full iteration grid")
    parser.add_argument("--psf-sensitivity", action="store_true",
                        help="also run the scaled-PSF variants from sensitivity.psf_scale")
    parser.add_argument("--backend", choices=["petpvc", "numpy"], default="petpvc")
    parser.add_argument("--workers", type=int, default=None, help="override pvc.workers")
    parser.add_argument("--no-resume", action="store_true", help="recompute even if output exists")
    args = parser.parse_args()
    config = start(args, "02_pvc")

    manifest = load_manifest(config.work / "manifest.json")
    native_dir = config.dir_native

    scans: list[tuple[str, object]] = []
    for entry in manifest:
        if not entry.usable:
            continue
        path = native_dir / f"{entry.scan_id}.nii.gz"
        if path.exists():
            scans.append((entry.scan_id, path))
        else:
            LOGGER.warning("%s has no native NIfTI; run stage 01", entry.scan_id)

    if args.ids:
        wanted = set(args.ids)
        scans = [s for s in scans if s[0] in wanted]
    if args.limit:
        scans = scans[: args.limit]
    if not scans:
        raise SystemExit("No scans to correct. Run stages 00 and 01 first.")

    psf = PSF.from_config(config)
    iterations = config.iterations_grid if args.sweep else [config.iterations_primary]
    alpha = float(config.get("pvc.rvc_alpha", 1.5))
    disable_stop = bool(config.get("pvc.disable_stopping_criterion", True))

    settings = [
        PVCSettings(method=method, iterations=k, psf=psf, alpha=alpha,
                    disable_stopping_criterion=disable_stop)
        for method in config.pvc_methods
        for k in iterations
    ]

    if args.backend == "petpvc" and not PetpvcBackend.is_available(config.petpvc_exe):
        LOGGER.error(
            "PETPVC is not available. Install it (conda install -c conda-forge petpvc) or set "
            "paths.petpvc_exe. Use --backend numpy only for development: those results are not "
            "the PETPVC results validated on the phantom."
        )
        return 1

    LOGGER.info(
        "%d scans x %d settings = %d corrections, %s, %s",
        len(scans), len(settings), len(scans) * len(settings), psf, args.backend,
    )
    if args.dry_run:
        for scan_id, _ in scans[:5]:
            for setting in settings:
                print(f"{scan_id}: {setting.tag}")
        return 0

    workers = args.workers if args.workers is not None else int(config.get("pvc.workers", 1))
    resume = not args.no_resume and bool(config.get("pvc.resume", True))

    report = run_batch(
        scans=scans,
        settings_list=settings,
        out_root=config.dir_pvc,
        backend_name=args.backend,
        executable=config.petpvc_exe,
        workers=workers,
        resume=resume,
    )

    if args.psf_sensitivity:
        for scale in config.get("sensitivity.psf_scale", [1.0]):
            if abs(float(scale) - 1.0) < 1e-9:
                continue
            scaled_settings = [
                PVCSettings(method=s.method, iterations=s.iterations, psf=psf.scaled(float(scale)),
                            alpha=alpha, disable_stopping_criterion=disable_stop)
                for s in settings if s.iterations == config.iterations_primary
            ]
            LOGGER.info("PSF sensitivity: FWHM x %.2f", float(scale))
            scaled_report = run_batch(
                scans=scans, settings_list=scaled_settings, out_root=config.dir_pvc,
                backend_name=args.backend, executable=config.petpvc_exe,
                workers=workers, resume=resume, psf_tag=f"psf{float(scale):.2f}".replace(".", "p"),
            )
            for key in ("written", "skipped", "failed", "warnings"):
                report[key].extend(scaled_report[key])

    report_path = config.dir_results / "pvc_batch_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_provenance(report_path, "02_pvc", config, backend=args.backend, psf=list(psf.fwhm_mm))

    print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in report.items()}, indent=2))
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
