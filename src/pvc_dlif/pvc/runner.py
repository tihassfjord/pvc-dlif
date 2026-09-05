"""Batch PVC over the dataset: scans x methods x iteration counts.

The runner is resumable (an existing, complete output is skipped) and
parallel over frames within a scan, because a full pass is roughly
``n_scans x n_frames x n_settings`` deconvolutions and the iteration-count
sensitivity analysis multiplies that by three.

Alongside every corrected series it records noise and recovery diagnostics per
frame.  Those are what connect the dynamic results back to the recovery-noise
trade-off characterised on the phantom: the same deconvolution that raises the
recovery coefficient also raises the background %STD, and in the dynamic data
that trade-off is expected to vary strongly with frame count statistics.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from ..logging_utils import get_logger
from ..data.dicom_io import DynamicScan, read_nifti_4d, write_nifti_4d
from .deconvolution import DeconvolutionBackend, PVCSettings, make_backend
from .psf import PSF

LOGGER = get_logger(__name__)

__all__ = ["FrameDiagnostics", "correct_scan", "run_batch", "output_path_for"]


@dataclass
class FrameDiagnostics:
    """Per-frame summary of what the correction did.

    ``noise_pct_std`` is the coefficient of variation inside a background
    region, the dynamic analogue of the NEMA uniformity %STD.  ``blood_peak``
    tracks the maximum inside the blood-pool region, which is where the
    spill-out that PVC is meant to undo shows up.
    """

    frame: int
    time_min: float
    duration_s: float
    counts_proxy: float
    mean_before: float
    mean_after: float
    max_before: float
    max_after: float
    noise_pct_std_before: float
    noise_pct_std_after: float
    blood_peak_before: float
    blood_peak_after: float
    negative_fraction_after: float

    @property
    def noise_amplification(self) -> float:
        if self.noise_pct_std_before <= 0:
            return float("nan")
        return self.noise_pct_std_after / self.noise_pct_std_before

    @property
    def peak_recovery(self) -> float:
        if self.blood_peak_before <= 0:
            return float("nan")
        return self.blood_peak_after / self.blood_peak_before


def _background_mask(volume: np.ndarray, body_fraction: float = 0.15) -> np.ndarray:
    """A coarse in-body, low-activity mask used for the noise estimate.

    Deliberately simple and identical before and after correction, so the
    before/after noise numbers are comparable: a segmentation that reacted to
    the correction would confound the measurement it is meant to make.
    """
    positive = volume[volume > 0]
    if positive.size == 0:
        return np.zeros_like(volume, dtype=bool)
    body_threshold = np.percentile(positive, 100 * (1 - body_fraction))
    body = volume > (0.1 * body_threshold)
    if body.sum() < 100:
        body = volume > 0
    in_body = volume[body]
    low, high = np.percentile(in_body, [20, 60])
    return body & (volume >= low) & (volume <= high)


def _blood_mask(reference_volume: np.ndarray, top_fraction: float = 0.001) -> np.ndarray:
    """The hottest voxels of an early frame: a stand-in for the blood pool.

    The real blood-pool VOI comes from the group's ``VOI_SUV`` files; this is
    only used for the per-frame diagnostic, and it is computed once from a
    fixed reference frame so it does not move between conditions.
    """
    flat = reference_volume.ravel()
    if flat.size == 0:
        return np.zeros_like(reference_volume, dtype=bool)
    k = max(int(flat.size * top_fraction), 8)
    threshold = np.partition(flat, -k)[-k]
    return reference_volume >= threshold


def _pct_std(volume: np.ndarray, mask: np.ndarray) -> float:
    values = volume[mask]
    if values.size < 2:
        return float("nan")
    mean = float(values.mean())
    if abs(mean) < 1e-12:
        return float("nan")
    return float(100.0 * values.std(ddof=1) / mean)


def output_path_for(pvc_root: Path, settings: PVCSettings, scan_id: str, psf_tag: str = "") -> Path:
    """``<pvc_root>/<tag>/<scan_id>.nii.gz`` for one correction setting."""
    tag = settings.tag + (f"_{psf_tag}" if psf_tag else "")
    return Path(pvc_root) / tag / f"{scan_id}.nii.gz"


def _correct_frames_chunk(args: tuple) -> tuple[int, np.ndarray]:
    """Worker entry point: correct one frame.  Top-level so it pickles."""
    (frame_index, volume, voxel_mm, method, iterations, fwhm_mm, alpha,
     disable_stop, backend_name, executable) = args
    settings = PVCSettings(
        method=method,
        iterations=iterations,
        psf=PSF(tuple(fwhm_mm)),
        alpha=alpha,
        disable_stopping_criterion=disable_stop,
    )
    backend = make_backend(backend_name, executable)
    return frame_index, backend.correct_volume(volume, voxel_mm, settings)


def correct_scan(
    scan: DynamicScan,
    settings: PVCSettings,
    backend: DeconvolutionBackend | None = None,
    workers: int = 1,
    backend_name: str = "petpvc",
    executable: Path | str | None = None,
    diagnostics: bool = True,
) -> tuple[np.ndarray, list[FrameDiagnostics], list[str]]:
    """Apply one PVC setting to a whole dynamic scan.

    Returns the corrected series, per-frame diagnostics and any warnings worth
    carrying into the results (for example a PSF that is undersampled on this
    grid).
    """
    warnings = settings.psf.check_sampling(scan.geometry.voxel_mm, scan.scan_id)

    series = scan.data
    voxel_mm = scan.geometry.voxel_mm
    corrected = np.empty_like(series, dtype=np.float32)

    if workers > 1:
        payload = [
            (t, series[t], voxel_mm, settings.method, settings.iterations,
             tuple(settings.psf.fwhm_mm), settings.alpha,
             settings.disable_stopping_criterion, backend_name,
             str(executable) if executable else None)
            for t in range(series.shape[0])
        ]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_correct_frames_chunk, item) for item in payload]
            done = 0
            for future in as_completed(futures):
                index, result = future.result()
                corrected[index] = result
                done += 1
                if done % 10 == 0 or done == len(futures):
                    LOGGER.info("  %s %s: %d/%d frames", scan.scan_id, settings.tag, done, len(futures))
    else:
        backend = backend or make_backend(backend_name, executable)
        for t in range(series.shape[0]):
            corrected[t] = backend.correct_volume(series[t], voxel_mm, settings)
            if (t + 1) % 10 == 0 or t == series.shape[0] - 1:
                LOGGER.info("  %s %s: %d/%d frames", scan.scan_id, settings.tag, t + 1, series.shape[0])

    frame_diagnostics: list[FrameDiagnostics] = []
    if diagnostics:
        reference_frame = int(np.argmax(series.reshape(series.shape[0], -1).max(axis=1)))
        blood = _blood_mask(series[reference_frame])
        counts = scan.frame_counts_proxy()
        times = scan.frame_mid_min
        for t in range(series.shape[0]):
            background = _background_mask(series[t])
            frame_diagnostics.append(
                FrameDiagnostics(
                    frame=t,
                    time_min=float(times[t]),
                    duration_s=float(scan.frame_duration_s[t]),
                    counts_proxy=float(counts[t]),
                    mean_before=float(series[t].mean()),
                    mean_after=float(corrected[t].mean()),
                    max_before=float(series[t].max()),
                    max_after=float(corrected[t].max()),
                    noise_pct_std_before=_pct_std(series[t], background),
                    noise_pct_std_after=_pct_std(corrected[t], background),
                    blood_peak_before=float(series[t][blood].max()) if blood.any() else float("nan"),
                    blood_peak_after=float(corrected[t][blood].max()) if blood.any() else float("nan"),
                    negative_fraction_after=float((corrected[t] < 0).mean()),
                )
            )

    return corrected, frame_diagnostics, warnings


def run_batch(
    scans: Iterable[tuple[str, Path]],
    settings_list: Sequence[PVCSettings],
    out_root: Path,
    backend_name: str = "petpvc",
    executable: Path | str | None = None,
    workers: int = 1,
    resume: bool = True,
    psf_tag: str = "",
) -> dict[str, Any]:
    """Correct many scans under many settings, writing 4D NIfTI per combination.

    ``scans`` yields ``(scan_id, native_nifti_path)``.  Returns a report that
    lists what was written, what was skipped and what failed, so a long run can
    be inspected without re-reading the logs.
    """
    out_root = Path(out_root)
    report: dict[str, Any] = {"written": [], "skipped": [], "failed": [], "warnings": []}

    backend = None if workers > 1 else make_backend(backend_name, executable)

    for scan_id, native_path in scans:
        for settings in settings_list:
            target = output_path_for(out_root, settings, scan_id, psf_tag)
            diag_path = target.with_suffix("").with_suffix(".diagnostics.json")

            if resume and target.exists() and diag_path.exists():
                report["skipped"].append(str(target))
                LOGGER.info("skip %s (exists)", target.name)
                continue

            try:
                scan = read_nifti_4d(Path(native_path), scan_id)
                LOGGER.info("PVC %s with %s (%s)", scan_id, settings.tag, settings.psf)
                corrected, diagnostics, warnings = correct_scan(
                    scan, settings, backend=backend, workers=workers,
                    backend_name=backend_name, executable=executable,
                )
                corrected_scan = DynamicScan(
                    scan_id=scan_id,
                    data=corrected,
                    frame_start_s=scan.frame_start_s,
                    frame_duration_s=scan.frame_duration_s,
                    geometry=scan.geometry,
                    source=native_path,
                    meta={**(scan.meta or {}), "pvc": settings.tag, "psf_fwhm_mm": list(settings.psf.fwhm_mm)},
                )
                write_nifti_4d(corrected_scan, target)

                diag_path.parent.mkdir(parents=True, exist_ok=True)
                with open(diag_path, "w", encoding="utf-8") as handle:
                    json.dump(
                        {
                            "scan_id": scan_id,
                            "settings": {
                                "method": settings.method,
                                "iterations": settings.iterations,
                                "alpha": settings.alpha,
                                "fwhm_mm": list(settings.psf.fwhm_mm),
                                "psf_tag": psf_tag or None,
                            },
                            "backend": backend_name,
                            "warnings": warnings,
                            "frames": [asdict(d) for d in diagnostics],
                        },
                        handle,
                        indent=2,
                    )
                report["written"].append(str(target))
                report["warnings"].extend(warnings)
            except Exception as exc:  # noqa: BLE001 - one bad scan must not stop the batch
                LOGGER.exception("PVC failed for %s / %s", scan_id, settings.tag)
                report["failed"].append({"scan_id": scan_id, "settings": settings.tag, "error": str(exc)})

    LOGGER.info(
        "PVC batch: %d written, %d skipped, %d failed",
        len(report["written"]), len(report["skipped"]), len(report["failed"]),
    )
    return report
