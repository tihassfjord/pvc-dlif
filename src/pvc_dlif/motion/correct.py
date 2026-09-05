"""Inter-frame motion correction.

The primary tool is FALCON, which the project description names.  Two things
about it are checked rather than assumed:

* FALCON was developed and validated mainly on human whole-body PET.  How well
  it transfers to mouse data with sub-millimetre voxels and very low counts in
  the early frames is an open question, so registration quality is measured
  before the corrected data are used.
* the interpolation involved in resampling smooths the images, which partly
  cancels the resolution recovery PVC is meant to give.  That is why the order
  of motion correction and PVC is tested rather than assumed, and why the
  smoothing cost is quantified in :func:`interpolation_smoothing_cost`.

A rigid, translation-based fallback is included so the subset analysis can run
without FALCON installed.  It is a fallback, not an equivalent: it corrects
translation only, and it says so in its output.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..logging_utils import get_logger

LOGGER = get_logger(__name__)

__all__ = [
    "MotionCorrectionResult", "FalconRunner", "rigid_translation_correct",
    "reference_frame_index", "interpolation_smoothing_cost",
]


@dataclass
class MotionCorrectionResult:
    """Corrected series plus what was done to produce it."""

    method: str
    reference_frame: int
    shifts_mm: list[list[float]]
    residual_displacement_mm: list[float]
    max_shift_mm: float
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def reference_frame_index(series: np.ndarray, strategy: str = "late_mean") -> int:
    """Pick the frame every other frame is registered to.

    ``late_mean`` uses the frame closest to the average of the second half of
    the scan: those frames have the best count statistics and the most stable
    anatomy, so they make a better target than the noisy early bolus frames.
    """
    n_t = series.shape[0]
    if strategy == "first":
        return 0
    if strategy == "max_counts":
        return int(np.argmax(series.reshape(n_t, -1).sum(axis=1)))
    if strategy == "late_mean":
        late = series[n_t // 2:]
        template = late.mean(axis=0)
        scores = [
            float(np.corrcoef(series[t].ravel(), template.ravel())[0, 1])
            for t in range(n_t // 2, n_t)
        ]
        return int(n_t // 2 + int(np.argmax(scores)))
    raise ValueError(f"unknown reference strategy {strategy!r}")


def _phase_shift(moving: np.ndarray, fixed: np.ndarray, upsample: int = 10) -> np.ndarray:
    """Sub-voxel translation between two volumes by phase correlation."""
    try:
        from skimage.registration import phase_cross_correlation

        shift, _, _ = phase_cross_correlation(
            fixed, moving, upsample_factor=upsample, normalization=None
        )
        return np.asarray(shift, dtype=float)
    except ImportError:
        # Integer-voxel fallback via the cross-power spectrum.
        product = np.fft.fftn(fixed) * np.conj(np.fft.fftn(moving))
        magnitude = np.abs(product)
        correlation = np.fft.ifftn(product / np.where(magnitude > 1e-12, magnitude, 1.0)).real
        peak = np.unravel_index(np.argmax(correlation), correlation.shape)
        shift = np.array(peak, dtype=float)
        for axis, size in enumerate(correlation.shape):
            if shift[axis] > size // 2:
                shift[axis] -= size
        return shift


def rigid_translation_correct(
    series: np.ndarray,
    voxel_mm: Sequence[float],
    reference: int | str = "late_mean",
    smooth_sigma_voxels: float = 1.0,
) -> tuple[np.ndarray, MotionCorrectionResult]:
    """Align every frame to a reference by translation only.

    Registration is estimated on smoothed copies -- the early frames are far too
    noisy to register raw -- but the shift is applied to the original data, so
    the correction does not itself blur the series beyond the single
    interpolation.
    """
    from scipy.ndimage import gaussian_filter, shift as ndshift

    if series.ndim != 4:
        raise ValueError(f"expected (T, Z, Y, X); got {series.shape}")

    ref_index = reference if isinstance(reference, int) else reference_frame_index(series, reference)
    voxel_zyx = np.array([float(voxel_mm[2]), float(voxel_mm[1]), float(voxel_mm[0])])

    fixed = gaussian_filter(series[ref_index].astype(np.float32), smooth_sigma_voxels)
    corrected = np.empty_like(series, dtype=np.float32)
    shifts: list[list[float]] = []
    residuals: list[float] = []
    notes = [
        "translation-only rigid correction (fallback); rotation and non-rigid motion are not corrected",
    ]

    for t in range(series.shape[0]):
        if t == ref_index:
            corrected[t] = series[t]
            shifts.append([0.0, 0.0, 0.0])
            residuals.append(0.0)
            continue

        moving = gaussian_filter(series[t].astype(np.float32), smooth_sigma_voxels)
        shift_voxels = _phase_shift(moving, fixed)
        corrected[t] = ndshift(series[t], shift_voxels, order=1, mode="nearest")

        shift_mm = shift_voxels * voxel_zyx
        shifts.append([float(v) for v in shift_mm[::-1]])   # store as (x, y, z)
        residuals.append(float(np.linalg.norm(shift_mm)))

    result = MotionCorrectionResult(
        method="rigid_translation",
        reference_frame=int(ref_index),
        shifts_mm=shifts,
        residual_displacement_mm=residuals,
        max_shift_mm=float(np.max(residuals)) if residuals else 0.0,
        notes=notes,
    )
    LOGGER.info(
        "rigid correction against frame %d: max shift %.2f mm", ref_index, result.max_shift_mm
    )
    return corrected, result


class FalconRunner:
    """Wrapper around the FALCON motion-correction command line tool."""

    def __init__(self, executable: Path | str | None = None):
        self.executable = self._resolve(executable)

    @staticmethod
    def _resolve(executable: Path | str | None) -> str | None:
        if executable:
            path = Path(executable)
            if path.exists():
                return str(path)
            found = shutil.which(str(executable))
            if found:
                return found
            LOGGER.warning("FALCON executable not found at %s", executable)
            return None
        for name in ("falcon", "falconz", "fal"):
            found = shutil.which(name)
            if found:
                return found
        return None

    @property
    def available(self) -> bool:
        return self.executable is not None

    def run(
        self,
        input_nifti: Path,
        output_dir: Path,
        reference_frame: int = -1,
        registration: str = "rigid",
        multi_resolution: str = "2x1",
        extra_args: Sequence[str] = (),
    ) -> Path:
        """Run FALCON on a 4D NIfTI and return the corrected 4D file.

        The FALCON command line has changed between releases, so the invocation
        is kept in one place and the raw stdout/stderr is written next to the
        output.  If FALCON's flags differ in the installed version, adjust here
        rather than scattering variants through the pipeline.
        """
        if not self.available:
            raise RuntimeError(
                "FALCON is not installed or not on PATH. Set paths.falcon_exe in the config, "
                "or use the rigid-translation fallback for a preliminary look."
            )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            str(self.executable),
            "-d", str(Path(input_nifti).parent),
            "-r", registration,
            "-i", str(reference_frame),
            "-sf", multi_resolution,
            *extra_args,
        ]
        LOGGER.info("Running FALCON: %s", " ".join(cmd))
        result = subprocess.run(cmd, cwd=str(output_dir), capture_output=True, text=True, check=False)

        (output_dir / "falcon_stdout.txt").write_text(result.stdout or "", encoding="utf-8")
        (output_dir / "falcon_stderr.txt").write_text(result.stderr or "", encoding="utf-8")

        if result.returncode != 0:
            raise RuntimeError(
                f"FALCON failed (exit {result.returncode}). See {output_dir}/falcon_stderr.txt"
            )

        candidates = sorted(output_dir.rglob("*moco*.nii*")) or sorted(output_dir.rglob("*.nii.gz"))
        if not candidates:
            raise RuntimeError(f"FALCON produced no NIfTI output under {output_dir}")
        return candidates[0]


def interpolation_smoothing_cost(
    original: np.ndarray,
    corrected: np.ndarray,
    voxel_mm: Sequence[float],
) -> dict[str, float]:
    """Quantify how much resolution the resampling cost.

    Motion correction resamples, and interpolation is a low-pass filter: it
    smooths the images and so partly cancels the resolution recovery PVC gives.
    Rather than assert that this is small, the gradient energy and the
    high-frequency power of the two series are compared -- both drop when a
    volume is smoothed, and the ratios say by how much.
    """
    def gradient_energy(series: np.ndarray) -> float:
        total = 0.0
        for t in range(series.shape[0]):
            grads = np.gradient(series[t].astype(np.float64))
            total += float(np.mean(sum(g ** 2 for g in grads)))
        return total / max(series.shape[0], 1)

    def high_frequency_power(series: np.ndarray) -> float:
        total = 0.0
        for t in range(series.shape[0]):
            spectrum = np.abs(np.fft.fftn(series[t].astype(np.float64)))
            spectrum = np.fft.fftshift(spectrum)
            centre = tuple(s // 2 for s in spectrum.shape)
            radius = min(spectrum.shape) // 4
            mask = np.ones(spectrum.shape, dtype=bool)
            slicer = tuple(slice(max(c - radius, 0), c + radius) for c in centre)
            mask[slicer] = False
            denom = float(spectrum.sum())
            total += float(spectrum[mask].sum() / denom) if denom > 0 else 0.0
        return total / max(series.shape[0], 1)

    ge_before, ge_after = gradient_energy(original), gradient_energy(corrected)
    hf_before, hf_after = high_frequency_power(original), high_frequency_power(corrected)

    return {
        "gradient_energy_before": ge_before,
        "gradient_energy_after": ge_after,
        "gradient_energy_ratio": float(ge_after / ge_before) if ge_before > 0 else float("nan"),
        "high_frequency_share_before": hf_before,
        "high_frequency_share_after": hf_after,
        "high_frequency_ratio": float(hf_after / hf_before) if hf_before > 0 else float("nan"),
    }
