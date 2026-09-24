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
import os
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
    "read_falcon_transforms", "transforms_from_result", "summarise_transforms",
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
        multi_resolution: str | None = None,
        start_frame: int | None = None,
        mode: str | None = None,
        keep_intermediates: bool = False,
        extra_args: Sequence[str] = (),
    ) -> Path:
        """Run FALCON on a 4D NIfTI and return the corrected 4D file.

        The flags below match falconz as installed:

            -d   directory holding the images to correct
            -rf  reference frame index (0-based)
            -sf  frame to start correcting from
            -r   rigid | affine | deformable
            -i   iterations per resolution level
            -m   cruise | dash
            -o   output 4D NIfTI

        Two things this gets right that are easy to get wrong.  ``-d`` takes a
        *directory*, and pointing it at the folder the input happens to live in
        would hand FALCON every other scan in that folder as well -- so the one
        series is copied into a private directory first.  And the output path
        is given explicitly with ``-o`` rather than recovered by globbing for
        ``*moco*``, which silently picks the wrong file as soon as a directory
        is reused.

        Optional arguments are omitted when not set, so falconz applies its own
        defaults instead of ours.  The full command and the raw stdout/stderr
        are written beside the output, so what actually ran is recoverable.
        """
        if not self.available:
            raise RuntimeError(
                "FALCON is not installed or not on PATH. Set paths.falcon_exe in the config, "
                "or use the rigid-translation fallback for a preliminary look."
            )

        input_nifti = Path(input_nifti)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # One scan per input directory: FALCON corrects everything it finds, so
        # pointing it at work/native would hand it all seventy.  A hard link
        # gives it a directory with exactly one series in it without a second
        # copy of the data on disk; copy only when linking is refused (a
        # different volume, or a filesystem without links).
        stage = output_dir / "input"
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir(parents=True)
        staged_input = stage / input_nifti.name
        try:
            os.link(input_nifti, staged_input)
        except OSError:
            shutil.copy2(input_nifti, staged_input)

        corrected = output_dir / f"{input_nifti.name.split('.')[0]}_moco.nii.gz"

        cmd = [str(self.executable), "-d", str(stage), "-r", registration]
        if reference_frame is not None and reference_frame >= 0:
            cmd += ["-rf", str(int(reference_frame))]
        if start_frame is not None:
            cmd += ["-sf", str(int(start_frame))]
        if multi_resolution:
            cmd += ["-i", str(multi_resolution)]
        if mode:
            cmd += ["-m", str(mode)]
        cmd += ["-o", str(corrected), *extra_args]

        # falconz prints emoji in its progress output.  With stdout captured
        # rather than attached to a terminal, Python picks the child's encoding
        # from the locale -- cp1252 on a Norwegian Windows install -- and the
        # run dies with UnicodeEncodeError on the first decorated line, before
        # it has looked at a single voxel.  Forcing UTF-8 on the child is the
        # whole fix; errors="replace" keeps a stray byte from doing it again.
        env = {
            **os.environ,
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
        }
        LOGGER.info("Running FALCON: %s", " ".join(cmd))
        result = subprocess.run(
            cmd, cwd=str(output_dir), capture_output=True, check=False,
            text=True, encoding="utf-8", errors="replace", env=env,
        )

        (output_dir / "falcon_command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
        (output_dir / "falcon_stdout.txt").write_text(result.stdout or "", encoding="utf-8")
        (output_dir / "falcon_stderr.txt").write_text(result.stderr or "", encoding="utf-8")

        if result.returncode != 0:
            raise RuntimeError(
                f"FALCON failed (exit {result.returncode}). See {output_dir}/falcon_stderr.txt"
            )

        if corrected.exists():
            self._tidy(output_dir, stage, keep_intermediates)
            return corrected
        # Older builds ignore -o and write where they like; fall back to a
        # search, but never into the staged input we just put there.
        candidates = [
            p for p in sorted(output_dir.rglob("*moco*.nii*")) + sorted(output_dir.rglob("*.nii.gz"))
            if stage not in p.parents
        ]
        if not candidates:
            raise RuntimeError(
                f"FALCON exited cleanly but produced no NIfTI under {output_dir}. "
                f"See falcon_stdout.txt."
            )
        return candidates[0]

    @staticmethod
    def _tidy(output_dir: Path, stage: Path, keep_intermediates: bool) -> None:
        """Keep the transforms, drop FALCON's scratch.

        A run leaves roughly 290 MB behind, of which about 250 MB is working
        data: every frame written out singly before correction, again after,
        the per-frame cross-correlation volumes, and a second copy of the
        merged result.  Over seventy scans and two orderings that is some 35 GB
        of files nothing reads again.

        The transforms are the exception and are moved up beside the output:
        they are a few hundred bytes each and they are the record of how far
        the animal actually moved, which is the number worth reporting.
        """
        run_dirs = sorted(output_dir.glob("FALCONZ-*"))
        for run_dir in run_dirs:
            transforms = run_dir / "transforms"
            if transforms.is_dir():
                target = output_dir / "transforms"
                if target.exists():
                    shutil.rmtree(target)
                shutil.move(str(transforms), str(target))
            if not keep_intermediates:
                shutil.rmtree(run_dir, ignore_errors=True)
        if not keep_intermediates and stage.exists():
            # Hard-linked, so this frees a directory entry and not the data.
            shutil.rmtree(stage, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Reading back what the correction actually did.
#
# This is the measurement worth reporting.  It is not an estimate made after
# the fact from the images: it is the transformation the registration chose and
# applied, in millimetres, per frame.  A homemade displacement measure has to
# separate the animal moving from the tracer redistributing and is easily
# fooled by the second; the registration's own parameters have no such problem,
# because a shift is all they can express.
# --------------------------------------------------------------------------- #


def _decompose(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """Translation in mm and rotation magnitude in degrees from a 4x4 rigid matrix."""
    translation = np.asarray(matrix[:3, 3], dtype=float)
    rotation = np.asarray(matrix[:3, :3], dtype=float)
    cosine = (float(np.trace(rotation)) - 1.0) / 2.0
    angle = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    return translation, angle


def read_falcon_transforms(transform_dir: Path) -> list[dict[str, Any]]:
    """Per-frame rigid transforms FALCON wrote, as millimetres and degrees.

    One ``vol_XXXX.nii.gz_rigid.mat`` per corrected frame; the reference frame
    has none, and is reported as an exact zero rather than being left out.
    """
    import re

    transform_dir = Path(transform_dir)
    rows: list[dict[str, Any]] = []
    for path in sorted(transform_dir.glob("*_rigid.mat")):
        match = re.search(r"vol_(\d+)", path.name)
        if not match:
            continue
        try:
            matrix = np.loadtxt(path)
        except (OSError, ValueError):
            LOGGER.warning("could not read transform %s", path)
            continue
        if matrix.shape != (4, 4):
            LOGGER.warning("transform %s is %s, expected 4x4", path, matrix.shape)
            continue
        translation, angle = _decompose(matrix)
        rows.append({
            "frame": int(match.group(1)),
            "tx_mm": float(translation[0]),
            "ty_mm": float(translation[1]),
            "tz_mm": float(translation[2]),
            "translation_mm": float(np.linalg.norm(translation)),
            "rotation_deg": angle,
        })
    rows.sort(key=lambda r: r["frame"])
    return rows


def transforms_from_result(result: MotionCorrectionResult) -> list[dict[str, Any]]:
    """The same per-frame table for the translation-only fallback.

    Given so the QC columns mean the same thing whichever method produced the
    correction.  Rotation is always zero here, which is exactly the fallback's
    limitation made visible rather than hidden.
    """
    rows: list[dict[str, Any]] = []
    for frame, (shift, magnitude) in enumerate(
        zip(result.shifts_mm, result.residual_displacement_mm)
    ):
        rows.append({
            "frame": frame,
            "tx_mm": float(shift[0]), "ty_mm": float(shift[1]), "tz_mm": float(shift[2]),
            "translation_mm": float(magnitude),
            "rotation_deg": 0.0,
        })
    return rows


def summarise_transforms(
    rows: Sequence[dict[str, Any]],
    rotation_suspect_deg: float = 2.0,
) -> dict[str, Any]:
    """Summarise a per-frame transform table, separating trustworthy frames.

    Before the bolus arrives there is almost no anatomy in a frame, and an
    intensity-based registration will still return *an* answer for it -- one
    fitted to noise.  Those failures are recognisable: a mouse in a holder does
    not rotate several degrees between neighbouring frames and then come back,
    so the rotation magnitude separates them from real motion far more cleanly
    than the translation does.  Frames above the threshold are counted and
    reported, and kept out of the summary statistics rather than inflating them.
    """
    if not rows:
        return {
            "n_frames": 0, "n_suspect": 0, "first_reliable_frame": None,
            "translation_median_mm": float("nan"), "translation_p95_mm": float("nan"),
            "translation_max_mm": float("nan"), "rotation_max_deg": float("nan"),
            "suspect_frames": "",
        }

    translation = np.array([r["translation_mm"] for r in rows], dtype=float)
    rotation = np.array([r["rotation_deg"] for r in rows], dtype=float)
    frames = np.array([r["frame"] for r in rows], dtype=int)
    suspect = rotation > rotation_suspect_deg

    trusted = translation[~suspect]
    trusted_rotation = rotation[~suspect]
    # Where the registration starts behaving: the first frame after the last
    # suspect one.  That is the value to give falconz as --start_frame.
    first_reliable = int(frames[suspect].max()) + 1 if suspect.any() else int(frames.min())

    return {
        "n_frames": int(len(rows)),
        "n_suspect": int(suspect.sum()),
        "suspect_frames": " ".join(str(f) for f in frames[suspect]),
        "first_reliable_frame": first_reliable,
        "translation_median_mm": float(np.median(trusted)) if trusted.size else float("nan"),
        "translation_p95_mm": float(np.percentile(trusted, 95)) if trusted.size else float("nan"),
        "translation_max_mm": float(trusted.max()) if trusted.size else float("nan"),
        "rotation_max_deg": float(trusted_rotation.max()) if trusted_rotation.size else float("nan"),
        "translation_max_including_suspect_mm": float(translation.max()),
    }


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
