"""Richardson-Lucy and Reblurred Van Cittert deconvolution.

Two interchangeable back-ends:

* :class:`PetpvcBackend` shells out to the PETPVC toolbox, which is what the
  project thesis validated on the NEMA NU 4-2008 phantom and therefore what the
  thesis results should be produced with.
* :class:`NumpyBackend` implements the same two algorithms directly.  It exists
  so the pipeline is testable without the binary, so the behaviour of the
  correction can be inspected frame by frame, and so a disagreement between the
  two can be caught rather than assumed away.

Both run a *fixed* number of iterations with no internal stopping criterion.
That is deliberate: varying the iteration count across frames would introduce a
time-dependent processing bias into the very quantity under study -- the shape
of the time-activity curve.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ..logging_utils import get_logger
from .psf import PSF

LOGGER = get_logger(__name__)

__all__ = ["DeconvolutionBackend", "PetpvcBackend", "NumpyBackend", "make_backend", "PVCSettings"]


@dataclass(frozen=True)
class PVCSettings:
    """Everything that defines one correction, other than the image itself."""

    method: str                 # "RL" or "RVC"
    iterations: int
    psf: PSF
    alpha: float = 1.5          # RVC relaxation parameter
    disable_stopping_criterion: bool = True

    def __post_init__(self) -> None:
        method = self.method.upper()
        if method not in {"RL", "RVC", "VC"}:
            raise ValueError(f"method must be RL, RVC or VC; got {self.method!r}")
        object.__setattr__(self, "method", method)
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")

    @property
    def tag(self) -> str:
        return f"{self.method.lower()}_i{self.iterations}"


class DeconvolutionBackend(ABC):
    """Applies one :class:`PVCSettings` to one 3D volume."""

    name = "abstract"

    @abstractmethod
    def correct_volume(
        self,
        volume: np.ndarray,
        voxel_mm: Sequence[float],
        settings: PVCSettings,
    ) -> np.ndarray:
        ...

    def correct_series(
        self,
        series: np.ndarray,
        voxel_mm: Sequence[float],
        settings: PVCSettings,
        progress: bool = False,
    ) -> np.ndarray:
        """Apply the correction independently to every frame of ``(T, Z, Y, X)``."""
        if series.ndim != 4:
            raise ValueError(f"expected a 4D series, got shape {series.shape}")
        out = np.empty_like(series, dtype=np.float32)
        for t in range(series.shape[0]):
            out[t] = self.correct_volume(series[t], voxel_mm, settings)
            if progress and (t % 10 == 0 or t == series.shape[0] - 1):
                LOGGER.info("  frame %d/%d", t + 1, series.shape[0])
        return out


# --------------------------------------------------------------------------- #
# PETPVC
# --------------------------------------------------------------------------- #
class PetpvcBackend(DeconvolutionBackend):
    """Runs the PETPVC command line tool on one frame at a time.

    PETPVC is given the FWHM in millimetres and reads the voxel size from the
    NIfTI header, so the mm -> voxel conversion is its responsibility, not ours.
    Frames are written as individual 3D volumes because PETPVC's deconvolution
    methods operate on 3D input.
    """

    name = "petpvc"

    def __init__(self, executable: Path | str | None = None, work_dir: Path | None = None):
        self.executable = self._resolve(executable)
        self.work_dir = Path(work_dir) if work_dir else None
        if self.work_dir:
            self.work_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _resolve(executable: Path | str | None) -> str:
        if executable is not None:
            path = Path(executable)
            if path.exists():
                return str(path)
            found = shutil.which(str(executable))
            if found:
                return found
            raise FileNotFoundError(f"PETPVC executable not found: {executable}")
        found = shutil.which("petpvc")
        if not found:
            raise FileNotFoundError(
                "PETPVC not found on PATH. Install it (conda install -c conda-forge petpvc) "
                "or set paths.petpvc_exe in the config."
            )
        return found

    @staticmethod
    def is_available(executable: Path | str | None = None) -> bool:
        try:
            PetpvcBackend._resolve(executable)
            return True
        except FileNotFoundError:
            return False

    def build_command(
        self,
        input_path: Path,
        output_path: Path,
        settings: PVCSettings,
    ) -> list[str]:
        fx, fy, fz = settings.psf.fwhm_mm
        cmd = [
            self.executable,
            "-i", str(input_path),
            "-o", str(output_path),
            "-p", settings.method,
            "-x", f"{fx:.6g}",
            "-y", f"{fy:.6g}",
            "-z", f"{fz:.6g}",
            # -k is PETPVC's deconvolution iteration count (RL, VC, RVC).
            "-k", str(settings.iterations),
        ]
        if settings.method in {"RVC", "VC"}:
            cmd += ["-a", f"{settings.alpha:.6g}"]
        if settings.disable_stopping_criterion:
            # A stopping criterion of zero means the loop always runs to -k.
            cmd += ["-s", "0"]
        return cmd

    def correct_volume(
        self,
        volume: np.ndarray,
        voxel_mm: Sequence[float],
        settings: PVCSettings,
    ) -> np.ndarray:
        import nibabel as nib

        volume = np.asarray(volume, dtype=np.float32)
        affine = np.diag([*(float(v) for v in voxel_mm), 1.0])

        with tempfile.TemporaryDirectory(prefix="petpvc_", dir=str(self.work_dir) if self.work_dir else None) as tmp:
            tmp_path = Path(tmp)
            in_path = tmp_path / "frame.nii.gz"
            out_path = tmp_path / "frame_pvc.nii.gz"

            # PETPVC reads axes as (x, y, z); our arrays are (z, y, x).
            nib.save(nib.Nifti1Image(np.ascontiguousarray(volume.transpose(2, 1, 0)), affine), str(in_path))

            cmd = self.build_command(in_path, out_path, settings)
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if result.returncode != 0 or not out_path.exists():
                raise RuntimeError(
                    "PETPVC failed (exit %d)\ncommand: %s\nstdout: %s\nstderr: %s"
                    % (result.returncode, " ".join(cmd), result.stdout.strip(), result.stderr.strip())
                )
            corrected = np.asarray(nib.load(str(out_path)).get_fdata(dtype=np.float32))

        return np.ascontiguousarray(corrected.transpose(2, 1, 0), dtype=np.float32)

    def version(self) -> str:
        result = subprocess.run([self.executable, "--version"], capture_output=True, text=True, check=False)
        return (result.stdout or result.stderr).strip().splitlines()[0] if (result.stdout or result.stderr) else "unknown"


# --------------------------------------------------------------------------- #
# Reference implementation
# --------------------------------------------------------------------------- #
class NumpyBackend(DeconvolutionBackend):
    """Direct implementation of RL and reblurred Van Cittert.

    Richardson-Lucy, with :math:`h` the PSF and :math:`g` the measured image::

        f_{k+1} = f_k * ( h^T (x) [ g / (h (x) f_k) ] )

    Reblurred Van Cittert, with relaxation :math:`\\alpha`::

        f_{k+1} = f_k + alpha * h^T (x) [ g - h (x) f_k ]

    The reblurring (the extra :math:`h^T` convolution of the residual) is what
    keeps VC stable at low count levels; it is the variant PETPVC calls RVC.
    The Gaussian PSF is symmetric, so :math:`h^T = h`.
    """

    name = "numpy"

    def correct_volume(
        self,
        volume: np.ndarray,
        voxel_mm: Sequence[float],
        settings: PVCSettings,
    ) -> np.ndarray:
        image = np.asarray(volume, dtype=np.float64)
        blur = lambda v: settings.psf.blur(v, voxel_mm).astype(np.float64)  # noqa: E731

        if settings.method == "RL":
            eps = 1e-9
            estimate = np.maximum(image, 0.0).copy()
            # A uniform positive start avoids zeros locking the multiplicative
            # update; scale it to the image mean so convergence is comparable.
            floor = float(np.mean(estimate)) * 1e-6
            estimate = np.maximum(estimate, floor if floor > 0 else eps)
            observed = np.maximum(image, 0.0)
            for _ in range(settings.iterations):
                predicted = blur(estimate)
                ratio = observed / np.maximum(predicted, eps)
                estimate = estimate * blur(ratio)
                estimate = np.maximum(estimate, 0.0)
            return estimate.astype(np.float32)

        # RVC / VC
        estimate = image.copy()
        for _ in range(settings.iterations):
            residual = image - blur(estimate)
            if settings.method == "RVC":
                residual = blur(residual)
            estimate = estimate + settings.alpha * residual
        return estimate.astype(np.float32)


def make_backend(
    prefer: str = "petpvc",
    executable: Path | str | None = None,
    work_dir: Path | None = None,
) -> DeconvolutionBackend:
    """Build the requested back-end, falling back with a clear warning.

    The thesis results should come from PETPVC; the fallback exists so that
    development and testing are not blocked, and it says so loudly.
    """
    prefer = prefer.lower()
    if prefer == "numpy":
        return NumpyBackend()
    try:
        return PetpvcBackend(executable, work_dir)
    except FileNotFoundError as exc:
        LOGGER.warning(
            "%s -- falling back to the in-package implementation. Results produced this way "
            "are NOT the PETPVC results validated in the project thesis; install PETPVC "
            "before generating thesis numbers.", exc,
        )
        return NumpyBackend()
