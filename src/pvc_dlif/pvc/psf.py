"""The system point spread function used by partial volume correction.

The FWHM values come from the point-source acquisition analysed in the project
thesis (0.864 x 0.874 x 0.994 mm on the LabPET8).  Two assumptions are carried
over and are worth stating wherever this module is used:

* the PSF is treated as spatially invariant, although it broadens towards the
  edge of the field of view.  The blood-pool structures of interest lie near
  the centre, so this is expected to be acceptable, but it is a limitation.
* it was measured on a *static* point source and is applied here to dynamic
  frames.  Both RL and RVC are sensitive to PSF mismatch, RL more so, which is
  why :func:`scaled` exists: the sensitivity analysis perturbs the FWHM and
  reports how much the result moves.

The module also converts FWHM in millimetres to voxels, which is the only
place that conversion should happen -- getting it wrong silently changes the
strength of the deconvolution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..logging_utils import get_logger

LOGGER = get_logger(__name__)

__all__ = ["PSF", "FWHM_TO_SIGMA", "SIGMA_TO_FWHM"]

FWHM_TO_SIGMA = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))  # ~0.42466
SIGMA_TO_FWHM = 1.0 / FWHM_TO_SIGMA


@dataclass(frozen=True)
class PSF:
    """A Gaussian PSF specified by its FWHM in millimetres, ordered ``(x, y, z)``."""

    fwhm_mm: tuple[float, float, float]

    def __post_init__(self) -> None:
        if len(self.fwhm_mm) != 3:
            raise ValueError("fwhm_mm must have three components (x, y, z)")
        if any(v <= 0 for v in self.fwhm_mm):
            raise ValueError(f"FWHM values must be positive, got {self.fwhm_mm}")

    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(cls, config) -> "PSF":
        return cls(tuple(float(v) for v in config.fwhm_mm))  # type: ignore[arg-type]

    def scaled(self, factor: float) -> "PSF":
        """The same PSF with every FWHM multiplied by ``factor``.

        Used by the PSF-mismatch sensitivity analysis.
        """
        return PSF(tuple(float(v) * float(factor) for v in self.fwhm_mm))  # type: ignore[arg-type]

    # ------------------------------------------------------------------ #
    @property
    def sigma_mm(self) -> tuple[float, float, float]:
        return tuple(v * FWHM_TO_SIGMA for v in self.fwhm_mm)  # type: ignore[return-value]

    def fwhm_voxels(self, voxel_mm: Sequence[float]) -> tuple[float, float, float]:
        """FWHM expressed in voxels on a grid with the given voxel size."""
        if len(voxel_mm) != 3:
            raise ValueError("voxel_mm must have three components (x, y, z)")
        return tuple(f / float(v) for f, v in zip(self.fwhm_mm, voxel_mm))  # type: ignore[return-value]

    def sigma_voxels(self, voxel_mm: Sequence[float]) -> tuple[float, float, float]:
        return tuple(v * FWHM_TO_SIGMA for v in self.fwhm_voxels(voxel_mm))  # type: ignore[return-value]

    # ------------------------------------------------------------------ #
    #: Below this many voxels per FWHM the blur is undersampled and
    #: deconvolution mostly amplifies noise.
    UNDERSAMPLED_FWHM_VOXELS = 1.5
    #: Between the two the sampling is marginal: worth recording, not alarming.
    MARGINAL_FWHM_VOXELS = 2.0

    def check_sampling(self, voxel_mm: Sequence[float], scan_id: str = "") -> list[str]:
        """Report when the grid cannot support meaningful deconvolution.

        A structure at or below the FWHM cannot be recovered reliably by
        deconvolution alone -- the project thesis showed exactly that for the
        1 mm rod.  The same logic applies to the sampling grid: if the PSF spans
        too few voxels, the blur is barely represented and deconvolving it
        mainly amplifies noise.

        Returns the problems rather than raising, so a caller can store them
        with the result.  Marginal sampling is logged but not returned, because
        it is the normal situation on this scanner -- 0.86 mm FWHM at 0.5 mm
        voxels is about 1.7 voxels -- and a warning on every frame would drown
        out the cases that matter.
        """
        problems: list[str] = []
        fwhm_vox = self.fwhm_voxels(voxel_mm)
        prefix = f"{scan_id}: " if scan_id else ""

        for axis, (fv, mm) in enumerate(zip(fwhm_vox, voxel_mm)):
            name = "xyz"[axis]
            detail = (
                f"{prefix}PSF FWHM along {name} is {fv:.2f} voxels "
                f"({self.fwhm_mm[axis]:.3f} mm at {float(mm):.3f} mm voxels)"
            )
            if fv < self.UNDERSAMPLED_FWHM_VOXELS:
                problems.append(
                    detail + ". The blur is undersampled on this grid, so deconvolution "
                    "mainly amplifies noise. Apply PVC on the reconstructed grid rather "
                    "than a downsampled one."
                )
            elif fv > 12.0:
                problems.append(
                    detail + ", which is very wide. Check that the FWHM and the voxel size "
                    "are in the same units."
                )
            elif fv < self.MARGINAL_FWHM_VOXELS:
                LOGGER.info(
                    "%s -- marginal sampling; recovery is limited by the grid as well as by "
                    "the method.", detail,
                )

        for message in problems:
            LOGGER.warning(message)
        return problems

    # ------------------------------------------------------------------ #
    def kernel(self, voxel_mm: Sequence[float], radius_sigma: float = 4.0) -> np.ndarray:
        """A normalised separable Gaussian kernel on the given grid.

        Used by the local (non-PETPVC) reference implementations and by the
        reblurring step of the fallback RVC, so the two paths use exactly the
        same PSF.
        """
        sigmas = self.sigma_voxels(voxel_mm)
        axes = []
        for sigma in sigmas:
            radius = max(int(math.ceil(radius_sigma * sigma)), 1)
            grid = np.arange(-radius, radius + 1, dtype=np.float64)
            values = np.exp(-0.5 * (grid / sigma) ** 2)
            axes.append(values / values.sum())
        kernel = np.einsum("i,j,k->ijk", axes[0], axes[1], axes[2])
        return kernel / kernel.sum()

    def blur(self, volume: np.ndarray, voxel_mm: Sequence[float]) -> np.ndarray:
        """Convolve a volume with this PSF (separable, edge-reflecting)."""
        from scipy.ndimage import gaussian_filter

        sigmas = self.sigma_voxels(voxel_mm)
        return gaussian_filter(np.asarray(volume, dtype=np.float32), sigma=sigmas, mode="reflect")

    # ------------------------------------------------------------------ #
    def __str__(self) -> str:
        x, y, z = self.fwhm_mm
        return f"PSF(FWHM = {x:.3f} x {y:.3f} x {z:.3f} mm)"
