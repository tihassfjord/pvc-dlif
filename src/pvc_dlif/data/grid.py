"""Mapping between the reconstructed voxel grid and the DLIF input grid.

The pretrained DLIF network consumes SUV volumes of shape ``(T, 96, 48, 48)``,
while the reconstruction and the measured PSF live on the native DICOM grid
(here 128 x 92 x 92 at 0.5 x 0.5 x 0.597 mm).  PVC has to be applied on the
native grid -- that is the grid the point-source PSF was measured on -- and the
corrected series then has to land on exactly the grid the network expects, or
the comparison against the pretrained model is confounded by a resampling
difference rather than by PVC.

The recipe the group used to build the distributed ``IMG_*.pkl`` files is not
part of the DLIF repository.  This module therefore does two things:

1. :func:`calibrate_grid` recovers the transform empirically, by fitting a
   crop window and a resampling factor per axis against the existing
   ``IMG_<ID>.pkl`` files, and reports how well the recovered transform
   reproduces them.
2. :class:`GridTransform` applies that transform to any native volume, so the
   PVC-corrected series is preprocessed identically to the baseline.

If the calibration does not reach a high agreement, that is a finding to act
on rather than paper over: confirm the recipe with the group before running the
comparison, since a preprocessing mismatch would show up as a spurious PVC
effect.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from ..logging_utils import get_logger

LOGGER = get_logger(__name__)

__all__ = ["GridTransform", "GridCalibration", "calibrate_grid", "load_grid_transform"]


def _resample_axis_profile(profile: np.ndarray, start: int, stop: int, out_len: int) -> np.ndarray:
    """Linear-interpolate ``profile[start:stop]`` onto ``out_len`` samples."""
    segment = profile[start:stop]
    if segment.size < 2:
        return np.zeros(out_len, dtype=float)
    src = np.linspace(0.0, 1.0, segment.size)
    dst = np.linspace(0.0, 1.0, out_len)
    return np.interp(dst, src, segment)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size != b.size or a.size < 2:
        return -1.0
    a_std, b_std = a.std(), b.std()
    if a_std < 1e-12 or b_std < 1e-12:
        return -1.0
    return float(np.corrcoef(a, b)[0, 1])


@dataclass
class GridTransform:
    """Crop-then-resample transform from the native grid to the DLIF grid.

    The transform is expressed on the DICOM axis order ``(Z, Y, X)``.

    Attributes
    ----------
    crop:
        ``((z0, z1), (y0, y1), (x0, x1))`` half-open windows in native voxels.
    out_shape:
        Target shape, ``(96, 48, 48)`` for this dataset.
    flip:
        Per-axis flip flags applied after cropping.
    order:
        Spline order for the resampling (1 = trilinear).
    scale:
        Multiplicative intensity factor applied after resampling.  Recovered by
        the calibration; 1.0 means the SUV scaling already matches.
    native_voxel_mm:
        Voxel size of the source grid, ``(x, y, z)`` in mm, kept for the record.
    """

    crop: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    out_shape: tuple[int, int, int] = (96, 48, 48)
    flip: tuple[bool, bool, bool] = (False, False, False)
    order: int = 1
    scale: float = 1.0
    native_voxel_mm: tuple[float, float, float] | None = None

    # ------------------------------------------------------------------ #
    @property
    def out_voxel_mm(self) -> tuple[float, float, float] | None:
        """Voxel size of the DLIF grid in ``(x, y, z)`` mm, if derivable."""
        if self.native_voxel_mm is None:
            return None
        vx, vy, vz = self.native_voxel_mm
        (z0, z1), (y0, y1), (x0, x1) = self.crop
        oz, oy, ox = self.out_shape
        return (
            vx * (x1 - x0) / ox,
            vy * (y1 - y0) / oy,
            vz * (z1 - z0) / oz,
        )

    # ------------------------------------------------------------------ #
    def apply(self, volume: np.ndarray) -> np.ndarray:
        """Map one native volume ``(Z, Y, X)`` onto the DLIF grid."""
        from scipy.ndimage import zoom as ndzoom

        if volume.ndim != 3:
            raise ValueError(f"expected a 3D volume, got shape {volume.shape}")

        cropped = crop_with_zero_padding(volume, self.crop).astype(np.float32, copy=False)
        for axis, do_flip in enumerate(self.flip):
            if do_flip:
                cropped = np.flip(cropped, axis=axis)

        if tuple(cropped.shape) == tuple(self.out_shape):
            # Pure crop: no resampling at all.  Taking this path rather than a
            # zoom by a factor of exactly 1.0 keeps the values bit-identical to
            # the reconstruction, which is what makes the DLIF input voxels the
            # same voxels the PSF was measured on.
            resampled = cropped
        else:
            factors = tuple(o / max(s, 1) for o, s in zip(self.out_shape, cropped.shape))
            resampled = ndzoom(cropped, factors, order=self.order, mode="nearest",
                               prefilter=self.order > 1)
            # zoom can be off by one voxel from rounding; pad or trim to be exact.
            resampled = _force_shape(resampled, self.out_shape)
        if self.scale != 1.0:
            resampled = resampled * self.scale
        return np.ascontiguousarray(resampled, dtype=np.float32)

    def apply_4d(self, series: np.ndarray) -> np.ndarray:
        """Map a whole dynamic series ``(T, Z, Y, X)`` onto the DLIF grid."""
        if series.ndim != 4:
            raise ValueError(f"expected a 4D series, got shape {series.shape}")
        out = np.empty((series.shape[0], *self.out_shape), dtype=np.float32)
        for t in range(series.shape[0]):
            out[t] = self.apply(series[t])
        return out

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["crop"] = [list(pair) for pair in self.crop]
        payload["out_shape"] = list(self.out_shape)
        payload["flip"] = list(self.flip)
        if self.native_voxel_mm is not None:
            payload["native_voxel_mm"] = list(self.native_voxel_mm)
        out_voxel = self.out_voxel_mm
        payload["out_voxel_mm"] = list(out_voxel) if out_voxel else None
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GridTransform":
        crop = tuple(tuple(int(v) for v in pair) for pair in payload["crop"])  # type: ignore[assignment]
        native = payload.get("native_voxel_mm")
        return cls(
            crop=crop,  # type: ignore[arg-type]
            out_shape=tuple(int(v) for v in payload.get("out_shape", (96, 48, 48))),  # type: ignore[arg-type]
            flip=tuple(bool(v) for v in payload.get("flip", (False, False, False))),  # type: ignore[arg-type]
            order=int(payload.get("order", 1)),
            scale=float(payload.get("scale", 1.0)),
            native_voxel_mm=(tuple(float(v) for v in native) if native else None),  # type: ignore[arg-type]
        )

    @classmethod
    def full_fov(
        cls,
        native_shape_zyx: Sequence[int],
        out_shape: Sequence[int] = (96, 48, 48),
        native_voxel_mm: Sequence[float] | None = None,
    ) -> "GridTransform":
        """The no-crop fallback: resample the whole field of view."""
        z, y, x = (int(v) for v in native_shape_zyx)
        return cls(
            crop=((0, z), (0, y), (0, x)),
            out_shape=tuple(int(v) for v in out_shape),  # type: ignore[arg-type]
            native_voxel_mm=(tuple(float(v) for v in native_voxel_mm) if native_voxel_mm else None),  # type: ignore[arg-type]
        )


def crop_with_zero_padding(
    volume: np.ndarray,
    crop: Sequence[tuple[int, int]],
) -> np.ndarray:
    """Crop a volume, filling with zeros wherever the window leaves it.

    The group's own windows run off the end of the reconstruction on some
    scans: seven of the seventy here need a 96-slice axial window that starts
    1 to 8 slices too late to fit in a 128-slice volume, and their distributed
    inputs carry exact zeros in those slices.  Reproducing that means padding,
    not shrinking the window - a shorter window would be resampled back up to
    96 and the result would no longer be the reconstruction's own voxels.

    Plain slicing cannot express this: ``volume[40:136]`` silently returns 88
    slices rather than 96 zeros-padded ones.
    """
    out_shape = tuple(int(hi) - int(lo) for lo, hi in crop)
    out = np.zeros(out_shape, dtype=volume.dtype)
    source: list[slice] = []
    target: list[slice] = []
    for axis, (lo, hi) in enumerate(crop):
        lo, hi = int(lo), int(hi)
        limit = volume.shape[axis]
        s0, s1 = max(lo, 0), min(hi, limit)
        if s0 >= s1:                      # the window misses the volume entirely
            return out
        source.append(slice(s0, s1))
        target.append(slice(s0 - lo, s1 - lo))
    out[tuple(target)] = volume[tuple(source)]
    return out


def _force_shape(array: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    """Trim or edge-pad ``array`` so its shape is exactly ``shape``."""
    out = array
    for axis, target in enumerate(shape):
        current = out.shape[axis]
        if current > target:
            slicer = [slice(None)] * out.ndim
            start = (current - target) // 2
            slicer[axis] = slice(start, start + target)
            out = out[tuple(slicer)]
        elif current < target:
            deficit = target - current
            before = deficit // 2
            pad = [(0, 0)] * out.ndim
            pad[axis] = (before, deficit - before)
            out = np.pad(out, pad, mode="edge")
    return out


@dataclass
class GridCalibration:
    """Result of fitting the native -> DLIF transform against reference data."""

    transform: GridTransform
    per_axis_correlation: tuple[float, float, float]
    volume_correlation: float
    reference_ids: list[str] = field(default_factory=list)
    n_frames_used: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def is_trustworthy(self) -> bool:
        """Whether the recovered transform reproduces the reference closely enough.

        The threshold is deliberately strict.  A preprocessing mismatch would
        propagate straight into the PVC-versus-baseline comparison, which is the
        very quantity under study.
        """
        return self.volume_correlation >= 0.98 and min(self.per_axis_correlation) >= 0.95

    def to_dict(self) -> dict[str, Any]:
        return {
            "transform": self.transform.to_dict(),
            "per_axis_correlation": list(self.per_axis_correlation),
            "volume_correlation": self.volume_correlation,
            "reference_ids": list(self.reference_ids),
            "n_frames_used": self.n_frames_used,
            "trustworthy": self.is_trustworthy,
            "notes": list(self.notes),
        }

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
        return path


def _fit_axis(
    native_profile: np.ndarray,
    target_profile: np.ndarray,
    out_len: int,
    coarse_step: int = 2,
) -> tuple[int, int, float]:
    """Find the crop window on one axis that best explains the target profile.

    Reduces the 6-parameter crop search to three independent 2-parameter
    searches by matching marginal profiles, which is both far cheaper and much
    better conditioned than a full 3D search.
    """
    n = native_profile.size
    best = (0, n, -1.0)

    min_span = max(out_len // 2, 8)
    for start in range(0, n - min_span + 1, coarse_step):
        for stop in range(start + min_span, n + 1, coarse_step):
            candidate = _resample_axis_profile(native_profile, start, stop, out_len)
            score = _corr(candidate, target_profile)
            if score > best[2]:
                best = (start, stop, score)

    # Local refinement at single-voxel resolution around the coarse optimum.
    c_start, c_stop, _ = best
    for start in range(max(0, c_start - coarse_step), min(n, c_start + coarse_step + 1)):
        for stop in range(max(start + min_span, c_stop - coarse_step), min(n, c_stop + coarse_step) + 1):
            candidate = _resample_axis_profile(native_profile, start, stop, out_len)
            score = _corr(candidate, target_profile)
            if score > best[2]:
                best = (start, stop, score)
    return best


def calibrate_grid(
    native_series: np.ndarray,
    reference_series: np.ndarray,
    native_voxel_mm: Sequence[float] | None = None,
    frames: Sequence[int] | None = None,
    scan_id: str = "",
) -> GridCalibration:
    """Recover the native -> DLIF transform for one scan.

    Parameters
    ----------
    native_series:
        Native dynamic series, ``(T, Z, Y, X)``.
    reference_series:
        The distributed DLIF input for the same scan, ``(T, 96, 48, 48)``.
    frames:
        Which frames to fit on.  Defaults to a spread of mid- and late-frames,
        which carry structure; the first frames are nearly empty and would fit
        noise.
    """
    if native_series.ndim != 4 or reference_series.ndim != 4:
        raise ValueError("both series must be 4D (T, Z, Y, X)")
    if native_series.shape[0] != reference_series.shape[0]:
        LOGGER.warning(
            "%s: %d native frames vs %d reference frames; using the overlap",
            scan_id, native_series.shape[0], reference_series.shape[0],
        )
    n_t = min(native_series.shape[0], reference_series.shape[0])

    if frames is None:
        # Skip the first few frames (essentially empty) and take a spread.
        candidates = [t for t in range(n_t) if t >= min(4, n_t - 1)]
        frames = candidates[:: max(1, len(candidates) // 6)][:6] or [n_t - 1]
    frames = [int(t) for t in frames if 0 <= int(t) < n_t]

    native_mean = native_series[frames].mean(axis=0).astype(float)
    target_mean = reference_series[frames].mean(axis=0).astype(float)
    out_shape = tuple(int(v) for v in target_mean.shape)

    notes: list[str] = []
    crop: list[tuple[int, int]] = []
    axis_scores: list[float] = []
    for axis in range(3):
        other = tuple(a for a in range(3) if a != axis)
        native_profile = native_mean.sum(axis=other)
        target_profile = target_mean.sum(axis=other)
        start, stop, score = _fit_axis(native_profile, target_profile, out_shape[axis])
        crop.append((start, stop))
        axis_scores.append(score)
        LOGGER.info(
            "%s axis %d: crop [%d, %d) of %d -> %d voxels (profile r = %.4f)",
            scan_id, axis, start, stop, native_profile.size, out_shape[axis], score,
        )

    transform = GridTransform(
        crop=(crop[0], crop[1], crop[2]),
        out_shape=out_shape,
        native_voxel_mm=(tuple(float(v) for v in native_voxel_mm) if native_voxel_mm else None),  # type: ignore[arg-type]
    )

    # Fit the intensity scale on the fitted crop, then score the full volume.
    mapped = transform.apply(native_mean)
    denom = float(np.sum(mapped * mapped))
    scale = float(np.sum(mapped * target_mean) / denom) if denom > 1e-12 else 1.0
    transform.scale = scale
    volume_r = _corr(mapped, target_mean)

    if abs(scale - 1.0) > 0.05:
        notes.append(
            f"intensity scale {scale:.4f} differs from 1: the reference pkl is not on the "
            "same SUV scale as the DICOM export; confirm the SUV normalisation with the group"
        )
    if volume_r < 0.98:
        notes.append(
            f"volume correlation {volume_r:.4f} is below 0.98: a plain crop-and-resample does "
            "not reproduce the distributed inputs. Likely an additional step (rotation, "
            "smoothing, body masking or a different reconstruction) is involved -- confirm the "
            "preprocessing recipe before running the PVC comparison"
        )

    return GridCalibration(
        transform=transform,
        per_axis_correlation=(axis_scores[0], axis_scores[1], axis_scores[2]),
        volume_correlation=volume_r,
        reference_ids=[scan_id] if scan_id else [],
        n_frames_used=len(frames),
        notes=notes,
    )


def shape_key(shape: Sequence[int]) -> str:
    """``(128, 92, 92)`` -> ``"128x92x92"``, the key used in the calibration file."""
    return "x".join(str(int(v)) for v in shape)


def load_grid_transform(path: Path, native_shape: Sequence[int] | None = None) -> GridTransform:
    """Load a transform written by :meth:`GridCalibration.save`.

    The dataset contains more than one reconstruction matrix size, so the
    calibration file holds one transform per native shape.  Pass
    ``native_shape`` to get the right one; a scan whose matrix size was never
    calibrated is an error rather than a silent fallback, because using another
    shape's crop would cut the wrong region out of the field of view.
    """
    with open(Path(path), "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    by_shape = payload.get("by_shape")
    if native_shape is not None and by_shape:
        key = shape_key(native_shape)
        if key not in by_shape:
            raise KeyError(
                f"No grid transform calibrated for matrix size {key}. Calibrated: "
                f"{sorted(by_shape)}. Rerun scripts/01_calibrate_grid.py so it covers this shape."
            )
        return GridTransform.from_dict(by_shape[key]["transform"])

    if "transform" in payload:
        return GridTransform.from_dict(payload["transform"])
    return GridTransform.from_dict(payload)
