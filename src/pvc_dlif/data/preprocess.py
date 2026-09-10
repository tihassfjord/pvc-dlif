"""Preprocessing from the reconstructed grid to the DLIF input grid.

What the group's preprocessing actually is
------------------------------------------
It is a **pure crop at native resolution** -- no resampling, no smoothing, no
rotation.  The arithmetic gives it away: the DLIF grid is 96 x 48 x 48 and the
reconstruction is 0.5 x 0.5 x 0.59675 mm, so the input covers
96 x 0.59675 = 57.3 mm axially and 48 x 0.5 = 24 mm in plane -- exactly a
96 x 48 x 48 voxel window cut out of the 128 x 92 x 92 volume.  Fitting that
window against a distributed ``IMG_*.pkl`` reproduces it at r = 1.0000, with a
single global intensity factor near 0.99.

That matters for the thesis well beyond tidiness:

* the input voxels *are* reconstruction voxels, so the measured point-source
  PSF applies to the network input directly, with no interpolation to account
  for;
* PVC applied at native resolution therefore maps into the network input
  one-to-one -- nothing is blurred back out by a resample on the way in;
* the uncorrected condition can be made identical to the distributed inputs,
  so the pretrained model is used exactly as it was trained.

How the window is chosen
------------------------
The window position is per-scan, because mice are not positioned identically.
Two ways to get it, in order of preference:

1. :func:`fit_offset_to_reference` -- recover the group's own window by
   matching against their distributed input for that scan.  Exact, and it makes
   the uncorrected tree reproduce theirs.
2. :func:`body_centred_offset` -- centre the window on the animal.  Used for any
   scan with no distributed input to match against.

Either way the window is fixed **once, from the uncorrected series**, stored,
and reused verbatim for every condition.  If it were recomputed on corrected
data, deconvolution would shift the centre of mass slightly and the conditions
would differ by a crop as well as by PVC -- the one confound that would
invalidate the whole comparison.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from ..logging_utils import get_logger
from .grid import GridTransform, crop_with_zero_padding

LOGGER = get_logger(__name__)

__all__ = [
    "BodyMeasurement", "PreprocessingPlan", "OffsetFit",
    "measure_body", "body_centred_offset", "fit_offset_to_reference",
    "derive_windows", "load_plan",
]

#: Frames used when matching a window against a reference: mid and late frames,
#: where the animal is clearly visible.  The first frames are nearly empty.
_FIT_FRAMES = (8, 14, 20, 30, 41)


@dataclass
class BodyMeasurement:
    """Where the animal is on one scan."""

    scan_id: str
    centre_vox: tuple[float, float, float]      # (z, y, x)
    extent_mm: tuple[float, float, float]       # (z, y, x) bounding-box size
    bbox_vox: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    voxel_mm: tuple[float, float, float]        # (x, y, z)
    shape_zyx: tuple[int, int, int]
    n_voxels: int

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("centre_vox", "extent_mm", "voxel_mm", "shape_zyx"):
            payload[key] = list(payload[key])
        payload["bbox_vox"] = [list(p) for p in self.bbox_vox]
        return payload


def measure_body(
    series: np.ndarray,
    voxel_mm: Sequence[float],
    scan_id: str = "",
    late_fraction: float = 0.4,
    threshold_fraction: float = 0.15,
) -> BodyMeasurement:
    """Locate the animal in a dynamic series.

    Measured on the mean of the last ``late_fraction`` of frames, where the
    distribution has settled and the counting statistics are best; the early
    bolus frames are nearly empty and their centre of mass is noise.  The
    threshold is a fraction of the 99.5th percentile rather than of the maximum,
    so a single hot voxel cannot set the scale.
    """
    if series.ndim != 4:
        raise ValueError(f"expected (T, Z, Y, X); got {series.shape}")

    n_t = series.shape[0]
    volume = series[max(int(n_t * (1.0 - late_fraction)), 0):].mean(axis=0).astype(np.float64)

    robust_max = float(np.percentile(volume, 99.5))
    if robust_max <= 0:
        raise ValueError(f"{scan_id}: the late-frame average is empty")
    mask = volume > (threshold_fraction * robust_max)

    try:
        from scipy.ndimage import binary_closing, binary_fill_holes, label

        mask = binary_fill_holes(binary_closing(mask, iterations=2))
        labelled, n_components = label(mask)
        if n_components > 1:
            sizes = np.bincount(labelled.ravel())
            sizes[0] = 0
            mask = labelled == int(sizes.argmax())
    except ImportError:  # pragma: no cover
        pass

    if not mask.any():
        raise ValueError(f"{scan_id}: body mask empty at {threshold_fraction:.0%} of the robust max")

    coords = np.array(np.nonzero(mask))
    bbox = tuple((int(coords[a].min()), int(coords[a].max()) + 1) for a in range(3))
    weights = np.where(mask, volume, 0.0)
    total = float(weights.sum())
    centre = tuple(
        float((np.arange(volume.shape[a]) *
               weights.sum(axis=tuple(o for o in range(3) if o != a))).sum() / total)
        for a in range(3)
    )

    vx, vy, vz = (float(v) for v in voxel_mm)
    voxel_zyx = (vz, vy, vx)
    return BodyMeasurement(
        scan_id=scan_id,
        centre_vox=centre,                                     # type: ignore[arg-type]
        extent_mm=tuple(float((hi - lo) * voxel_zyx[a])
                        for a, (lo, hi) in enumerate(bbox)),   # type: ignore[arg-type]
        bbox_vox=bbox,                                         # type: ignore[arg-type]
        voxel_mm=(vx, vy, vz),
        shape_zyx=tuple(int(s) for s in volume.shape),         # type: ignore[arg-type]
        n_voxels=int(mask.sum()),
    )


def body_centred_offset(
    measurement: BodyMeasurement,
    crop_shape: Sequence[int],
) -> tuple[tuple[int, int, int], list[str]]:
    """Window offset that centres a fixed-size crop on the animal."""
    warnings: list[str] = []
    offset: list[int] = []
    for axis in range(3):
        span = int(crop_shape[axis])
        limit = measurement.shape_zyx[axis]
        if span > limit:
            raise ValueError(
                f"{measurement.scan_id}: a {span}-voxel crop does not fit in {limit} voxels "
                f"on axis {'zyx'[axis]}"
            )
        start = int(round(measurement.centre_vox[axis] - span / 2.0))
        clamped = min(max(start, 0), limit - span)
        if clamped != start and span < limit:
            voxel_zyx = (measurement.voxel_mm[2], measurement.voxel_mm[1], measurement.voxel_mm[0])
            shift_mm = abs(clamped - start) * voxel_zyx[axis]
            if shift_mm > 1.0:
                warnings.append(
                    f"{measurement.scan_id}: the window on axis {'zyx'[axis]} was shifted "
                    f"{shift_mm:.1f} mm to stay inside the reconstructed volume"
                )
        offset.append(clamped)
    return (offset[0], offset[1], offset[2]), warnings


@dataclass
class OffsetFit:
    """Result of matching a crop window against a distributed reference input."""

    offset: tuple[int, int, int]
    correlation: float
    scale: float
    exact: bool

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["offset"] = list(self.offset)
        return payload


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel()
    b = b.ravel()
    if a.std() < 1e-12 or b.std() < 1e-12:
        return -1.0
    return float(np.corrcoef(a, b)[0, 1])


def fit_offset_to_reference(
    native_series: np.ndarray,
    reference_series: np.ndarray,
    start: Sequence[int] | None = None,
    frames: Sequence[int] = _FIT_FRAMES,
    scan_id: str = "",
    exact_threshold: float = 0.999,
) -> OffsetFit:
    """Recover the crop window the group used, by matching their own input.

    Because the transform is a pure crop, the search is over three integers.
    It runs in two passes: an exhaustive sweep at two-voxel steps scored on
    every second voxel (eight times cheaper per candidate), then an exhaustive
    refinement at single-voxel steps around the winner, scored on every voxel.

    The coarse pass has to be exhaustive.  The objective is near zero except
    within a voxel or two of the true offset -- there is no gradient to follow
    towards it -- so a descent from a starting guess stalls wherever it begins.
    ``start`` only breaks ties.
    """
    crop_shape = tuple(int(s) for s in reference_series.shape[1:])
    usable = [f for f in frames if f < min(native_series.shape[0], reference_series.shape[0])]
    if not usable:
        usable = [min(native_series.shape[0], reference_series.shape[0]) - 1]

    source = native_series[usable].mean(axis=0).astype(np.float64)
    target = reference_series[usable].mean(axis=0).astype(np.float64)
    limits = [source.shape[a] - crop_shape[a] for a in range(3)]
    if any(limit < 0 for limit in limits):
        raise ValueError(
            f"{scan_id}: the reference input {crop_shape} is larger than the reconstruction "
            f"{source.shape} on at least one axis"
        )

    # The window is allowed to hang off the volume, and the part outside is
    # zero-filled.  It has to be: the group's own windows do exactly that on
    # seven of the seventy scans here - a 96-slice axial window starting 1 to 8
    # slices too late to fit in 128 - and their distributed inputs carry exact
    # zeros in those slices, so a search restricted to windows that fit inside
    # the volume pins against the wall and never finds the right one.  Padding
    # is capped at a quarter of the window so a mostly-empty match can never win.
    pad = [int(s) // 4 for s in crop_shape]

    dz, dy, dx = crop_shape

    # The coarse pass scores on smoothed, subsampled copies.  Smoothing is what
    # makes it work: on the raw volumes a one-voxel error already decorrelates
    # the subsample, so the coarse grid would see noise and land nowhere near
    # the optimum.  Blurred, a near-miss still scores high, which is exactly the
    # gradient the coarse pass needs.  The fine pass then works on the raw data,
    # so the smoothing never affects the recovered offset.
    try:
        from scipy.ndimage import gaussian_filter

        coarse_source = gaussian_filter(source, 1.5)
        coarse_target = gaussian_filter(target, 1.5)
    except ImportError:  # pragma: no cover
        coarse_source, coarse_target = source, target

    def window_of(volume: np.ndarray, offset: Sequence[int]) -> np.ndarray:
        z, y, x = (int(v) for v in offset)
        # Fast path for a window that fits: a view, no allocation.  Almost every
        # candidate takes it, and the search evaluates tens of thousands.
        if (z >= 0 and y >= 0 and x >= 0
                and z + dz <= volume.shape[0]
                and y + dy <= volume.shape[1]
                and x + dx <= volume.shape[2]):
            return volume[z:z + dz, y:y + dy, x:x + dx]
        return crop_with_zero_padding(volume, ((z, z + dz), (y, y + dy), (x, x + dx)))

    def score(offset: Sequence[int], coarse: bool = False) -> float:
        if coarse:
            window = window_of(coarse_source, offset)
            return _corr(window[::2, ::2, ::2], coarse_target[::2, ::2, ::2])
        return _corr(window_of(source, offset), target)

    # Coarse: exhaustive at two-voxel steps, over the windows that fit.
    coarse_step = 2 if max(limits) > 6 else 1
    best_offset = tuple(
        min(max(int(start[a]) if start is not None else limits[a] // 2, 0), limits[a])
        for a in range(3)
    )
    best = score(best_offset, coarse=True)
    for z in range(0, limits[0] + 1, coarse_step):
        for y in range(0, limits[1] + 1, coarse_step):
            for x in range(0, limits[2] + 1, coarse_step):
                candidate = score((z, y, x), coarse=True)
                if candidate > best:
                    best, best_offset = candidate, (z, y, x)

    # A winner against a wall of that box is a warning, not an answer: the box
    # only holds windows that fit inside the reconstruction, and the group's do
    # not always.  Widen the offending axis and sweep the band that was out of
    # reach -- exhaustively, and over all three axes at once, because a greedy
    # step outward on one axis stalls before the other two are refitted around
    # its new position.
    lo_bound, hi_bound = [0, 0, 0], list(limits)
    for axis in range(3):
        for direction in (-1, 1):
            if best_offset[axis] != (0 if direction < 0 else limits[axis]):
                continue
            reach_out = pad[axis]
            if direction < 0:
                lo_bound[axis] = -reach_out
                band = range(-reach_out, 0, coarse_step)
            else:
                hi_bound[axis] = limits[axis] + reach_out
                band = range(limits[axis] + 1, limits[axis] + reach_out + 1, coarse_step)
            others = [a for a in range(3) if a != axis]
            for value in band:
                for u in range(lo_bound[others[0]], hi_bound[others[0]] + 1, coarse_step):
                    for v in range(lo_bound[others[1]], hi_bound[others[1]] + 1, coarse_step):
                        offset = [0, 0, 0]
                        offset[axis], offset[others[0]], offset[others[1]] = value, u, v
                        candidate = score(offset, coarse=True)
                        if candidate > best:
                            best, best_offset = candidate, tuple(offset)

    # Fine: exhaustive within one coarse step of the winner, every voxel.
    reach = coarse_step + 1
    best = score(best_offset)
    for z in range(max(best_offset[0] - reach, lo_bound[0]), min(best_offset[0] + reach, hi_bound[0]) + 1):
        for y in range(max(best_offset[1] - reach, lo_bound[1]), min(best_offset[1] + reach, hi_bound[1]) + 1):
            for x in range(max(best_offset[2] - reach, lo_bound[2]), min(best_offset[2] + reach, hi_bound[2]) + 1):
                candidate = score((z, y, x))
                if candidate > best:
                    best, best_offset = candidate, (z, y, x)

    z, y, x = best_offset
    window = crop_with_zero_padding(
        source, ((z, z + crop_shape[0]), (y, y + crop_shape[1]), (x, x + crop_shape[2]))
    )
    denominator = float(np.dot(window.ravel(), window.ravel()))
    scale = float(np.dot(window.ravel(), target.ravel()) / denominator) if denominator > 1e-12 else 1.0

    fit = OffsetFit(
        offset=(z, y, x), correlation=best, scale=scale, exact=best >= exact_threshold
    )
    LOGGER.info(
        "%s: crop offset (%d, %d, %d), r = %.5f, intensity factor %.4f%s",
        scan_id, z, y, x, best, scale, "" if fit.exact else "  [not an exact match]",
    )
    return fit


@dataclass
class PreprocessingPlan:
    """The preprocessing rule plus the window it gives each scan."""

    crop_shape: tuple[int, int, int]
    voxel_mm: tuple[float, float, float]
    windows: dict[str, dict[str, Any]] = field(default_factory=dict)
    fits: dict[str, dict[str, Any]] = field(default_factory=dict)
    measurements: dict[str, dict[str, Any]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    settings: dict[str, Any] = field(default_factory=dict)

    def transform(self, scan_id: str, out_shape: Sequence[int]) -> GridTransform:
        """The transform for one scan onto one output grid.

        ``out_shape`` equal to the stored crop is the plain window.  A smaller
        output is taken as a *centred sub-crop* of it, so it stays a pure crop
        at native resolution -- that is how the 64 x 48 x 48 grid the pretrained
        model needs is produced from the 96 x 48 x 48 window.
        """
        if scan_id not in self.windows:
            raise KeyError(
                f"No preprocessing window for {scan_id}. Rerun scripts/01_prepare_data.py "
                "so it covers this scan."
            )
        stored = GridTransform.from_dict(self.windows[scan_id])
        out_shape = tuple(int(v) for v in out_shape)

        crop: list[tuple[int, int]] = []
        for axis in range(3):
            lo, hi = stored.crop[axis]
            span = hi - lo
            want = out_shape[axis]
            if want > span:
                raise ValueError(
                    f"{scan_id}: cannot produce {want} voxels on axis {'zyx'[axis]} from a "
                    f"{span}-voxel window without resampling; widen the crop in stage 01."
                )
            inset = (span - want) // 2
            crop.append((lo + inset, lo + inset + want))

        return GridTransform(
            crop=(crop[0], crop[1], crop[2]),   # type: ignore[arg-type]
            out_shape=out_shape,                # type: ignore[arg-type]
            flip=stored.flip,
            order=stored.order,
            scale=stored.scale,
            native_voxel_mm=stored.native_voxel_mm,
        )

    def out_voxel_mm(self, out_shape: Sequence[int] | None = None) -> tuple[float, float, float]:
        """Voxel size of the DLIF grid: the reconstruction's own, unchanged."""
        return self.voxel_mm

    @property
    def n_exact(self) -> int:
        return sum(1 for f in self.fits.values() if f.get("exact"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "crop_shape": list(self.crop_shape),
            "voxel_mm": list(self.voxel_mm),
            "fov_mm": [
                round(self.crop_shape[0] * self.voxel_mm[2], 4),
                round(self.crop_shape[1] * self.voxel_mm[1], 4),
                round(self.crop_shape[2] * self.voxel_mm[0], 4),
            ],
            "n_scans": len(self.windows),
            "n_matched_to_distributed": len(self.fits),
            "n_exact": self.n_exact,
            "settings": self.settings,
            "warnings": self.warnings,
            "windows": self.windows,
            "fits": self.fits,
            "measurements": self.measurements,
        }

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


def load_plan(path: Path) -> PreprocessingPlan:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return PreprocessingPlan(
        crop_shape=tuple(int(v) for v in payload["crop_shape"]),   # type: ignore[arg-type]
        voxel_mm=tuple(float(v) for v in payload["voxel_mm"]),     # type: ignore[arg-type]
        windows=payload.get("windows", {}),
        fits=payload.get("fits", {}),
        measurements=payload.get("measurements", {}),
        warnings=payload.get("warnings", []),
        settings=payload.get("settings", {}),
    )


def derive_windows(
    scans: Iterable[tuple[str, np.ndarray, Sequence[float], np.ndarray | None]],
    crop_shape: Sequence[int],
    apply_intensity_scale: bool = True,
    late_fraction: float = 0.4,
    threshold_fraction: float = 0.15,
) -> PreprocessingPlan:
    """Fix one crop window per scan.

    ``scans`` yields ``(scan_id, uncorrected_series, voxel_mm, reference_or_None)``.
    The series must be the *uncorrected* one: the window is fixed here and
    reused for every condition, so that PVC cannot change the crop.

    Where a distributed reference input is supplied, the window is recovered
    from it, which reproduces the group's own input exactly.  Otherwise it is
    centred on the animal.
    """
    crop_shape = tuple(int(v) for v in crop_shape)
    windows: dict[str, dict[str, Any]] = {}
    fits: dict[str, dict[str, Any]] = {}
    measurements: dict[str, dict[str, Any]] = {}
    all_warnings: list[str] = []
    voxel_seen: tuple[float, float, float] | None = None

    for scan_id, series, voxel_mm, reference in scans:
        voxel_mm = tuple(float(v) for v in voxel_mm)  # type: ignore[assignment]
        if voxel_seen is None:
            voxel_seen = voxel_mm  # type: ignore[assignment]
        elif voxel_mm != voxel_seen:
            all_warnings.append(
                f"{scan_id}: voxel size {voxel_mm} differs from {voxel_seen}; the crop covers a "
                "different physical field of view on this scan"
            )

        try:
            measurement = measure_body(series, voxel_mm, scan_id, late_fraction, threshold_fraction)
            measurements[scan_id] = measurement.to_dict()
            start, warnings = body_centred_offset(measurement, crop_shape)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("%s: could not locate the animal (%s); starting from the volume centre",
                           scan_id, exc)
            start = tuple(  # type: ignore[assignment]
                max((series.shape[a + 1] - crop_shape[a]) // 2, 0) for a in range(3)
            )
            warnings = []

        scale = 1.0
        if reference is not None:
            try:
                fit = fit_offset_to_reference(series, reference, start, scan_id=scan_id)
                fits[scan_id] = fit.to_dict()
                start = fit.offset
                if apply_intensity_scale:
                    scale = fit.scale
                if not fit.exact:
                    warnings.append(
                        f"{scan_id}: the window matches the distributed input at r = "
                        f"{fit.correlation:.4f} rather than exactly; that scan's distributed "
                        "input may come from a different reconstruction"
                    )
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("%s: could not match the distributed input (%s); using the "
                               "body-centred window", scan_id, exc)

        transform = GridTransform(
            crop=tuple((int(start[a]), int(start[a]) + crop_shape[a]) for a in range(3)),  # type: ignore[arg-type]
            out_shape=crop_shape,       # type: ignore[arg-type]
            scale=scale,
            native_voxel_mm=voxel_mm,   # type: ignore[arg-type]
        )
        windows[scan_id] = transform.to_dict()
        all_warnings.extend(warnings)

    if not windows:
        raise RuntimeError("no scan could be windowed; cannot derive a preprocessing plan")

    for message in all_warnings:
        LOGGER.warning(message)

    plan = PreprocessingPlan(
        crop_shape=crop_shape,                              # type: ignore[arg-type]
        voxel_mm=voxel_seen or (1.0, 1.0, 1.0),             # type: ignore[arg-type]
        windows=windows,
        fits=fits,
        measurements=measurements,
        warnings=all_warnings,
        settings={
            "rule": "pure crop at native resolution, no resampling",
            "window_source": "matched to the distributed input where available, "
                             "otherwise centred on the animal",
            "apply_intensity_scale": apply_intensity_scale,
            "late_fraction": late_fraction,
            "threshold_fraction": threshold_fraction,
        },
    )
    LOGGER.info(
        "Preprocessing plan: %d scans, %s voxel crop at %.3f x %.3f x %.3f mm "
        "(%.1f x %.1f x %.1f mm field of view); %d/%d matched the distributed inputs exactly",
        len(windows), "x".join(str(v) for v in crop_shape), *plan.voxel_mm,
        crop_shape[0] * plan.voxel_mm[2], crop_shape[1] * plan.voxel_mm[1],
        crop_shape[2] * plan.voxel_mm[0], plan.n_exact, len(fits),
    )
    return plan
