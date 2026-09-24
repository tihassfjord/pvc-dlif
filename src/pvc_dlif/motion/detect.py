"""Detecting the inter-frame motion artifact.

Members of the group observed a cyclic inter-frame artifact in a subset of the
DLIF scans.  Which scans are affected is decided in consultation with
C. Salomonsen, who has looked at them; this module exists to *propose*
candidates and to quantify what it sees, so the review starts from a ranked
list and a number rather than from scrolling through frame series.

Two measures, and the difference between them decides what you can claim.

``centre of mass``
    Frame-to-frame displacement of the centre of mass of the thresholded
    volume.  Cheap, but it moves for two quite different reasons: the animal
    shifting, and the tracer redistributing from blood pool to liver, brain and
    bladder over the course of the scan.  On this dataset the second dominates
    completely -- the median frame-to-frame value is above 13 mm, which is a
    bolus transiting, not a mouse.  Treated alone it flags every scan and
    therefore says nothing.

``registration shift``
    Translation of each frame relative to a late reference, estimated by phase
    correlation on smoothed copies.  This is what registration itself measures,
    so it responds to the animal moving and not to activity appearing somewhere
    new.  It is the number to quote, and the one that makes a before/after
    comparison meaningful: a correction that works drives it toward zero while
    leaving the centre-of-mass trace almost untouched.

Neither addresses intra-frame motion: that is already averaged into each
reconstructed frame and cannot be undone by registration.  It is noted as a
contribution to the effective resolution that the point-source PSF does not
capture.

The early frames are a known weak spot for both measures.  Before the tracer
distributes there is little anatomy to register, so the honest answer there is
"uncertain" rather than a small number -- ``skip_early_frames`` keeps those out
of the summary statistics instead of letting them set them.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Sequence

import numpy as np

from ..logging_utils import get_logger

LOGGER = get_logger(__name__)

__all__ = [
    "MotionTrace", "centre_of_mass_trace", "registration_shift_trace",
    "detect_motion", "screen_dataset",
]


@dataclass
class MotionTrace:
    """Per-frame motion summary for one scan."""

    scan_id: str
    displacement_mm: list[float]        # frame-to-frame displacement
    centre_mm: list[list[float]]        # centre of mass per frame, (x, y, z) in mm
    max_displacement_mm: float
    mean_displacement_mm: float
    cyclic_score: float                 # peak autocorrelation of the displacement series
    cyclic_period_frames: int | None
    flagged: bool
    reason: str = ""

    # Registration-based displacement against a late reference frame.  This is
    # the measure that means "the animal moved"; everything above can be moved
    # by the tracer alone.
    shift_mm: list[float] = field(default_factory=list)          # magnitude per frame
    shift_vector_mm: list[list[float]] = field(default_factory=list)  # (x, y, z) per frame
    max_shift_mm: float = float("nan")
    mean_shift_mm: float = float("nan")
    p95_shift_mm: float = float("nan")
    reference_frame: int = -1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def centre_of_mass_trace(
    series: np.ndarray,
    voxel_mm: Sequence[float],
    threshold_percentile: float = 90.0,
) -> np.ndarray:
    """Centre of mass of the high-activity volume, per frame, in millimetres.

    Thresholding at a high percentile keeps the measure on the animal rather
    than on background noise, which otherwise dominates the early frames where
    the animal is barely visible.
    """
    if series.ndim != 4:
        raise ValueError(f"expected (T, Z, Y, X); got {series.shape}")

    n_t = series.shape[0]
    centres = np.full((n_t, 3), np.nan)
    vx, vy, vz = (float(v) for v in voxel_mm)

    grids = np.meshgrid(
        np.arange(series.shape[1], dtype=np.float64) * vz,
        np.arange(series.shape[2], dtype=np.float64) * vy,
        np.arange(series.shape[3], dtype=np.float64) * vx,
        indexing="ij",
    )

    for t in range(n_t):
        volume = series[t]
        positive = volume[volume > 0]
        if positive.size < 10:
            continue
        threshold = np.percentile(positive, threshold_percentile)
        weights = np.where(volume >= threshold, volume, 0.0)
        total = weights.sum()
        if total <= 0:
            continue
        z = float((grids[0] * weights).sum() / total)
        y = float((grids[1] * weights).sum() / total)
        x = float((grids[2] * weights).sum() / total)
        centres[t] = (x, y, z)

    return centres


def registration_shift_trace(
    series: np.ndarray,
    voxel_mm: Sequence[float],
    reference: int | str = "late_mean",
    smooth_sigma_voxels: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Translation of every frame relative to a reference, in millimetres.

    Returns ``(vectors (T, 3) as (x, y, z), magnitudes (T,), reference index)``.

    This estimates the same shifts :func:`~pvc_dlif.motion.correct.rigid_translation_correct`
    would apply, but does not apply them: it is a measurement, not a
    correction, so it can be run before and after any correction method --
    FALCON included -- and compared.

    Registration is done on Gaussian-smoothed copies for the same reason the
    correction is: the early frames are far too noisy to register raw.  The
    reference is a late frame by default, where counts are highest and the
    anatomy is stable.
    """
    from scipy.ndimage import gaussian_filter

    from .correct import _phase_shift, reference_frame_index

    if series.ndim != 4:
        raise ValueError(f"expected (T, Z, Y, X); got {series.shape}")

    ref_index = reference if isinstance(reference, int) else reference_frame_index(series, reference)
    voxel_zyx = np.array([float(voxel_mm[2]), float(voxel_mm[1]), float(voxel_mm[0])])

    fixed = gaussian_filter(series[ref_index].astype(np.float32), smooth_sigma_voxels)
    vectors = np.zeros((series.shape[0], 3), dtype=float)

    for t in range(series.shape[0]):
        if t == ref_index:
            continue
        frame = series[t]
        # A frame with essentially no signal cannot be registered; leave it as
        # not-a-number rather than reporting a confident zero.
        if not np.any(frame > 0):
            vectors[t] = np.nan
            continue
        moving = gaussian_filter(frame.astype(np.float32), smooth_sigma_voxels)
        shift_mm = _phase_shift(moving, fixed) * voxel_zyx
        vectors[t] = shift_mm[::-1]          # store as (x, y, z)

    magnitudes = np.linalg.norm(vectors, axis=1)
    return vectors, magnitudes, int(ref_index)


def _cyclic_score(signal: np.ndarray) -> tuple[float, int | None]:
    """Peak of the normalised autocorrelation at lag >= 2.

    A high value at a consistent lag is what distinguishes a cyclic artifact
    from a single drift or one sudden movement.
    """
    values = np.asarray(signal, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 6:
        return 0.0, None

    centred = values - values.mean()
    denom = float(np.sum(centred ** 2))
    if denom < 1e-12:
        return 0.0, None

    correlation = np.correlate(centred, centred, mode="full")[centred.size - 1:] / denom
    search = correlation[2: max(3, centred.size // 2)]
    if search.size == 0:
        return 0.0, None
    best = int(np.argmax(search))
    return float(search[best]), int(best + 2)


def detect_motion(
    scan_id: str,
    series: np.ndarray,
    voxel_mm: Sequence[float],
    com_threshold_mm: float = 0.5,
    cyclic_autocorr_threshold: float = 0.4,
    skip_early_frames: int = 3,
    shift_threshold_mm: float = 0.5,
    reference: int | str = "late_mean",
    with_registration: bool = True,
) -> MotionTrace:
    """Measure frame-to-frame motion and flag a scan as a motion candidate.

    ``skip_early_frames`` drops the first frames, which are nearly empty before
    the bolus arrives; their centre of mass is noise and would otherwise produce
    a large spurious displacement.

    ``with_registration`` adds the phase-correlation shift against a late
    reference.  It costs one registration per frame, so it is far slower than
    the centre-of-mass trace -- but it is the only one of the two that
    distinguishes the animal moving from the tracer redistributing, and the
    flag is raised on it whenever it is available.  Set it to False when you
    only want the cheap screen.
    """
    centres = centre_of_mass_trace(series, voxel_mm)
    displacement = np.full(centres.shape[0], np.nan)
    for t in range(1, centres.shape[0]):
        if np.all(np.isfinite(centres[t])) and np.all(np.isfinite(centres[t - 1])):
            displacement[t] = float(np.linalg.norm(centres[t] - centres[t - 1]))

    usable = displacement[skip_early_frames:]
    usable = usable[np.isfinite(usable)]
    max_disp = float(np.max(usable)) if usable.size else float("nan")
    mean_disp = float(np.mean(usable)) if usable.size else float("nan")
    score, period = _cyclic_score(usable)

    vectors = np.empty((0, 3))
    shifts = np.empty(0)
    ref_index = -1
    max_shift = mean_shift = p95_shift = float("nan")
    if with_registration:
        vectors, shifts, ref_index = registration_shift_trace(series, voxel_mm, reference)
        usable_shift = shifts[skip_early_frames:]
        usable_shift = usable_shift[np.isfinite(usable_shift)]
        if usable_shift.size:
            max_shift = float(np.max(usable_shift))
            mean_shift = float(np.mean(usable_shift))
            # The 95th percentile is the number to report: one badly registered
            # early frame should not become the scan's headline figure.
            p95_shift = float(np.percentile(usable_shift, 95))

    reasons: list[str] = []
    # The registration shift decides the flag when it exists.  Centre-of-mass
    # displacement is kept in the table because it is informative about the
    # tracer, but it moves by more than a centimetre on a perfectly still
    # animal, so it cannot carry the decision.
    if with_registration and np.isfinite(p95_shift):
        if p95_shift > shift_threshold_mm:
            reasons.append(
                f"registration shift p95 {p95_shift:.2f} mm > {shift_threshold_mm} mm "
                f"(max {max_shift:.2f} mm, reference frame {ref_index})"
            )
    elif np.isfinite(max_disp) and max_disp > com_threshold_mm:
        reasons.append(
            f"centre-of-mass displacement {max_disp:.2f} mm > {com_threshold_mm} mm "
            "(no registration estimate; includes tracer redistribution)"
        )
    if score > cyclic_autocorr_threshold:
        reasons.append(f"cyclic pattern (autocorrelation {score:.2f} at lag {period})")

    return MotionTrace(
        scan_id=scan_id,
        displacement_mm=[float(v) for v in displacement],
        centre_mm=[[float(c) for c in row] for row in centres],
        max_displacement_mm=max_disp,
        mean_displacement_mm=mean_disp,
        cyclic_score=score,
        cyclic_period_frames=period,
        flagged=bool(reasons),
        reason="; ".join(reasons),
        shift_mm=[float(v) for v in shifts],
        shift_vector_mm=[[float(c) for c in row] for row in vectors],
        max_shift_mm=max_shift,
        mean_shift_mm=mean_shift,
        p95_shift_mm=p95_shift,
        reference_frame=ref_index,
    )


def screen_dataset(
    scans: Any,
    com_threshold_mm: float = 0.5,
    cyclic_autocorr_threshold: float = 0.4,
    shift_threshold_mm: float = 0.5,
    reference: int | str = "late_mean",
    with_registration: bool = True,
):
    """Screen many scans and return a ranked candidate table.

    ``scans`` yields ``(scan_id, series, voxel_mm)``.  The result is sorted by
    registration shift, largest first, so the review starts with the scans that
    moved most.  The output is a proposal: the affected set is confirmed by
    eye, not by this threshold.
    """
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for scan_id, series, voxel_mm in scans:
        trace = detect_motion(
            scan_id, series, voxel_mm, com_threshold_mm, cyclic_autocorr_threshold,
            shift_threshold_mm=shift_threshold_mm, reference=reference,
            with_registration=with_registration,
        )
        rows.append(
            {
                "scan_id": trace.scan_id,
                "p95_shift_mm": trace.p95_shift_mm,
                "max_shift_mm": trace.max_shift_mm,
                "mean_shift_mm": trace.mean_shift_mm,
                "reference_frame": trace.reference_frame,
                "max_displacement_mm": trace.max_displacement_mm,
                "mean_displacement_mm": trace.mean_displacement_mm,
                "cyclic_score": trace.cyclic_score,
                "cyclic_period_frames": trace.cyclic_period_frames,
                "flagged": trace.flagged,
                "reason": trace.reason,
            }
        )
        LOGGER.info(
            "%s: shift p95 %.2f mm (max %.2f), centre-of-mass %.2f mm, cyclic %.2f -> %s",
            trace.scan_id, trace.p95_shift_mm, trace.max_shift_mm,
            trace.max_displacement_mm, trace.cyclic_score,
            "candidate" if trace.flagged else "clean",
        )

    frame = pd.DataFrame(rows)
    sort_by = "p95_shift_mm" if with_registration else "cyclic_score"
    return frame.sort_values([sort_by, "max_displacement_mm"], ascending=False)
