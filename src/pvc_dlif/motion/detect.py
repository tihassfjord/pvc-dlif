"""Detecting the inter-frame motion artifact.

Members of the group observed a cyclic inter-frame artifact in a subset of the
DLIF scans.  Which scans are affected is decided in consultation with
C. Salomonsen, who has looked at them; this module exists to *propose*
candidates and to quantify what it sees, so the review starts from a ranked
list and a number rather than from scrolling through frame series.

The measure is the frame-to-frame displacement of the centre of mass of the
thresholded volume, in millimetres.  Intra-frame motion is not addressed: it is
already averaged into each reconstructed frame and cannot be undone by
registration.  It is noted as a contribution to the effective resolution that
the point-source PSF does not capture.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Sequence

import numpy as np

from ..logging_utils import get_logger

LOGGER = get_logger(__name__)

__all__ = ["MotionTrace", "centre_of_mass_trace", "detect_motion", "screen_dataset"]


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
) -> MotionTrace:
    """Measure frame-to-frame motion and flag a scan as a motion candidate.

    ``skip_early_frames`` drops the first frames, which are nearly empty before
    the bolus arrives; their centre of mass is noise and would otherwise produce
    a large spurious displacement.
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

    reasons: list[str] = []
    if np.isfinite(max_disp) and max_disp > com_threshold_mm:
        reasons.append(f"max frame-to-frame displacement {max_disp:.2f} mm > {com_threshold_mm} mm")
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
    )


def screen_dataset(
    scans: Any,
    com_threshold_mm: float = 0.5,
    cyclic_autocorr_threshold: float = 0.4,
):
    """Screen many scans and return a ranked candidate table.

    ``scans`` yields ``(scan_id, series, voxel_mm)``.  The result is sorted by
    the cyclic score, so the review can start with the strongest candidates.
    The output is a proposal: the affected set is confirmed by eye, not by this
    threshold.
    """
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for scan_id, series, voxel_mm in scans:
        trace = detect_motion(
            scan_id, series, voxel_mm, com_threshold_mm, cyclic_autocorr_threshold
        )
        rows.append(
            {
                "scan_id": trace.scan_id,
                "max_displacement_mm": trace.max_displacement_mm,
                "mean_displacement_mm": trace.mean_displacement_mm,
                "cyclic_score": trace.cyclic_score,
                "cyclic_period_frames": trace.cyclic_period_frames,
                "flagged": trace.flagged,
                "reason": trace.reason,
            }
        )
        LOGGER.info(
            "%s: max %.2f mm, cyclic %.2f -> %s",
            trace.scan_id, trace.max_displacement_mm, trace.cyclic_score,
            "candidate" if trace.flagged else "clean",
        )

    return pd.DataFrame(rows).sort_values(["cyclic_score", "max_displacement_mm"], ascending=False)
