"""Reading the dynamic mouse PET exports and writing them as 4D NIfTI.

The scans are PMOD-exported multi-frame DICOM files (SOP class
``1.2.840.10008.5.1.4.1.1.20``, Positron Emission Tomography Image Storage),
one file per animal, named ``dPET_dcm_<ID>``.  A single file holds
``NumberOfSlices * NumberOfTimeSlots`` frames stored with the slice index
running fastest, in SUV units.

Three things are needed downstream and all three come from here:

* the 4D activity array in SUV,
* the voxel geometry in millimetres, because the PSF used by PVC is specified
  in millimetres and must be converted to voxels on the *reconstructed* grid,
* the frame start times and durations, because the recovery-noise trade-off is
  expected to depend on frame count statistics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from ..logging_utils import get_logger

LOGGER = get_logger(__name__)

# PMOD private tags observed in this dataset (group 0x0055, creator "PMOD_1").
TAG_FRAME_START_S = (0x0055, 0x1001)   # frame start times, seconds
TAG_FRAME_DURATION_MS = (0x0055, 0x1004)  # frame durations, milliseconds
TAG_RESCALE_VECTOR = (0x0055, 0x1005)  # per-frame rescale slope

_ID_PATTERN = re.compile(r"dPET_dcm_(?P<sid>.+?)$", re.IGNORECASE)

__all__ = [
    "ScanGeometry",
    "DynamicScan",
    "scan_id_from_path",
    "iter_dicom_scans",
    "read_dynamic_dicom",
    "write_nifti_4d",
    "read_nifti_4d",
]


@dataclass(frozen=True)
class ScanGeometry:
    """Voxel geometry of a reconstructed dynamic series.

    ``voxel_mm`` is ordered ``(x, y, z)`` to match the NIfTI axis convention
    used everywhere downstream; ``shape_zyx`` is the array shape as stored in
    DICOM (slice, row, column).
    """

    voxel_mm: tuple[float, float, float]
    shape_zyx: tuple[int, int, int]
    units: str = "SUV"

    @property
    def voxel_volume_mm3(self) -> float:
        return float(np.prod(self.voxel_mm))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DynamicScan:
    """A dynamic PET series with its timing and geometry.

    Attributes
    ----------
    scan_id:
        Animal identifier, e.g. ``"AA1"``.
    data:
        Activity array of shape ``(T, Z, Y, X)`` in SUV.
    frame_start_s, frame_duration_s:
        Frame timing, in seconds from injection.
    geometry:
        Voxel geometry of the reconstructed grid.
    """

    scan_id: str
    data: np.ndarray
    frame_start_s: np.ndarray
    frame_duration_s: np.ndarray
    geometry: ScanGeometry
    source: Path | None = None
    meta: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.data.ndim != 4:
            raise ValueError(f"{self.scan_id}: expected a 4D array, got shape {self.data.shape}")
        n_t = self.data.shape[0]
        if len(self.frame_start_s) != n_t or len(self.frame_duration_s) != n_t:
            raise ValueError(
                f"{self.scan_id}: {n_t} frames but {len(self.frame_start_s)} start times "
                f"and {len(self.frame_duration_s)} durations"
            )

    @property
    def n_frames(self) -> int:
        return int(self.data.shape[0])

    @property
    def frame_mid_min(self) -> np.ndarray:
        """Frame mid-points in minutes, the time axis used for the input functions."""
        return (self.frame_start_s + self.frame_duration_s / 2.0) / 60.0

    @property
    def frame_start_min(self) -> np.ndarray:
        return self.frame_start_s / 60.0

    def frame_counts_proxy(self) -> np.ndarray:
        """A per-frame proxy for count statistics.

        True prompt counts are not in the SUV export, so total activity times
        frame duration is used as a monotone stand-in.  It is only ever used to
        *order* frames by count level, never as an absolute count.
        """
        return self.data.reshape(self.n_frames, -1).sum(axis=1) * self.frame_duration_s


def scan_id_from_path(path: Path) -> str:
    """Extract the animal ID from a ``dPET_dcm_<ID>`` file or directory name."""
    stem = path.name
    for suffix in (".dcm", ".DCM", ".ima", ".IMA"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    match = _ID_PATTERN.match(stem)
    if match:
        return match.group("sid")
    return stem


def iter_dicom_scans(dicom_root: Path) -> list[tuple[str, Path]]:
    """List ``(scan_id, path)`` for every dynamic export under ``dicom_root``.

    Handles both layouts seen in this dataset: one multi-frame file per animal,
    and one directory of single-frame files per animal.
    """
    dicom_root = Path(dicom_root)
    if not dicom_root.exists():
        raise FileNotFoundError(f"DICOM root does not exist: {dicom_root}")

    found: dict[str, Path] = {}
    for entry in sorted(dicom_root.iterdir()):
        if entry.name.startswith("."):
            continue
        if not entry.name.lower().startswith("dpet_dcm_"):
            continue
        found[scan_id_from_path(entry)] = entry

    if not found:
        LOGGER.warning("No dPET_dcm_* entries found under %s", dicom_root)
    return sorted(found.items())


def read_geometry_only(path: Path, scan_id: str | None = None) -> tuple[ScanGeometry, int, dict[str, Any]]:
    """Read geometry, frame count and metadata without decoding the pixel data.

    A dynamic export here is ~87 MB, essentially all of it pixels, so reading
    the header alone turns an inventory pass over the whole dataset from many
    minutes into a few seconds.  Returns ``(geometry, n_frames, meta)``.
    """
    import pydicom

    path = Path(path)
    scan_id = scan_id or scan_id_from_path(path)

    if path.is_dir():
        scan = _read_dicom_directory(path, scan_id)
        return scan.geometry, scan.n_frames, (scan.meta or {})

    dataset = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)

    n_slices = int(getattr(dataset, "NumberOfSlices", 0) or 0)
    n_total = int(getattr(dataset, "NumberOfFrames", 0) or 0)
    if n_slices <= 0 or n_total <= 0:
        raise ValueError(f"{scan_id}: NumberOfSlices/NumberOfFrames missing from the header")
    if n_total % n_slices != 0:
        raise ValueError(f"{scan_id}: {n_total} frames is not a multiple of {n_slices} slices")

    spacing = getattr(dataset, "PixelSpacing", [1.0, 1.0])
    slice_mm = float(
        getattr(dataset, "SpacingBetweenSlices", None) or getattr(dataset, "SliceThickness", 1.0)
    )
    geometry = ScanGeometry(
        voxel_mm=(float(spacing[1]), float(spacing[0]), slice_mm),
        shape_zyx=(n_slices, int(dataset.Rows), int(dataset.Columns)),
        units=str(getattr(dataset, "Units", "SUV")),
    )

    meta = {
        "patient_name": str(getattr(dataset, "PatientName", "")),
        "patient_id": str(getattr(dataset, "PatientID", "")),
        "patient_weight_kg": float(getattr(dataset, "PatientWeight", 0.0) or 0.0),
        "series_date": str(getattr(dataset, "SeriesDate", "")),
        "study_description": str(getattr(dataset, "StudyDescription", "")),
        "corrected_image": [str(c) for c in getattr(dataset, "CorrectedImage", [])],
        "units": str(getattr(dataset, "Units", "")),
    }

    starts, durations = _frame_timing(dataset, n_total // n_slices, n_slices)
    meta["frame_start_s"] = [float(v) for v in starts]
    meta["frame_duration_s"] = [float(v) for v in durations]

    return geometry, n_total // n_slices, meta


def _private_vector(dataset: Any, tag: tuple[int, int]) -> np.ndarray | None:
    try:
        element = dataset[tag]
    except (KeyError, IndexError):
        return None
    value = element.value
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        return None
    try:
        return np.asarray(value, dtype=float).ravel()
    except (TypeError, ValueError):
        return None


def _frame_timing(dataset: Any, n_frames: int, n_slices: int) -> tuple[np.ndarray, np.ndarray]:
    """Recover frame start times (s) and durations (s).

    Prefers the PMOD private vectors, falls back to the standard
    ``FrameReferenceTime`` / ``ActualFrameDuration`` attributes, and finally to
    a uniform schedule with a loud warning -- timing must never be silently
    wrong, because the early/late-frame analysis depends on it.
    """
    starts = _private_vector(dataset, TAG_FRAME_START_S)
    durations = _private_vector(dataset, TAG_FRAME_DURATION_MS)

    if starts is not None and len(starts) >= n_frames:
        starts = starts[:n_frames].astype(float)
    else:
        starts = None

    if durations is not None and len(durations) >= n_frames:
        durations = durations[:n_frames].astype(float) / 1000.0
    else:
        durations = None

    if starts is None:
        ref = getattr(dataset, "FrameReferenceTime", None)
        if ref is not None:
            arr = np.asarray(ref, dtype=float).ravel()
            if arr.size >= n_frames * n_slices:
                starts = arr.reshape(n_frames, n_slices)[:, 0] / 1000.0
            elif arr.size >= n_frames:
                starts = arr[:n_frames] / 1000.0

    if durations is None:
        dur = getattr(dataset, "ActualFrameDuration", None)
        if dur is not None:
            arr = np.asarray(dur, dtype=float).ravel()
            if arr.size >= n_frames:
                durations = arr[:n_frames] / 1000.0
            elif arr.size == 1:
                durations = np.full(n_frames, float(arr[0]) / 1000.0)

    if starts is None or durations is None:
        LOGGER.warning(
            "Frame timing could not be read from the DICOM header; falling back to a "
            "uniform schedule. The early/late-frame analysis will not be meaningful "
            "until real timing is supplied."
        )
        if starts is None:
            starts = np.arange(n_frames, dtype=float)
        if durations is None:
            durations = np.diff(np.append(starts, starts[-1] + (starts[-1] - starts[-2] if n_frames > 1 else 1.0)))

    return np.asarray(starts, dtype=float), np.asarray(durations, dtype=float)


def read_dynamic_dicom(path: Path, scan_id: str | None = None) -> DynamicScan:
    """Read one dynamic PET export into a :class:`DynamicScan`.

    Parameters
    ----------
    path:
        A multi-frame DICOM file, or a directory of single-frame files.
    """
    import pydicom  # imported lazily so the rest of the package works without it

    path = Path(path)
    scan_id = scan_id or scan_id_from_path(path)

    if path.is_dir():
        return _read_dicom_directory(path, scan_id)

    dataset = pydicom.dcmread(str(path), force=True)
    pixels = dataset.pixel_array

    n_slices = int(getattr(dataset, "NumberOfSlices", 0) or 0)
    n_total = int(getattr(dataset, "NumberOfFrames", pixels.shape[0]))
    if n_slices <= 0:
        raise ValueError(f"{scan_id}: NumberOfSlices missing; cannot unfold the multi-frame file")
    if n_total % n_slices != 0:
        raise ValueError(
            f"{scan_id}: {n_total} frames is not a multiple of {n_slices} slices"
        )
    n_time = n_total // n_slices

    rows, cols = int(dataset.Rows), int(dataset.Columns)
    # Slice index runs fastest inside each time slot (SliceVector cycles 1..N).
    volume = pixels.reshape(n_time, n_slices, rows, cols).astype(np.float32)

    rescale = _private_vector(dataset, TAG_RESCALE_VECTOR)
    if rescale is not None and rescale.size == n_total:
        volume *= rescale.reshape(n_time, n_slices, 1, 1).astype(np.float32)
    elif rescale is not None and rescale.size == n_time:
        volume *= rescale.reshape(n_time, 1, 1, 1).astype(np.float32)
    else:
        slope = float(getattr(dataset, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(dataset, "RescaleIntercept", 0.0) or 0.0)
        if slope != 1.0 or intercept != 0.0:
            volume = volume * slope + intercept
        elif rescale is None:
            LOGGER.warning("%s: no rescale information found; using raw stored values", scan_id)

    spacing = getattr(dataset, "PixelSpacing", [1.0, 1.0])
    row_mm, col_mm = float(spacing[0]), float(spacing[1])
    slice_mm = float(
        getattr(dataset, "SpacingBetweenSlices", None)
        or getattr(dataset, "SliceThickness", 1.0)
    )
    geometry = ScanGeometry(
        voxel_mm=(col_mm, row_mm, slice_mm),   # (x, y, z)
        shape_zyx=(n_slices, rows, cols),
        units=str(getattr(dataset, "Units", "SUV")),
    )

    starts, durations = _frame_timing(dataset, n_time, n_slices)

    meta = {
        "patient_name": str(getattr(dataset, "PatientName", "")),
        "patient_id": str(getattr(dataset, "PatientID", "")),
        "patient_weight_kg": float(getattr(dataset, "PatientWeight", 0.0) or 0.0),
        "series_date": str(getattr(dataset, "SeriesDate", "")),
        "study_description": str(getattr(dataset, "StudyDescription", "")),
        "reconstruction_diameter_mm": float(getattr(dataset, "ReconstructionDiameter", 0.0) or 0.0),
        "corrected_image": [str(c) for c in getattr(dataset, "CorrectedImage", [])],
        "units": str(getattr(dataset, "Units", "")),
    }

    LOGGER.info(
        "%s: %d frames x %s voxels, %.3f x %.3f x %.3f mm, %s",
        scan_id, n_time, geometry.shape_zyx, *geometry.voxel_mm, geometry.units,
    )
    return DynamicScan(
        scan_id=scan_id,
        data=volume,
        frame_start_s=starts,
        frame_duration_s=durations,
        geometry=geometry,
        source=path,
        meta=meta,
    )


def _read_dicom_directory(directory: Path, scan_id: str) -> DynamicScan:
    """Read a per-slice DICOM directory, sorting by time then slice."""
    import pydicom

    files = [p for p in sorted(directory.rglob("*")) if p.is_file()]
    if not files:
        raise FileNotFoundError(f"{scan_id}: no files under {directory}")

    entries = []
    for file_path in files:
        try:
            ds = pydicom.dcmread(str(file_path), force=True)
        except Exception:  # noqa: BLE001 - non-DICOM stragglers are skipped
            continue
        if not hasattr(ds, "pixel_array"):
            continue
        entries.append(ds)
    if not entries:
        raise ValueError(f"{scan_id}: no readable DICOM images under {directory}")

    def sort_key(ds: Any) -> tuple[float, float]:
        t = float(getattr(ds, "FrameReferenceTime", 0.0) or 0.0)
        z = float(getattr(ds, "SliceLocation", getattr(ds, "InstanceNumber", 0)) or 0.0)
        return (t, z)

    entries.sort(key=sort_key)
    times = sorted({float(getattr(ds, "FrameReferenceTime", 0.0) or 0.0) for ds in entries})
    n_time = len(times)
    n_slices = len(entries) // max(n_time, 1)

    first = entries[0]
    rows, cols = int(first.Rows), int(first.Columns)
    volume = np.zeros((n_time, n_slices, rows, cols), dtype=np.float32)
    for index, ds in enumerate(entries):
        t_index, z_index = divmod(index, n_slices)
        if t_index >= n_time:
            break
        slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
        volume[t_index, z_index] = ds.pixel_array.astype(np.float32) * slope + intercept

    spacing = getattr(first, "PixelSpacing", [1.0, 1.0])
    geometry = ScanGeometry(
        voxel_mm=(float(spacing[1]), float(spacing[0]),
                  float(getattr(first, "SpacingBetweenSlices", None) or getattr(first, "SliceThickness", 1.0))),
        shape_zyx=(n_slices, rows, cols),
        units=str(getattr(first, "Units", "SUV")),
    )
    starts = np.asarray(times, dtype=float) / 1000.0
    durations = np.asarray(
        [float(getattr(ds, "ActualFrameDuration", 0.0) or 0.0) / 1000.0 for ds in entries[::n_slices]],
        dtype=float,
    )
    if durations.size != n_time or not np.any(durations):
        durations = np.diff(np.append(starts, starts[-1] * 2 - starts[-2] if n_time > 1 else starts[-1] + 1))

    return DynamicScan(
        scan_id=scan_id,
        data=volume,
        frame_start_s=starts,
        frame_duration_s=durations,
        geometry=geometry,
        source=directory,
    )


# --------------------------------------------------------------------------- #
# NIfTI round-trip
# --------------------------------------------------------------------------- #
def _affine_from_voxel(voxel_mm: Sequence[float]) -> np.ndarray:
    affine = np.eye(4)
    affine[0, 0], affine[1, 1], affine[2, 2] = (float(v) for v in voxel_mm)
    return affine


def write_nifti_4d(scan: DynamicScan, path: Path) -> Path:
    """Write a :class:`DynamicScan` as a 4D NIfTI with axes ``(X, Y, Z, T)``.

    PETPVC works on NIfTI, and it expects the spatial axes first, so the array
    is transposed from the DICOM ``(T, Z, Y, X)`` order here and transposed
    back on read.  Frame timing is stored alongside as a sidecar JSON, since
    NIfTI has nowhere sensible to keep it.
    """
    import json

    import nibabel as nib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data_xyzt = np.transpose(scan.data, (3, 2, 1, 0))  # (T,Z,Y,X) -> (X,Y,Z,T)
    image = nib.Nifti1Image(np.ascontiguousarray(data_xyzt, dtype=np.float32),
                            _affine_from_voxel(scan.geometry.voxel_mm))
    image.header.set_xyzt_units("mm", "sec")
    nib.save(image, str(path))

    sidecar = path.with_suffix("").with_suffix(".timing.json")
    with open(sidecar, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "scan_id": scan.scan_id,
                "frame_start_s": scan.frame_start_s.tolist(),
                "frame_duration_s": scan.frame_duration_s.tolist(),
                "geometry": scan.geometry.to_dict(),
                "meta": scan.meta or {},
                "source": str(scan.source) if scan.source else None,
            },
            handle,
            indent=2,
        )
    return path


def read_nifti_4d(path: Path, scan_id: str | None = None) -> DynamicScan:
    """Inverse of :func:`write_nifti_4d`."""
    import json

    import nibabel as nib

    path = Path(path)
    image = nib.load(str(path))
    data_xyzt = np.asarray(image.dataobj, dtype=np.float32)
    if data_xyzt.ndim == 3:
        data_xyzt = data_xyzt[..., None]
    data = np.transpose(data_xyzt, (3, 2, 1, 0))  # -> (T, Z, Y, X)

    zooms = image.header.get_zooms()[:3]
    geometry = ScanGeometry(
        voxel_mm=(float(zooms[0]), float(zooms[1]), float(zooms[2])),
        shape_zyx=(data.shape[1], data.shape[2], data.shape[3]),
    )

    sidecar = path.with_suffix("").with_suffix(".timing.json")
    meta: dict[str, Any] = {}
    if sidecar.exists():
        with open(sidecar, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        starts = np.asarray(payload["frame_start_s"], dtype=float)
        durations = np.asarray(payload["frame_duration_s"], dtype=float)
        scan_id = scan_id or payload.get("scan_id")
        meta = payload.get("meta", {})
    else:
        LOGGER.warning("No timing sidecar next to %s; frame times will be indices", path)
        starts = np.arange(data.shape[0], dtype=float)
        durations = np.ones(data.shape[0], dtype=float)

    return DynamicScan(
        scan_id=scan_id or path.stem,
        data=data,
        frame_start_s=starts,
        frame_duration_s=durations,
        geometry=geometry,
        source=path,
        meta=meta,
    )
