"""The scan manifest: one row per animal, the spine of every later stage.

The manifest answers, for every scan: does it have a native DICOM, a DLIF
input, and an arterial ground-truth curve; which experimental group and tracer
it belongs to; how many frames it has; whether it is excluded; and whether it
is in the motion-affected subset.

The group labels come from the DLIF repository's ``shared_dicts.groups`` and
are what the analysis stratifies on when asking whether PVC helps or hurts as a
function of animal and tracer variability.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..logging_utils import get_logger
from . import pkl_io
from .dicom_io import iter_dicom_scans

LOGGER = get_logger(__name__)

__all__ = ["ScanEntry", "Manifest", "build_manifest", "load_manifest", "GROUP_LABELS"]

# Mirrors DLIF-main/src/datahandlers/shared_dicts.py.  Kept here so the
# pipeline can build a manifest without importing the group's repo, and checked
# against it by :func:`build_manifest` when the repo is available.
GROUP_LABELS: dict[str, list[str]] = {
    "30\u03bcl": ["M2", "N2", "N4", "V4", "X1", "X2"],
    "150\u03bcl": ["O1", "O2", "O3", "W2", "Y2", "Y3"],
    "15s": ["O4", "P1", "P2", "Z2", "Z3", "AA1"],
    "60s": ["P3", "P4", "Q1", "AF3", "AF4", "AG1"],
    "Reference": ["M3", "M4", "N3", "U2", "W1", "U4", "X3", "AG3", "V3"],
    "Balb/cAnNCrl (GER)": ["AV1", "AV2", "AV3", "AV4", "AX1", "AX2", "AX4"],
    "12w": ["AN1", "AN2", "AN4", "AO2", "AO3", "AO4", "T1", "T2", "T3"],
    "16w": ["AA2", "AA3", "AA4", "AP2", "AP4", "AQ1"],
    "20w": ["AB1", "AB2", "AB3", "AQ2", "AQ3", "AQ4"],
    "24w": ["T4", "U1", "AB4", "AC1", "AR2", "AR3"],
    "C57bl6 (GER)": ["V1", "AD3", "AG4", "AH1", "AH2", "AH3", "AH4", "AI4"],
    "FDOPA": ["AS1", "AS2", "AS4", "AT1", "AT2", "AT3"],
    "PSMA": ["AT4", "AU1", "AU2", "AU3"],
    "Balb/cAnNCrl (CAN)": [
        "Sherbrooke-A", "Sherbrooke-B", "Sherbrooke-C", "Sherbrooke-D", "Sherbrooke-E",
    ],
}

# Tracer per group, so the analysis can separate FDG from the other tracers.
_TRACER_BY_GROUP = {"FDOPA": "FDOPA", "PSMA": "PSMA"}


def _group_of(scan_id: str) -> str | None:
    for label, members in GROUP_LABELS.items():
        if scan_id in members:
            return label
    return None


def _tracer_of(group: str | None) -> str:
    if group is None:
        return "FDG"
    return _TRACER_BY_GROUP.get(group, "FDG")


def _canonical(scan_id: str) -> str:
    """Normalise an ID for matching.

    The Sherbrooke scans are named ``Sherbrooke_A`` in the DICOM directory but
    ``Sherbrooke-A`` in the group's exclusion list, so a literal comparison
    silently fails to exclude them.
    """
    return str(scan_id).strip().replace("-", "_").upper()


@dataclass
class ScanEntry:
    """One animal's row in the manifest."""

    scan_id: str
    group: str | None = None
    tracer: str = "FDG"
    has_dicom: bool = False
    has_dlif_img: bool = False
    has_aif: bool = False
    has_voi: bool = False
    n_frames: int | None = None          # frames in the arterial curve
    n_dicom_frames: int | None = None    # frames in the reconstruction
    dicom_path: str | None = None
    native_shape_zyx: list[int] | None = None
    native_voxel_mm: list[float] | None = None
    units: str | None = None
    frame_start_s: list[float] | None = None
    frame_duration_s: list[float] | None = None
    excluded: bool = False
    exclusion_reason: str | None = None
    motion_affected: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        """Whether the scan can contribute to the primary comparison.

        The design is paired, so a scan is only usable if it can be pushed
        through *every* condition: it needs a native series for PVC, a DLIF
        input, and an arterial ground truth to score against.
        """
        return (
            not self.excluded
            and self.has_dicom
            and self.has_dlif_img
            and self.has_aif
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Manifest:
    """An ordered collection of :class:`ScanEntry` with convenience filters."""

    def __init__(self, entries: Sequence[ScanEntry], meta: Mapping[str, Any] | None = None):
        self.entries: list[ScanEntry] = sorted(entries, key=lambda e: e.scan_id)
        self.meta: dict[str, Any] = dict(meta or {})

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    def __getitem__(self, scan_id: str) -> ScanEntry:
        for entry in self.entries:
            if entry.scan_id == scan_id:
                return entry
        raise KeyError(scan_id)

    # ------------------------------------------------------------------ #
    @property
    def ids(self) -> list[str]:
        return [e.scan_id for e in self.entries]

    @property
    def usable_ids(self) -> list[str]:
        return [e.scan_id for e in self.entries if e.usable]

    @property
    def motion_ids(self) -> list[str]:
        return [e.scan_id for e in self.entries if e.motion_affected and e.usable]

    def by_group(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for entry in self.entries:
            if entry.usable:
                out.setdefault(entry.group or "ungrouped", []).append(entry.scan_id)
        return out

    def by_tracer(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for entry in self.entries:
            if entry.usable:
                out.setdefault(entry.tracer, []).append(entry.scan_id)
        return out

    def summary(self) -> dict[str, Any]:
        return {
            "n_total": len(self.entries),
            "n_usable": len(self.usable_ids),
            "n_excluded": sum(1 for e in self.entries if e.excluded),
            "n_missing_dicom": sum(1 for e in self.entries if not e.has_dicom and not e.excluded),
            "n_missing_dlif_img": sum(1 for e in self.entries if not e.has_dlif_img and not e.excluded),
            "n_missing_aif": sum(1 for e in self.entries if not e.has_aif and not e.excluded),
            "n_motion_affected": len(self.motion_ids),
            "groups": {k: len(v) for k, v in sorted(self.by_group().items())},
            "tracers": {k: len(v) for k, v in sorted(self.by_tracer().items())},
        }

    # ------------------------------------------------------------------ #
    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(
                {"meta": self.meta, "summary": self.summary(),
                 "scans": [e.to_dict() for e in self.entries]},
                handle, indent=2,
            )
        return path

    def to_dataframe(self, with_schedules: bool = False):
        """One row per scan.  The per-frame schedules stay out by default --
        they belong in the JSON, not in a spreadsheet cell."""
        import pandas as pd

        frame = pd.DataFrame([e.to_dict() for e in self.entries])
        if not with_schedules:
            frame = frame.drop(columns=[c for c in ("frame_start_s", "frame_duration_s") if c in frame])
        return frame


def load_manifest(path: Path) -> Manifest:
    with open(Path(path), "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    entries = [ScanEntry(**row) for row in payload["scans"]]
    return Manifest(entries, payload.get("meta"))


def _groups_from_repo(dlif_repo: Path) -> dict[str, list[str]] | None:
    """Read ``shared_dicts.groups`` from the group's repository, if present."""
    candidate = Path(dlif_repo) / "src" / "datahandlers" / "shared_dicts.py"
    if not candidate.exists():
        return None
    namespace: dict[str, Any] = {}
    try:
        exec(compile(candidate.read_text(encoding="utf-8"), str(candidate), "exec"), namespace)  # noqa: S102
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Could not read group labels from %s: %s", candidate, exc)
        return None
    groups = namespace.get("groups")
    if isinstance(groups, Mapping):
        return {str(k): [str(v) for v in vals] for k, vals in groups.items()}
    return None


def build_manifest(
    dicom_root: Path,
    dlif_data_root: Path,
    dlif_repo: Path | None = None,
    img_shape: Sequence[int] = (96, 48, 48),
    ignore_ids: Iterable[str] = (),
    motion_affected: Iterable[str] = (),
    read_geometry: bool = False,
    verify_inputs: bool = False,
) -> Manifest:
    """Inventory every scan and record what is available for it.

    ``read_geometry=True`` reads each DICOM *header* to record the
    reconstructed grid and the frame schedule.  It costs milliseconds per scan
    and is worth doing every time: a scan reconstructed on a different grid, or
    acquired on a different frame schedule, must not be silently mixed into the
    comparison.

    ``verify_inputs=True`` additionally opens every DLIF input pickle to check
    its shape against the arterial curve.  That reads the whole dataset -- tens
    of gigabytes -- so it is off by default and worth running once, not on
    every inventory.
    """
    ignore = {_canonical(i) for i in ignore_ids}
    motion = {_canonical(i) for i in motion_affected}

    repo_groups = _groups_from_repo(dlif_repo) if dlif_repo else None
    if repo_groups:
        mismatch = {k for k in repo_groups if k not in GROUP_LABELS} | {
            k for k in GROUP_LABELS if k not in repo_groups
        }
        if mismatch:
            LOGGER.warning(
                "Group labels differ from the DLIF repository (%s); using the repository's",
                sorted(mismatch),
            )
        groups = repo_groups
    else:
        groups = GROUP_LABELS

    canonical_groups = {
        label: {_canonical(m) for m in members} for label, members in groups.items()
    }

    def group_of(scan_id: str) -> str | None:
        key = _canonical(scan_id)
        for label, members in canonical_groups.items():
            if key in members:
                return label
        return None

    dicom_entries = dict(iter_dicom_scans(Path(dicom_root))) if Path(dicom_root).exists() else {}
    dlif_ids = set(pkl_io.available_ids(dlif_data_root, img_shape, require_aif=False))

    # Union of every ID the dataset knows about, de-duplicated across the
    # naming variants (Sherbrooke_A / Sherbrooke-A) so one animal is one row.
    all_ids: dict[str, str] = {}
    for source in (dicom_entries, dlif_ids, *groups.values()):
        for scan_id in source:
            all_ids.setdefault(_canonical(scan_id), str(scan_id))
    all_ids = sorted(all_ids.values())

    entries: list[ScanEntry] = []
    for scan_id in all_ids:
        group = group_of(scan_id)
        entry = ScanEntry(
            scan_id=scan_id,
            group=group,
            tracer=_tracer_of(group),
            has_dicom=scan_id in dicom_entries,
            has_dlif_img=scan_id in dlif_ids,
            has_aif=pkl_io.aif_path(dlif_data_root, scan_id).exists(),
            has_voi=pkl_io.voi_path(dlif_data_root, scan_id).exists(),
            dicom_path=str(dicom_entries[scan_id]) if scan_id in dicom_entries else None,
            motion_affected=_canonical(scan_id) in motion,
        )
        if _canonical(scan_id) in ignore:
            entry.excluded = True
            entry.exclusion_reason = "listed in dlif.ignore_ids"

        # The frame count comes from the arterial curve, which is ~1 kB.
        # Reading it from the image pickle instead would mean unpickling ~75 MB
        # per scan -- several gigabytes across the dataset -- for one integer.
        if entry.has_aif:
            try:
                aif, _ = pkl_io.load_aif(dlif_data_root, scan_id)
                entry.n_frames = int(aif.shape[0])
            except Exception as exc:  # noqa: BLE001
                entry.notes.append(f"could not read the arterial curve: {exc}")
                entry.has_aif = False

        if verify_inputs and entry.has_dlif_img:
            try:
                image, _ = pkl_io.load_img(dlif_data_root, scan_id, img_shape)
                if entry.n_frames is not None and int(image.shape[0]) != entry.n_frames:
                    entry.notes.append(
                        f"DLIF input has {image.shape[0]} frames but the arterial curve has "
                        f"{entry.n_frames}"
                    )
                if tuple(image.shape[1:]) != tuple(img_shape):
                    entry.notes.append(
                        f"DLIF input is {tuple(image.shape[1:])}, expected {tuple(img_shape)}"
                    )
            except Exception as exc:  # noqa: BLE001
                entry.notes.append(f"could not read DLIF input: {exc}")
                entry.has_dlif_img = False

        if read_geometry and entry.has_dicom:
            try:
                from .dicom_io import read_geometry_only

                geometry, n_frames, meta = read_geometry_only(Path(entry.dicom_path), scan_id)  # type: ignore[arg-type]
                entry.native_shape_zyx = list(geometry.shape_zyx)
                entry.native_voxel_mm = [round(v, 6) for v in geometry.voxel_mm]
                entry.n_dicom_frames = n_frames
                entry.units = geometry.units
                entry.frame_start_s = [round(v, 3) for v in meta.get("frame_start_s", [])]
                entry.frame_duration_s = [round(v, 3) for v in meta.get("frame_duration_s", [])]
                if entry.n_frames is not None and n_frames != entry.n_frames:
                    entry.notes.append(
                        f"frame count mismatch: {n_frames} in DICOM vs {entry.n_frames} in DLIF input"
                    )
            except Exception as exc:  # noqa: BLE001
                entry.notes.append(f"could not read DICOM geometry: {exc}")

        entries.append(entry)

    manifest = Manifest(entries, meta={
        "dicom_root": str(dicom_root),
        "dlif_data_root": str(dlif_data_root),
        "img_shape": list(img_shape),
        "group_source": "dlif_repo" if repo_groups else "builtin",
    })

    # Voxel size decides how the measured PSF maps onto the grid, so a
    # difference there changes the strength of the deconvolution.
    voxel_sizes = {
        tuple(e.native_voxel_mm) for e in manifest.entries if e.native_voxel_mm is not None
    }
    if len(voxel_sizes) > 1:
        LOGGER.warning(
            "Scans use %d different voxel sizes: %s. The PSF is specified in millimetres, so "
            "this changes the deconvolution strength per scan; check before correcting.",
            len(voxel_sizes), sorted(voxel_sizes),
        )
        manifest.meta["voxel_size_warning"] = [list(v) for v in sorted(voxel_sizes)]

    # Matrix size decides the crop onto the DLIF input grid.  Two scans can
    # share a voxel size and still have different fields of view, and a single
    # crop fitted on one of them would be wrong for the other.
    shapes: dict[tuple[int, ...], list[str]] = {}
    for entry in manifest.entries:
        if entry.native_shape_zyx:
            shapes.setdefault(tuple(entry.native_shape_zyx), []).append(entry.scan_id)
    manifest.meta["native_shapes"] = {
        "x".join(str(v) for v in shape): sorted(ids) for shape, ids in shapes.items()
    }
    if len(shapes) > 1:
        counts = {"x".join(str(v) for v in s): len(ids) for s, ids in shapes.items()}
        LOGGER.warning(
            "Scans were reconstructed onto %d different matrix sizes (%s). The crop onto the "
            "DLIF input grid must be fitted per matrix size, not once for the dataset.",
            len(shapes), counts,
        )
        majority_shape = max(shapes, key=lambda s: len(shapes[s]))
        for entry in manifest.entries:
            if entry.native_shape_zyx and tuple(entry.native_shape_zyx) != majority_shape:
                entry.notes.append(
                    "reconstructed on a minority matrix size "
                    f"({'x'.join(str(v) for v in entry.native_shape_zyx)})"
                )

    # The ground-truth curves are in SUV, so a scan reconstructed on another
    # scale would bias every metric for that animal.  Some exports tag the
    # units as '_SUV_' rather than 'SUV'; that decoration is cosmetic, and
    # flagging it would bury a real unit mismatch in noise.
    units = {e.units for e in manifest.entries if e.units}
    manifest.meta["units_seen"] = sorted(units)
    non_suv = {u for u in units if u.strip().strip("_").upper() != "SUV"}
    if non_suv:
        LOGGER.warning(
            "Some scans are not in SUV: %s. The arterial curves are in SUV, so these cannot "
            "be compared against them without converting first.",
            sorted(non_suv),
        )
        for entry in manifest.entries:
            if entry.units and entry.units.strip().strip("_").upper() != "SUV":
                entry.notes.append(f"units are {entry.units!r}, not SUV")

    # Different frame schedules would make the frame-wise analysis compare
    # different time windows across animals, and the network takes a fixed
    # number of frames.
    schedules: dict[tuple[float, ...], list[str]] = {}
    for entry in manifest.entries:
        if entry.frame_start_s:
            schedules.setdefault(tuple(entry.frame_start_s), []).append(entry.scan_id)
    if len(schedules) > 1:
        sizes = sorted(((len(v), len(k)) for k, v in schedules.items()), reverse=True)
        LOGGER.warning(
            "Scans use %d different frame schedules (largest group %d scans, %d frames). "
            "The frame-wise analysis compares frames by index, so scans on a minority "
            "schedule must be excluded or handled separately.",
            len(schedules), sizes[0][0], sizes[0][1],
        )
        majority = max(schedules.values(), key=len)
        manifest.meta["frame_schedule_groups"] = {
            f"{len(ids)} scans, {len(starts)} frames": sorted(ids)
            for starts, ids in sorted(schedules.items(), key=lambda kv: -len(kv[1]))
        }
        for entry in manifest.entries:
            if entry.frame_start_s and entry.scan_id not in majority:
                entry.notes.append("frame schedule differs from the majority of the dataset")

    LOGGER.info("Manifest: %s", json.dumps(manifest.summary(), indent=None))
    return manifest
