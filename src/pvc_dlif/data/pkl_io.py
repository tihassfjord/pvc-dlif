"""Reading and writing the group's pickle data format.

The DLIF repository stores one pickle per scan per modality::

    <root>/AIF_SUV/AIF_<ID>.pkl          {"AIF_t_int", "AIF_A_int", "AIF_A_int_disp"}
    <root>/IMG_SUV_96x48x48/IMG_<ID>.pkl {"IMG": (T, 96, 48, 48), "IDIF_t": (T,)}
    <root>/VOI_SUV/VOI_<ID>.pkl          {"VOI_t", "VOI_A": {"Brain","LV","Liver","Myocardium"}}

PVC-corrected inputs are written in exactly this layout, so the group's
dataloader, trainer and evaluator run against them unchanged -- only the data
root differs.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from ..logging_utils import get_logger

LOGGER = get_logger(__name__)

__all__ = [
    "IMG_KEY", "AIF_KEY", "AIF_TIME_KEY",
    "img_dirname", "img_path", "aif_path", "voi_path",
    "load_img", "load_aif", "load_voi", "save_img",
    "available_ids", "ScanRecord", "load_scan_record",
]

IMG_KEY = "IMG"
IMG_TIME_KEY = "IDIF_t"
AIF_KEY = "AIF_A_int"
AIF_TIME_KEY = "AIF_t_int"
AIF_DISP_KEY = "AIF_A_int_disp"


def img_dirname(shape: Iterable[int]) -> str:
    """``(96, 48, 48)`` -> ``"IMG_SUV_96x48x48"``."""
    return "IMG_SUV_" + "x".join(str(int(s)) for s in shape)


def img_path(root: Path, scan_id: str, shape: Iterable[int] = (96, 48, 48)) -> Path:
    return Path(root) / img_dirname(shape) / f"IMG_{scan_id}.pkl"


def aif_path(root: Path, scan_id: str) -> Path:
    return Path(root) / "AIF_SUV" / f"AIF_{scan_id}.pkl"


def voi_path(root: Path, scan_id: str) -> Path:
    return Path(root) / "VOI_SUV" / f"VOI_{scan_id}.pkl"


def _read_pickle(path: Path) -> dict[str, Any]:
    with open(Path(path), "rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, Mapping):
        raise TypeError(f"{path}: expected a dict, got {type(payload).__name__}")
    return dict(payload)


def load_img(root: Path, scan_id: str, shape: Iterable[int] = (96, 48, 48)) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(image (T, *shape), frame times in minutes)``."""
    payload = _read_pickle(img_path(root, scan_id, shape))
    image = np.asarray(payload[IMG_KEY], dtype=np.float32)
    times = np.asarray(payload.get(IMG_TIME_KEY, np.arange(image.shape[0])), dtype=float)
    return image, times


def load_aif(root: Path, scan_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(arterial input function in SUV, time in minutes)``.

    This is the arterial blood sampling ground truth every condition is scored
    against.
    """
    payload = _read_pickle(aif_path(root, scan_id))
    activity = np.asarray(payload[AIF_KEY], dtype=float)
    times = np.asarray(payload[AIF_TIME_KEY], dtype=float)
    return activity, times


def load_voi(root: Path, scan_id: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Return ``({voi name: time-activity curve}, time in minutes)``."""
    payload = _read_pickle(voi_path(root, scan_id))
    curves = {str(k): np.asarray(v, dtype=float) for k, v in dict(payload["VOI_A"]).items()}
    times = np.asarray(payload["VOI_t"], dtype=float)
    return curves, times


def save_img(
    root: Path,
    scan_id: str,
    image: np.ndarray,
    times: np.ndarray,
    shape: Iterable[int] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write an ``IMG_<ID>.pkl`` in the group's format.

    ``image`` is stored as float64 to match the distributed files exactly, so a
    corrected data root is byte-compatible with the group's dataloader.
    """
    image = np.asarray(image)
    if image.ndim != 4:
        raise ValueError(f"{scan_id}: expected a 4D image (T, Z, Y, X), got {image.shape}")
    shape = tuple(shape) if shape is not None else tuple(image.shape[1:])

    path = img_path(root, scan_id, shape)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        IMG_KEY: np.asarray(image, dtype=np.float64),
        IMG_TIME_KEY: np.asarray(times, dtype=np.float64),
    }
    if extra:
        payload.update(dict(extra))
    with open(path, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def available_ids(root: Path, shape: Iterable[int] = (96, 48, 48), require_aif: bool = True) -> list[str]:
    """Scan IDs that have both an image and (optionally) a ground-truth AIF."""
    root = Path(root)
    image_dir = root / img_dirname(shape)
    if not image_dir.exists():
        return []
    ids = sorted(p.stem[len("IMG_"):] for p in image_dir.glob("IMG_*.pkl"))
    if require_aif:
        ids = [i for i in ids if aif_path(root, i).exists()]
    return ids


@dataclass
class ScanRecord:
    """Everything one scan contributes to the study."""

    scan_id: str
    image: np.ndarray            # (T, 96, 48, 48) SUV
    image_time_min: np.ndarray   # (T,)
    aif: np.ndarray              # (T,) SUV, arterial sampling ground truth
    aif_time_min: np.ndarray     # (T,)
    voi: dict[str, np.ndarray] | None = None
    voi_time_min: np.ndarray | None = None

    @property
    def n_frames(self) -> int:
        return int(self.image.shape[0])


def load_scan_record(
    root: Path,
    scan_id: str,
    shape: Iterable[int] = (96, 48, 48),
    with_voi: bool = True,
) -> ScanRecord:
    image, image_time = load_img(root, scan_id, shape)
    aif, aif_time = load_aif(root, scan_id)
    voi = voi_time = None
    if with_voi and voi_path(root, scan_id).exists():
        voi, voi_time = load_voi(root, scan_id)

    if image.shape[0] != aif.shape[0]:
        LOGGER.warning(
            "%s: %d image frames but %d AIF samples", scan_id, image.shape[0], aif.shape[0]
        )
    return ScanRecord(
        scan_id=scan_id,
        image=image,
        image_time_min=image_time,
        aif=aif,
        aif_time_min=aif_time,
        voi=voi,
        voi_time_min=voi_time,
    )
