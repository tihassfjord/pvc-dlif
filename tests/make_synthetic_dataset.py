"""Build a small synthetic dataset that mimics the real one.

Used to exercise the whole pipeline end to end without the 7 GB of real data:
multi-frame PMOD-style DICOM exports, matching ``AIF_``/``IMG_``/``VOI_``
pickles, and a config pointing at them.  The images are a blurred blood pool
plus tissue, so PVC has something real to recover, and the ground-truth input
function is the unblurred blood-pool curve.

    python tests/make_synthetic_dataset.py --out /tmp/synthetic
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

NATIVE_SHAPE = (96, 64, 64)     # (slices, rows, columns)
VOXEL_MM = (0.5, 0.5, 0.59675)  # (x, y, z), as on the LabPET8
N_FRAMES = 12

FRAME_START_S = np.array([0, 30, 35, 40, 50, 65, 90, 130, 210, 400, 900, 1800], float)
FRAME_DURATION_S = np.array([30, 5, 5, 10, 15, 25, 40, 80, 190, 500, 900, 900], float)

SCAN_IDS = [
    "M2", "N2", "O1", "O2", "O4", "P1", "P3", "P4",
    "M3", "M4", "N3", "U2", "AA1", "AA2", "AB1", "AB2",
    "T4", "U1", "V1", "AD3",
]


def _input_function(times_min: np.ndarray, peak: float, peak_time: float) -> np.ndarray:
    """A plausible bolus: sharp rise, fast washout, slow tail."""
    rise = 1.0 / (1.0 + np.exp(-(times_min - peak_time * 0.6) * 25))
    fast = np.exp(-(times_min - peak_time).clip(0) * 3.0)
    slow = np.exp(-(times_min - peak_time).clip(0) * 0.05)
    return peak * rise * (0.75 * fast + 0.25 * slow)


def _volume(blood: float, tissue: float, rng: np.random.Generator) -> np.ndarray:
    """One frame: a small hot blood pool inside a warmer tissue body."""
    z, y, x = NATIVE_SHAPE
    volume = np.zeros(NATIVE_SHAPE, dtype=np.float32)

    zz, yy, xx = np.ogrid[:z, :y, :x]
    body = (
        ((yy - y / 2) / (y * 0.35)) ** 2
        + ((xx - x / 2) / (x * 0.35)) ** 2
        + ((zz - z / 2) / (z * 0.42)) ** 2
    ) <= 1.0
    volume[body] = tissue

    # Blood pool: ~2.5 mm across at 0.5 mm voxels, close to the system PSF --
    # the regime where partial volume effects bite.
    radius_vox = 2.5
    pool = (
        ((zz - z * 0.45) / radius_vox) ** 2
        + ((yy - y / 2) / radius_vox) ** 2
        + ((xx - x / 2) / radius_vox) ** 2
    ) <= 1.0
    volume[pool] = blood

    return volume + rng.normal(0, max(tissue, 0.05) * 0.02, NATIVE_SHAPE).astype(np.float32)


def _write_dicom(path: Path, series: np.ndarray, scan_id: str) -> None:
    import pydicom
    from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    n_t, n_z, n_y, n_x = series.shape
    scale = float(series.max()) / 32000.0 or 1.0
    stored = np.clip(series / scale, 0, 32767).astype(np.int16)

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.20"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.ImplementationClassUID = generate_uid()

    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Modality = "PT"
    ds.Manufacturer = "PMOD Technologies"
    ds.PatientName = f"DLIF_{scan_id}"
    ds.PatientID = scan_id
    ds.PatientWeight = 0.0215
    ds.StudyDescription = "SUV AC corr"
    ds.SeriesDescription = "[Rot] [Scale]"
    ds.Units = "SUV"
    ds.Rows, ds.Columns = n_y, n_x
    ds.NumberOfFrames = n_t * n_z
    ds.NumberOfSlices = n_z
    ds.NumberOfTimeSlots = n_t
    ds.PixelSpacing = [VOXEL_MM[1], VOXEL_MM[0]]
    ds.SliceThickness = VOXEL_MM[2]
    ds.SpacingBetweenSlices = VOXEL_MM[2]
    ds.ReconstructionDiameter = n_x * VOXEL_MM[0]
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    ds.SliceVector = list(range(1, n_z + 1)) * n_t
    ds.TimeSlotVector = [i for i in range(1, n_t + 1) for _ in range(n_z)]

    # PMOD private tags: frame start times (s), durations (ms), rescale slopes.
    block = ds.private_block(0x0055, "PMOD_1", create=True)
    block.add_new(0x01, "FD", list(FRAME_START_S[:n_t]))
    block.add_new(0x04, "FD", list(FRAME_DURATION_S[:n_t] * 1000.0))
    block.add_new(0x05, "FD", [scale] * (n_t * n_z))

    ds.PixelData = stored.reshape(-1, n_y, n_x).tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(path), enforce_file_format=False)


def build(out: Path, n_scans: int = 20, seed: int = 0) -> dict:
    from scipy.ndimage import gaussian_filter, zoom

    out = Path(out)
    dicom_root = out / "dicom"
    data_root = out / "dlif_data"
    rng = np.random.default_rng(seed)

    times_min = (FRAME_START_S + FRAME_DURATION_S / 2) / 60.0
    sigma_vox = tuple(
        (f / v) / (2 * np.sqrt(2 * np.log(2)))
        for f, v in zip((0.864, 0.874, 0.994), (VOXEL_MM[2], VOXEL_MM[1], VOXEL_MM[0]))
    )

    scan_ids = SCAN_IDS[:n_scans]
    for index, scan_id in enumerate(scan_ids):
        peak = 8.0 + rng.normal(0, 1.5)
        peak_time = 0.6 + rng.normal(0, 0.05)
        blood = _input_function(times_min, peak, peak_time)
        tissue = 0.25 * np.cumsum(blood) / max(len(blood), 1) + 0.1 * blood

        sharp = np.stack([_volume(blood[t], tissue[t], rng) for t in range(len(blood))])
        # The scanner blurs: this is the partial volume effect PVC has to undo.
        blurred = np.stack([gaussian_filter(v, sigma_vox) for v in sharp])

        _write_dicom(dicom_root / f"dPET_dcm_{scan_id}", blurred, scan_id)

        (data_root / "AIF_SUV").mkdir(parents=True, exist_ok=True)
        with open(data_root / "AIF_SUV" / f"AIF_{scan_id}.pkl", "wb") as handle:
            pickle.dump(
                {"AIF_t_int": times_min, "AIF_A_int": blood, "AIF_A_int_disp": blood},
                handle,
            )

        (data_root / "VOI_SUV").mkdir(parents=True, exist_ok=True)
        with open(data_root / "VOI_SUV" / f"VOI_{scan_id}.pkl", "wb") as handle:
            pickle.dump(
                {"VOI_t": times_min,
                 "VOI_A": {"LV": blood * 0.9, "Brain": tissue * 0.8,
                           "Liver": tissue * 1.2, "Myocardium": tissue}},
                handle,
            )

        # The reference DLIF tree, built the way the real one is: a pure crop
        # at native resolution, offset per animal, times a global factor.
        z0 = 10 + (index % 5) * 3
        y0 = 6 + (index % 4) * 2
        x0 = 7 + (index % 3) * 2
        reference = blurred[:, z0:z0 + 64, y0:y0 + 48, x0:x0 + 48] * 0.9937
        directory = data_root / "IMG_SUV_64x48x48"
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / f"IMG_{scan_id}.pkl", "wb") as handle:
            pickle.dump({"IMG": reference.astype(np.float64), "IDIF_t": times_min}, handle)

    return {"dicom_root": str(dicom_root), "dlif_data_root": str(data_root), "n_scans": len(scan_ids)}


def write_config(out: Path, info: dict, target: Path) -> Path:
    """Write a config pointing at the synthetic dataset, based on example.yaml.

    Only the four paths are replaced, so the synthetic run exercises exactly the
    settings the real run uses.
    """
    import re

    template = (Path(__file__).resolve().parents[1] / "configs" / "example.yaml")
    text = template.read_text(encoding="utf-8")
    substitutions = {
        "dicom_root": info["dicom_root"],
        "dlif_data_root": info["dlif_data_root"],
        "dlif_repo": str(Path(out) / "dlif_repo"),   # stub; stages 04/05 need the real one
        "work": str(Path(out) / "work"),
    }
    for key, value in substitutions.items():
        text = re.sub(rf'^(\s*{key}:\s*).*$', lambda m, v=value: f'{m.group(1)}"{v}"',
                      text, count=1, flags=re.MULTILINE)

    # The generator writes IMG_SUV_64x48x48 and invents its own scan IDs, so the
    # reference shape and the real exclusion list both have to be relaxed.
    text = re.sub(r'^(\s*reference_shape:\s*).*$', r'\g<1>[64, 48, 48]',
                  text, count=1, flags=re.MULTILINE)
    text = re.sub(r'^(\s*ignore_ids:).*?(?=^\S)', '\\g<1> []\n\n',
                  text, count=1, flags=re.MULTILINE | re.DOTALL)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="/tmp/synthetic")
    parser.add_argument("--n-scans", type=int, default=20)
    parser.add_argument("--write-config", nargs="?", const="configs/synthetic.yaml",
                        default=None,
                        help="also write a config pointing at the generated data")
    args = parser.parse_args()
    info = build(Path(args.out), args.n_scans)
    print(info)
    if args.write_config:
        path = write_config(Path(args.out), info, Path(args.write_config))
        print(f"config: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
