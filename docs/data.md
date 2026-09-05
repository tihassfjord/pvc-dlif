# Data

Neither the mouse dataset nor the pretrained DLIF weights are distributed with
this repository. They belong to the UiT ML4PET group and the UNN PET Imaging
Center, and are available from them on request. This document says exactly what
the pipeline expects, so the same layout can be assembled from another dataset.

---

## The three input trees

```
dicom_root/
  dPET_dcm_AA1/            one folder (or one multi-frame file) per scan
  dPET_dcm_AA2/
  …

dlif_data_root/
  AIF_SUV/       AIF_AA1.pkl        arterial ground truth
  IMG_SUV_96x48x48/  IMG_AA1.pkl    the network inputs the group distributes
  VOI_SUV/       VOI_AA1.pkl        tissue curves, for the kinetic modelling

dlif_repo/                          a clone of the group's DLIF repository
  src/models/models.py
  src/models/pretrained_weigths/DLIFNet.pt
  src/datahandlers/shared_dicts.py  group membership, used for stratification
```

The scan ID is the part after `dPET_dcm_`, and it is the key that joins all
three trees. Stage 00 reports any ID that is present in one tree and missing
from another rather than dropping it silently.

---

## What each file has to contain

**DICOM.** A dynamic PET series, one frame per temporal position. Frame timing
is read from the PMOD private tags rather than the standard attributes:

| Tag | Meaning | Units |
|---|---|---|
| `(0055,1001)` | frame start times | seconds |
| `(0055,1004)` | frame durations | milliseconds |
| `(0055,1005)` | per-frame rescale slopes | — |

If your export puts timing in the standard attributes instead, `data/dicom_io.py`
is where to add that path. Everything downstream works from the parsed
`DynamicScan`, not from DICOM.

**`AIF_<ID>.pkl`.** A dict with the measured arterial curve and its time axis.
Small — about a kilobyte. Stage 00 reads this to count frames, which is why the
inventory takes seconds rather than unpickling gigabytes of image data.

**`IMG_<ID>.pkl`.** A dict with key `IMG`, holding a `(frames, 96, 48, 48)`
array, and `IDIF_t`, the time axis in minutes. This is the group's own
preprocessed input. The pipeline does not consume it directly — it rebuilds the
inputs from DICOM — but stage 01 fits its crop window *against* this file and
reports the correlation, which is the check that the rebuild is faithful. See
[`findings.md`](findings.md) §3.

**`VOI_<ID>.pkl`.** Tissue time-activity curves keyed by region name (`Brain`,
`Myocardium`, `Liver`, …). Only needed for the kinetic modelling in stage 06;
without it that section is skipped with a warning.

---

## Acquisition parameters

Set in `configs/*.yaml`, and correct for the LabPET8 dataset this was built for:

| | |
|---|---|
| Reconstruction | 128 × 92 × 92 voxels |
| Voxel size | 0.5 × 0.5 × 0.59675 mm |
| Field of view | 76.4 × 46 × 46 mm |
| Frames | 42 |
| Frame schedule | 1 × 30 s, 24 × 5 s, 9 × 20 s, 8 × 300 s |
| Total duration | 2730 s (45.5 min) |
| Units | SUV |
| PSF FWHM | 0.864 / 0.874 / 0.994 mm |

**The PSF is the one parameter you must change for another scanner.** It is what
the deconvolution inverts, and getting it wrong is not a small error: `psf.fwhm_mm`
in the config, and `sensitivity.psf_scale` runs the ±10 % mismatch check.

---

## Which scans are used

Of 102 dynamic exports, 70 are usable:

| | Scans |
|---|---|
| Dynamic exports in the archive | 102 |
| Excluded by the group's `ignore_ids` | −32 |
| No paired arterial curve (Sherbrooke series) | −5 |
| **Usable, fully paired** | **70** |

All 70 are [¹⁸F]FDG — the exclusion list removes every FDOPA and PSMA scan, so
no claim about tracer generalisation is available without revisiting it.

The exclusion list is `dlif.ignore_ids` in the config, carried over verbatim
from the group's own configuration so the analysis set matches theirs. Two
naming variants exist for the same animals (`Sherbrooke_A` in the DICOM
directory, `Sherbrooke-A` in the exclusion list); the manifest normalises them,
because otherwise those exclusions fail silently.

---

## Using a different dataset

The pipeline needs three things per scan, and nothing else:

1. a dynamic PET series with per-frame timing,
2. a measured blood curve on the same time axis,
3. a scan ID that joins them.

The DLIF-specific parts — the distributed `IMG_*.pkl` and the pretrained
weights — are only needed for the pretrained conditions. Drop those conditions
from `conditions:` in the config and the retrained arm runs on your own data
alone, with the crop window placed from the body threshold rather than fitted
against a distributed input.

`tests/make_synthetic_dataset.py` generates a complete dataset in this format,
including DICOM with the private timing tags. It is the reference for what the
readers expect, and it is small enough to iterate on.

---

## Publishing

If you fork this for your own work: the code here is MIT, but the dataset and
the pretrained weights are not covered by it and are not yours to redistribute.
Ship the code, ship this document, and let people obtain the data from its
owners.
