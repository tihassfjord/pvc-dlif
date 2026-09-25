# pvc-dlif

Partial volume correction for deep-learning-derived input functions in preclinical PET.

Dynamic PET kinetic modelling needs an arterial input function. In mice, arterial blood
sampling is invasive, near-terminal, and limited by a total blood volume of 1.5–2 mL, so a
deep-learning-derived input function (DLIF) predicted from the images is an attractive
alternative. But the structures it reads — the left ventricular cavity, the great vessels,
2–3 mm across — sit at the scanner's resolution limit, so partial volume effects bias exactly
the signal the network depends on.

This repository asks whether correcting that blur improves the predicted input function, and
it answers the question with a fully paired design: every scan is pushed through every
condition, and the conditions differ **only** by the correction under test.

It contains two things:

- **the study pipeline** — nine reproducible stages from raw DICOM to the figures and LaTeX
  tables of a thesis;
- **a phantom toolkit** — NEMA NU 4-2008 recovery coefficients, spillover ratios, uniformity
  and spatial resolution, used to characterise the recovery–noise trade-off that the dynamic
  study then tests.

Written for a master's thesis at UiT — The Arctic University of Norway (FYS-3941), against
data from the UiT ML4PET group and the UNN PET Imaging Center.

---

## What it does

```
DICOM  ──▶  native NIfTI  ──▶  RL / RVC deconvolution  ──▶  network input
                                (at reconstructed resolution)        │
                                                                     ▼
   arterial blood sampling  ◀── paired comparison ──  predicted input function
        (ground truth)                                 (pretrained + retrained)
                                        │
                                        ▼
              metrics · bias/variance · Wilcoxon · kinetics · figures · LaTeX
```

Nine stages, each resumable, each writing a provenance record:

| | Stage | Produces |
|---|---|---|
| 00 | Inventory the dataset | `manifest.json` — what exists, what is usable, and why |
| 01 | Convert and fix preprocessing | 4D NIfTI at native resolution + the per-scan crop window |
| 02 | Partial volume correction | RL and RVC corrected series + per-frame diagnostics |
| 03 | Build network inputs | one data tree per experimental condition |
| 04 | Pretrained inference | predictions from the deployed model |
| 05 | Retrain | models trained on each input representation |
| 06 | Evaluate | metrics, bias/variance, paired statistics, kinetics |
| 07 | Motion subset | detection, correction, and the cost of resampling |
| 08 | Report | thesis figures and `\input`-ready LaTeX tables |

---

## Install

PETPVC is a compiled toolbox from conda-forge and is not on PyPI, so conda does the
environment and pip does the package:

```bash
conda env create -f environment.yml
conda activate pvc-dlif
pip install -e ".[all]"

petpvc --help          # verify the toolbox is on PATH
python -m pytest tests/ # verify the install
```

A CUDA-capable GPU is effectively required for stage 05 — the full cross-validation is
3 conditions × 10 folds × 10 repeats = 300 trainings.

## Configure

Everything lives in one file. Copy the template and set four paths:

```bash
cp configs/example.yaml configs/thesis.yaml
```

```yaml
paths:
  dicom_root:     "/path/to/dynamic/dicom"    # dPET_dcm_<ID> exports
  dlif_data_root: "/path/to/dlif/data"        # AIF_SUV/ IMG_SUV_*/ VOI_SUV/
  dlif_repo:      "/path/to/DLIF"             # the model and its weights
  work:           "/path/to/output"           # everything generated lands here
```

Nothing else needs changing for a first run. The PSF values, iteration counts,
cross-validation scheme and statistical settings are already set to the published protocol.

## Run

```bash
python scripts/run_all.py --pilot     # minutes: proves the wiring, not a result
python scripts/run_all.py             # the real thing
```

Or one stage at a time — every stage takes `--config`, `--limit`, `--ids` and `--dry-run`:

```bash
python scripts/02_run_pvc.py --ids AA1 AA2 --sweep
```

## Or use the GUI

```bash
python gui/run_gui.py     # from a clone   (run_gui.bat on Windows)
pvc-dlif-gui              # once installed
```

Four tabs doing exactly what the command line does — **Setup** (paths,
key parameters, and a preflight check that names what is missing and how to
fix it), **PVC** (correct one image, a folder, or a set of thesis scans, with
method, iterations, alpha and PSF on screen), **Pipeline** (the nine stages
with status read from the work folder, run detached so closing the window does
not stop them), and **Analysis** (tables, the thesis figures, a per-scan
browser, and an export for Overleaf).

## Data

The mouse dataset and the pretrained DLIF weights are **not** distributed here — they are not
ours to redistribute. See [`docs/data.md`](docs/data.md) for what the pipeline expects and how
to obtain or substitute it. `tests/make_synthetic_dataset.py` generates a synthetic dataset in
the real format, which is enough to exercise every stage end to end.

---

## Documentation

| | |
|---|---|
| [`docs/getting-started.md`](docs/getting-started.md) | Install to first result |
| [`docs/data.md`](docs/data.md) | What the pipeline expects on disk |
| [`docs/findings.md`](docs/findings.md) | Non-obvious things discovered about this data and model — read before changing anything |
| [`docs/methods.md`](docs/methods.md) | Why the pipeline is built the way it is |
| [`docs/cluster.md`](docs/cluster.md) | Running the retraining stage as one job per fold on a cluster |

---

## Design commitments

Three choices that the results depend on. Changing any of them silently invalidates the
comparison, so they are stated here and tested in `tests/test_pipeline.py`.

**The crop window is fixed on uncorrected data and reused for every condition.** Deconvolution
shifts the centre of mass slightly; a window recomputed per condition would mean the conditions
differ by a crop as well as by the correction.

**Correction happens at the reconstructed resolution.** The measured PSF describes that grid.
The network input is a pure crop of it — no resampling — so the correction reaches the model
intact rather than being partly undone by interpolation.

**The iteration count is fixed across all frames of a scan.** Adapting it to each frame's
counting statistics would apply different amounts of resolution recovery at different times,
imposing a time-dependent bias on the very curve being measured. The iteration grid is reported
as a sensitivity analysis, never searched for the best result.

---

## Citation

See [`CITATION.cff`](CITATION.cff). This work builds on the DLIF model of Kuttner et al.
(*Frontiers in Nuclear Medicine*, 2024) and the PETPVC toolbox of Thomas et al. (*Physics in
Medicine & Biology*, 2016).

## License

MIT — see [`LICENSE`](LICENSE). The license covers this code only, not the datasets or the
pretrained weights.
