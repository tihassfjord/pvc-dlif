# Task: finish the pvc-dlif GUI and drive the first real run

You are taking over a working but unfinished research repository for a master's thesis
(FYS-3941, UiT). The owner is a physics student who knows Python and reads code
comfortably — he is not afraid of the source, he just doesn't want to type long command
lines dozens of times while producing thesis data.

Two things are wanted, in this order:

1. **Finish the Tkinter GUI.** A skeleton exists and runs. It is roughly 60 % of what is
   specified in section 5; your job is the rest, not a rewrite.
2. **Drive the first full run on the real 70-scan dataset** and fix what surfaces.

Read sections 2, 3 and 8 before writing code. They exist to stop you re-deriving things that
already cost days, and to stop you silently breaking the science.

---

## 1. Where everything is

The project was consolidated into a single standalone repository. **Work only in this
repository.** The older scattered folders (`MASTER`, `MASTER1`, `Simplified_Project`,
`ML4PET_MASTER\kode`) are kept for reference and must not be edited — anything still needed
from them has already been carried across.

| What | Path (Windows) |
|---|---|
| **The repository — everything you touch** | `E:\ML4PET\pvc-dlif` |
| Raw DICOM, 102 dynamic mouse scans | `E:\ML4PET\ML4PET_DLIF\aif-mice-dicom-pet` |
| Group's DLIF data (AIF / IMG / VOI pickles) | `E:\ML4PET\ML4PET_DLIF\data` |
| Group's DLIF repo + pretrained weights | `E:\ML4PET\ML4PET_DLIF\DLIF-main(1)\DLIF-main` |
| All generated output | `E:\ML4PET\thesis_work` (set in the config) |
| Thesis LaTeX | `ML4PET_MASTER\00_MASTER_THESIS\tex\01_Draft\` |
| Old pipeline copy (reference only, do not edit) | `ML4PET_MASTER\thesis_pipeline` |

```
pvc-dlif/
├── configs/thesis.yaml           # single source of truth; example.yaml is the template
├── src/pvc_dlif/                 # library: data/ pvc/ dlif/ eval/ motion/ report/ phantom/
├── scripts/00..08_*.py           # one CLI per stage, + run_all.py
├── gui/pvc_dlif_gui/             # the GUI skeleton you are finishing
├── tests/test_pipeline.py        # 85 tests, all passing
│   tests/make_synthetic_dataset.py  # fake dataset in the real format (--write-config)
└── docs/                         # getting-started, data, findings, methods
```

Note the package is now `pvc_dlif` (it was `ml4pet_thesis`). `src/pvc_dlif/phantom/` is the
NEMA NU 4-2008 phantom toolkit from the earlier project thesis, carried across because the
thesis cites it; it is independent of stages 00–08.

Stages: `00` inventory → `01` convert + fix preprocessing → `02` PVC → `03` build network
inputs → `04` pretrained inference → `05` retrain → `06` evaluate → `07` motion subset →
`08` figures + LaTeX tables. Every stage is resumable and takes `--config --limit --ids
--dry-run`.

**`docs/findings.md` is the file to read before changing anything in `data/` or `dlif/`.**
Section 3 below is its summary.

---

## 2. How the owner wants code written

He will read and maintain this. Match the style already in `gui/pvc_dlif_gui/`:

- **Short docstrings on every function and class** — what it does, what the parameters mean,
  what comes back.
- **Explanatory comments where the reason is not obvious from the code.** Say *why*, not what.
  A comment explaining a physics or statistics choice is worth five explaining syntax.
- **Plain, obvious Python.** No metaclasses, no clever decorators, no dependency injection.
  Straightforward functions and one class per screen.
- **Keep the GUI thin.** All computation stays in `src/pvc_dlif/`; the GUI collects
  parameters, calls a library function, and shows the result. A GUI that grows its own maths
  becomes a second implementation that drifts.
- Type hints where they clarify, skipped where they clutter.
- No file past a few hundred lines. One tab per module.

---

## 3. Five findings that must NOT be re-derived or reverted

Established empirically against the real data. Each is load-bearing. Full detail with the
numbers is in `docs/findings.md`.

**3.1 — `DLIFNet.pt` does not match the repo's `models.py`.**
The checkpoint is a pickled `nn.Module`, not a state dict, and carries the *original*
Frontiers architecture: 2-channel encoder of width 8/16/32/64/128 with BatchNorm, TCN
256→128→64→32, 2 203 104 parameters. The repo's current `models.py` builds a narrower network
with BatchNorm commented out. Rebuilding from that code and loading the weights transfers
**2 of 18 tensors** — a "pretrained baseline" that is ~95 % random, and it fails quietly.
→ Always load via `pvc_dlif.dlif.adapter.load_pretrained_module()`. Do not "fix" this by
editing the group's repo.

**3.2 — The pretrained model needs 64 × 48 × 48 and 2 channels.**
Its bottleneck is `Conv3d(kernel=(4,3,3))` after four pooling stages, which collapses to one
voxel *only* at 64 × 48 × 48 — not the 96 × 48 × 48 tree shipped in `data/`. The second
channel is `add_average` = mean of **frames ≥ 35**, not the whole-series mean.
`adapter.infer_input_spec()` recovers this from the weights. Sanity value on scan AA1:
r = 0.98 against the arterial curve, peak height ratio **0.79** — the spill-out signature the
thesis is about.

**3.3 — The group's preprocessing is a PURE CROP at native resolution.**
96 × 48 × 48 voxels cut from the 128 × 92 × 92 reconstruction, **no resampling, no smoothing,
no rotation**, times a global 0.9937. Arithmetic tell: 96 × 0.59675 = 57.3 mm,
48 × 0.5 = 24 mm. Recovering the per-scan window reproduces the distributed `IMG_*.pkl` at
**r = 1.00000** (9/10 scans tested; AB3 gives 0.97, its distributed input evidently came from
a different reconstruction). An earlier crop-and-resample model only reached r ≈ 0.96 and was
abandoned — do not reintroduce resampling.

**3.4 — The crop window is fixed on UNCORRECTED data and reused for every condition.**
`01_prepare_data.py` writes `<work>/preprocessing_plan.json`; every later stage reads it.
Recomputed on corrected data, deconvolution would shift the centre of mass and the conditions
would differ by a crop as well as by PVC. **This is the single change that would invalidate
the entire comparison.** Guarded by
`TestPreprocessing::test_corrected_data_gets_the_uncorrected_window`.

**3.5 — n = 70, not 94.**
102 exports; 32 removed by the group's `ignore_ids` (including **all ten** FDOPA and PSMA
scans); 5 Sherbrooke scans have no arterial curve. The remaining 70 are all [¹⁸F]FDG.
The thesis Abstract still says 94; Methods has been corrected.

Verified dataset facts (use these, don't re-measure): reconstruction 128 × 92 × 92 at
0.5 × 0.5 × 0.59675 mm, SUV units; 42 frames = 1×30 s + 24×5 s + 9×20 s + 8×300 s = 2730 s;
PSF FWHM 0.864 / 0.874 / 0.994 mm = 1.73 / 1.75 / 1.67 voxels. Five Sherbrooke scans use a
different matrix (128 × 120 × 120) but are all excluded.

---

## 4. Verification status — be honest about this

| Stage | Real data | Synthetic end-to-end |
|---|---|---|
| 00 inventory | ✅ 70/102 usable | ✅ |
| 01 convert + preprocessing | ✅ r = 1.00000 | ✅ r = 0.99999 |
| 02 PVC | ❌ PETPVC not installed | ✅ (numpy backend) |
| 03 network inputs | ✅ | ✅ |
| 04 pretrained inference | ✅ | ✅ |
| 05 retrain | ❌ | ✅ |
| 06 evaluate | ❌ | ✅ |
| 07 motion | ❌ | ✅ (screen only) |
| 08 report | ❌ | ✅ |

**No stage has ever run on the full 70-scan real dataset, and PETPVC is not installed on the
owner's machine.** Installing it (`conda install -c conda-forge petpvc`) and confirming CUDA
are the first two practical steps.

Reproduce the synthetic pass in minutes:

```bash
python tests/make_synthetic_dataset.py --out /tmp/synthetic --n-scans 8 \
       --write-config configs/synthetic.yaml
python scripts/run_all.py --config configs/synthetic.yaml --from 00 --to 03
```

---

## 5. Finish the GUI

`gui/pvc_dlif_gui/` already runs (`python gui/run_gui.py`). What exists:

| Module | State |
|---|---|
| `app.py` | ✅ window, ttk theming, three tabs, status bar reporting PETPVC/torch availability |
| `widgets.py` | ✅ `ToolTip`, `LogPane`, `PathPicker`, `LabelledEntry`, `TableView` |
| `jobs.py` | ✅ `ThreadJob` and `ProcessJob` — queue-based, no widget is touched off the Tk thread |
| `tab_pvc.py` | ⚠️ works on arbitrary NIfTI; parameters, sweep, sidecars, progress |
| `tab_pipeline.py` | ⚠️ stage checkboxes, config check, sequential subprocess run with live output |
| `tab_analysis.py` | ⚠️ loads result tables, summary panel, five plots, save-figure |

The threading model is already correct — **do not rewrite it.** Workers push strings onto a
`queue.Queue`; the Tk thread drains it on an `after()` poll. Keep that contract.

### 5.1 What is missing — build these

**PVC tab**
- A *Thesis dataset* input mode beside *Pick files*: a checkbox list of the 70 usable scan IDs
  read from `<work>/manifest.json`, with select-all / none.
- Route the batch through `pvc_dlif.pvc.runner.run_batch()` instead of calling
  `backend.correct_volume` per frame as the skeleton does. `run_batch` already gives
  resumability and the per-frame diagnostics; the current code throws both away.
- A results `Treeview`: one row per scan × method × k with `noise_amplification` and
  `peak_recovery` from `FrameDiagnostics`, plus CSV export.
- Progress at *scan i of N, frame j of 42*, and a Stop that checks a `threading.Event`
  **between scans** so a run interrupts cleanly rather than being killed.

**Pipeline tab**
- Stage status computed **from the filesystem**, not held in memory, so it survives crashes:
  count `native/*.nii.gz`, `pvc/<tag>/*.nii.gz`, `dlif_inputs/<tag>/`,
  `models/<cond>/fold_XX/run_YY/summary.json`, `predictions/*.parquet`, `results/*.csv`,
  `report/*`, and show "43 / 70 done" per stage with a status light.
- Two results must be impossible to miss — a coloured banner, not a log line:
  after stage 00, **n usable = 70** with the exclusion breakdown; after stage 01,
  **`agreement_with_distributed_median_r` = 1.0**, naming any scan below 0.999.
- Stage 05 runs for days. Record `{pid, stage, log_path}` in `<work>/gui_state.json` and
  re-attach on restart by tailing the log, rather than orphaning or restarting the job.

**Analysis tab**
- Reuse `pvc_dlif.report.figures` for the plots rather than the ad-hoc matplotlib in the
  skeleton — those functions already produce the thesis figures, so the GUI and the thesis
  cannot disagree. Add: recovery vs noise amplification per frame coloured by count level,
  and RMSE against iteration count.
- Sortable `Treeview` columns and a CSV export per table.
- A per-scan browser: pick a scan ID, see its curves under every condition plus its PVC
  diagnostics. This is what he will use to find interesting cases for the Discussion.
- An **Export for thesis** button copying `report/*.pdf` and the `.tex` fragments to a chosen
  folder, ready for Overleaf.

**Setup tab (does not exist yet — build it)**
- The four paths, editable with Browse buttons, saved back to `configs/thesis.yaml`.
- The few parameters worth changing from the GUI: `pvc.iterations_primary`, `pvc.methods`,
  `dlif.cv.n_folds` / `n_runs`, `dlif.train.device`, `motion.affected_ids`. Everything else
  stays in the YAML.
- Validate on save by round-tripping through `pvc_dlif.config.load_config`, which already
  raises informative errors (for example, a condition asking for an iteration count stage 02
  never produced).
- **Preflight check** with ✅/❌ per item and the exact fix for each ❌: Python dependencies,
  `petpvc` on PATH, torch + CUDA, all four paths readable, free disk against an estimate.
  He currently has no PETPVC — this is how he finds out.

**Packaging**
- `run_gui.bat` in the repository root, so it is a double-click on Windows.

---

## 6. Then: the real full run

Drive a complete run and fix what breaks. Known risks:

- **Stage 02 scale.** 70 scans × 42 frames × 2 methods. `pvc.workers` parallelises over
  frames; tune it. `--sweep` triples the work and is needed for the sensitivity analysis.
- **Stage 05 cost.** 3 retrained conditions × 10 folds × 10 runs = **300 trainings**. Not
  feasible on CPU. Confirm CUDA works; if it does not, that is a blocker to raise, not to
  route around. `--runs 3` gives a directional answer sooner; more repeats only sharpen the
  variance term.
- **Disk.** Native NIfTI is ~26 MB/scan compressed; PVC output multiplies by (methods × k).
  Estimate before starting and show it in preflight.

Expect scale problems — memory, runtime, disk — rather than logic problems.

---

## 7. Repository hygiene

The repository is meant to be publishable as a tool, not only to produce this thesis.

- The dataset and the pretrained weights are **not** the owner's to redistribute. `docs/data.md`
  documents the expected layout instead. Never commit data, weights, `work/`, or `*.pkl`.
- Keep `configs/example.yaml` free of real paths; `configs/thesis.yaml` is gitignored.
- `pyproject.toml` exposes `pvc-dlif-gui`; keep that entry point working.
- Keep `python -m pytest tests/` green throughout, and add a test for anything you fix during
  the real run. The suite is the regression net for all of section 3.

---

## 8. Guardrails — changes that would invalidate the science

Do not make these without flagging them explicitly:

- Recomputing the crop window on corrected data (see 3.4).
- Deriving folds from anything other than scan IDs + seed — the paired design requires the
  same scan in the same test fold under every condition.
- Choosing the iteration count by best test performance. `k = 15` is primary; the grid is a
  **sensitivity analysis**, not a search.
- Averaging the 10 repeats before the statistical test (each scan must contribute one paired
  observation), or treating repeats as independent samples.
- Switching `reference_condition` away from `baseline_retrained`. The pretrained conditions
  measure distribution shift, not the main hypothesis.
- Letting the GUI default to the built-in numpy PVC backend for real results. It exists for
  development; PETPVC is what the phantom validation used.

---

## 9. Three decisions that are the owner's, not yours

Surface these; do not resolve them.

1. **Which scans are motion-affected.** Stage 07 `--screen` ranks candidates by cyclic
   centre-of-mass displacement. Confirmation is a judgement call with Christian Salomonsen.
2. **Whether n stays at 70.** Including FDOPA/PSMA gives 80; everything with complete data
   gives 97. Changes what can be claimed about tracer generalisation. Question for Samuel
   Kuttner.
3. **Whether the pretrained arm is reportable** as more than a robustness check.

---

## 10. Thesis-side loose ends

- `methodology.tex` (written, compiles clean, 13 pages) is a drop-in replacement for
  `\chapter{Methodology}`. It fixes a LaTeX error in the existing PSF table
  (`\textbf{FWHM}_x` — math outside math mode, breaks the build) and corrects `-p VC` → `-p RVC`.
- **Abstract still says "approximately 94 scans"** → change to 70.
- Results / Discussion / Conclusion are placeholders. Once a real run completes they can be
  drafted from `results/summary.json`, `comparisons.csv`, `bias_variance.csv`,
  `error_by_time_bin.csv` and `failure_modes.csv`.
- Two references to add if cited: Holm (1979) for the step-down correction, Kerby (2014) for
  the rank-biserial effect size.

---

## 11. Acceptance criteria

Done when the owner can:

1. Double-click `run_gui.bat` and see a preflight page telling him exactly what to install.
2. Select several scans, set method / iterations / FWHM, click Run, and watch per-frame
   progress without the window freezing — and stop it cleanly if he changes his mind.
3. Click through stages 00–08, close his laptop mid-run, come back, and see it still going.
4. Read n = 70 and r = 1.0 as unmissable confirmations, not buried log lines.
5. Open Analysis, read the headline numbers, look at every plot, and export the figures and
   the LaTeX tables straight into Overleaf.
6. Re-run everything from scratch on another machine and get identical numbers.
7. Open any file you wrote and follow it without asking what it does.
