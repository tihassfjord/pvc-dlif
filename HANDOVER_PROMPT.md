# Task: finish the pvc-dlif study

You are taking over a working research repository for a master's thesis (FYS-3941, UiT).
The owner is a physics student who knows Python and reads code comfortably — he is not
afraid of the source, he just doesn't want to type long command lines dozens of times while
producing thesis data.

**The pipeline has now been run on the real data through stage 05.** Stages 00–04 are done
on all 70 usable scans; stage 05 is running on the UiT Springfield GPU cluster. What remains
is stages 06–08 and the writing. Read section 0 for the live state, then sections 2, 3 and 8
before writing code — they exist to stop you re-deriving things that already cost days, and
to stop you silently breaking the science.

---

## 0. Live state — updated 2026-09-10

### Where the study stands

| Stage | State |
|---|---|
| 00 inventory | done — 70 usable of 102 scans |
| 01 convert + window fit | done — **70 of 70 match the distributed inputs exactly** |
| 02 PVC (RL, RVC) | done — 70 scans x 2 methods, native resolution |
| 03 network inputs | done — three trees: `orig`, `rl_i15`, `rvc_i15` |
| 04 pretrained inference | done |
| 05 retrain | **running on Springfield**, see below |
| 06 evaluate | not started |
| 07 motion subset | blocked — needs the affected-scan list from C. Salomonsen |
| 08 report | not started |

### Decisions taken this week

**The training regime is selected in the config, not hardcoded.** `dlif.regime` picks between
`"2024"` (Kuttner et al., Frontiers in Nuclear Medicine 4, 2024: 17 folds, 1 run, 200 epochs,
lr 2e-4, plain MSE) and `"2026"` (EJNMMI Research: 10 folds, 10 runs, 1000 epochs, lr 1e-4,
weighted MSE). The patch is merged into `dlif.cv` and `dlif.train` at load, so every reader
sees resolved values and the provenance record shows which was used.

**The study runs the 2026 regime on `DLIFNet_MAX`.** The reasoning: `DLIFNet` cannot be built
from the upstream repo as it stands — `Encoder` outputs 32 channels and `DLIFNet`'s bottleneck
`Conv3d` expects 128, so only `DLIFNet_MAX` runs. The 2024 architecture exists only inside the
pickled `DLIFNet.pt`, reachable by unpickling and re-initialising the weights. Rather than do
that, the study uses one architecture consistently with the protocol it was tuned for. Awaiting
Samuel's confirmation (asked on Slack).

**The pretrained condition is reported on its own terms, not differenced against the retrained
one.** Since the retrained conditions use a different architecture from `DLIFNet.pt`, the
`retrained − pretrained` decomposition in step 3 of the project description would mix an
architecture difference into the result. The three retrained conditions share an architecture,
so the main hypothesis is unaffected. **Stage 06 must not compute that decomposition** — this
is a change from the project description and belongs in the limitations.

**1000 epochs is right for this model, empirically.** Across the first 85 completed runs:
median best epoch 917 of 1000, IQR 720–974, 62 of 85 peaking after epoch 750, and the last 250
epochs improving on the first 750 in 62 of 85 runs (median +6.1 %). Best-val-loss checkpoint
selection, not last epoch. Median best val loss 0.2374 (IQR 0.195–0.295).

### Two bugs fixed this week — do not reintroduce

**PETPVC calls reblurred Van Cittert `VC`, not `RVC`.** `PETPVC_METHOD_CODE` in
`pvc/deconvolution.py` maps the study's names to the toolbox's. The thesis text was wrong too
and has been corrected.

**The crop window may run off the end of the reconstruction, and the group's own windows do.**
Seven scans needed a 96-slice axial window starting 1–8 slices too late to fit in a 128-slice
volume; the distributed inputs carry exact zeros there. `crop_with_zero_padding` in `data/grid.py`
pads rather than shortening — a shortened window would be resampled back up and the input would
no longer be reconstruction voxels — and `fit_offset_to_reference` widens the search past the
wall when the optimum lands on it. All 70 scans now match at r = 1.000000, up from 63.

### Stage 05 on the cluster

The bundle from `scripts/cluster/pack_stage05.py` is at `~/stage05_bundle` on Springfield
(mounted as `/storage/stage05_bundle` inside a job). Submit with:

```bash
cd /mnt/e/ML4PET/stage05_bundle
bash run/submit_frink.sh --queue 4 --runs 3     # runs 1-3 everywhere first
bash run/submit_frink.sh --queue 4              # then top up to 10
```

`--queue N` writes one Indexed Job that holds N GPUs and refills as pieces finish; the index
maps to (condition, fold). Monitor with `k9s`, or `kubectl get job pvc-dlif-05` — `frink ls`
reports completions wrong for indexed jobs.

Cluster facts learned the hard way:

* **The GPU quota is 2 per namespace** (`requests.nvidia.com/gpu=2`). Parallelism above that is
  harmless — Kubernetes retries — but does nothing until an admin raises it.
* The CPU quota is 4500m and counts `requests`, not `limits`, so the pods request 500m and
  limit 2.
* `backoffLimit` counts failures across all indexes. It is set to 50 because pods get killed
  with `OutOfnvidia.com/gpu` (exit 137) when the scheduler races the device plugin — three of
  those ended an otherwise healthy queue eight pieces in.
* Speed depends entirely on the node: 2.1 s/epoch on flanders (H200), ~10 s on a 2080 Ti,
  3–4x worse on the 1080 Ti nodes where AMP buys nothing. Do not report a single training time.
* Files the cluster shell reads must be written with `newline="\n"`; a bundle packed on Windows
  carried CRLF and `10\r` is not a number to `seq`.
* `pack_stage05.py` downcasts inputs to float32, which rewrites the pickles with the packing
  machine's numpy — so `run/requirements.txt` pins the numpy generation that wrote them.

Bring results home with `collect_stage05.py --from <downloaded models>`; it copies only runs
that finished their epoch budget and reports what is missing.

### Open questions for the supervisors

1. Architecture: `DLIFNet_MAX` with the 1000-epoch protocol, or the 2024 architecture
   re-initialised from the pickle? (Asked on Slack, unanswered.)
2. Which scans carry the cyclic inter-frame motion artifact — C. Salomonsen. Stage 07 is
   blocked without it; `motion.affected_ids` is empty.
3. Is n = 70 the study's n? The project description says "approximately 94 dynamic scans" and
   the thesis abstract still says 94.
4. Does the bias/variance decomposition mean variance over training seeds or over scans? It
   decides whether `n_runs` must be 10.

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
| Group's DLIF repo (clean clone of private github.com/Kuttner/DLIF) | `E:\ML4PET\DLIF-upstream` |
| Old experimented-in copy of the same repo (do not use) | `E:\ML4PET\ML4PET_DLIF\DLIF-main(1)\DLIF-main` |
| All generated output | `E:\ML4PET\thesis_work` (set in the config) |
| Thesis LaTeX | `ML4PET_MASTER\00_MASTER_THESIS\tex\01_Draft\` |
| Old pipeline copy (reference only, do not edit) | `ML4PET_MASTER\thesis_pipeline` |

```
pvc-dlif/
├── configs/thesis.yaml           # single source of truth; example.yaml is the template
├── src/pvc_dlif/                 # library: data/ pvc/ dlif/ eval/ motion/ report/ phantom/
├── scripts/00..08_*.py           # one CLI per stage, + run_all.py
├── gui/pvc_dlif_gui/             # the Tkinter GUI (four tabs, see section 5)
├── tests/                        # ~100 tests, all passing (test_pipeline, test_gui, test_gui_support)
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

**3.1 — The repo holds two published models; `DLIFNet.pt` is not a `DLIFNet_MAX`.**
`DLIFNet.pt` is a pickled `nn.Module` from the Frontiers 2024 lineage: 2-channel encoder
8/16/32/64/128, TCN 256→128→64→32, 2 203 104 parameters, 64 × 48 × 48 input. `DLIFNet_MAX`
in `models.py` is the EJNMMI Research 2026 model (FC-DLIF): 1 channel, 96 × 48 × 48,
**90 124 parameters**. Loading the checkpoint into a `DLIFNet_MAX` transfers 2 of 18 tensors.
→ Pretrained arm: `adapter.load_pretrained_module()`. Retrained arm: `DLIFNet_MAX` from
scratch, with the *published* protocol (1000 epochs, batch 8, Adam 1e-4, no scheduler,
weighted MSE, Poisson noise + flips). `paths.dlif_repo` must be a **clean clone** of the
private `Kuttner/DLIF` repository (`E:\ML4PET\DLIF-upstream`, commit `6526932`); the
`DLIF-main(1)` copy had extra experiments and drifted hyperparameters. Stages 04/05 record
the commit in provenance.

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

> **Superseded by section 0 for the real-data column.** This table records the state
> before the first real run; stages 00-05 have since been run on the real 70-scan
> dataset. Kept because the synthetic column still describes what
> `tests/make_synthetic_dataset.py` exercises, which is how to test a change without
> the real data.

| Stage | Real data | Synthetic, with the group's real DLIF repo |
|---|---|---|
| 00 inventory | ✅ 70/102 usable | ✅ |
| 01 convert + preprocessing | ✅ r = 1.00000 | ✅ r = 0.99999 |
| 02 PVC | ❌ PETPVC not installed | ✅ (numpy backend) |
| 03 network inputs | ✅ | ✅ |
| 04 pretrained inference | ✅ | ✅ real `DLIFNet.pt`, 2 203 104 params, 2 ch, 64×48×48 |
| 05 retrain | ❌ | ✅ real `models.py`, CPU pilot: 1 fold × 2 runs × 3 epochs, checkpoints + summaries written |
| 06 evaluate | ❌ | ✅ predicts from those checkpoints; bias/variance estimable from the repeats |
| 07 motion | ❌ | ✅ (screen only) |
| 08 report | ❌ | ✅ |

The synthetic column for 04–06 was produced with `paths.dlif_repo` pointing at a copy of the
real repository (`src/models`, `src/datahandlers`, `src/training`, the weights). For that run
the config was changed to `dlif_grid.shapes: [[64,48,48]]` and
`retrain_model.input_shape: [64,48,48]`, because the synthetic reference is 64 slices and no
96-slice window exists to cut; the real data needs no such change. Numbers from that run mean
nothing — 3 epochs on 12 fake scans — but every file the next stage reads was written and read.

One bug found and fixed by doing it: `--folds 0` on stage 05 selected no fold (they are
numbered 1..n) and exited 0 having trained nothing. It now exits with an error naming the
valid range.

**No stage has ever run on the full 70-scan real dataset, and PETPVC is not installed on the
owner's machine.** Installing it (`conda install -c conda-forge petpvc`) and confirming CUDA
are the first two practical steps — the Setup tab's preflight lists both.

Stages 06 and 08 have additionally been run on synthetic data with fabricated prediction
tables (to exercise the Analysis tab); those outputs prove the plumbing, nothing else.

Test suite: `python -m pytest tests/` — 102 passed on the reference environment, plus the
tkinter-dependent tests in `tests/test_gui.py` where tkinter is available.

Reproduce the synthetic pass in minutes:

```bash
python tests/make_synthetic_dataset.py --out /tmp/synthetic --n-scans 8 \
       --write-config configs/synthetic.yaml
python scripts/run_all.py --config configs/synthetic.yaml --from 00 --to 03
```

---

## 5. The GUI — built, verified headlessly, never used on real data

`gui/pvc_dlif_gui/`, launched by `run_gui.bat` or `python gui/run_gui.py`. Four tabs.

| Module | What it does | Verified |
|---|---|---|
| `app.py` | window, four tabs, status bar with PETPVC/torch availability; opens on Setup when neither is installed | Xvfb |
| `widgets.py` | `ToolTip`, `LogPane`, `PathPicker`, `LabelledEntry`, sortable+exportable `TableView`, `Banner`, `CheckList` | Xvfb |
| `jobs.py` | `ThreadJob` (queue-pumped, captures `pvc_dlif` log records) and `ProcessJob` | Xvfb |
| `detached.py` | stages as detached subprocesses, `gui_state.json`, log tailing, re-attach, exit code via `_wrap.py` | pytest + Xvfb |
| `tab_setup.py` | four paths + key parameters saved back into the YAML with comments intact (`config_edit.py`), preflight with a fix per failed item (`preflight.py`) | Xvfb |
| `tab_pvc.py` | pick files / thesis-dataset checklist from the manifest; RL/RVC, iterations or sweep, alpha, PSF, backend, workers; routed through `pvc.runner.run_batch` (resumable, diagnostics); Stop at scan boundary; results + per-frame tables | pytest + synthetic |
| `tab_pipeline.py` | nine stages with lights and counts from `status.py`; banners for *n usable* and *median r*; detached runs, queue continues after restart | Xvfb, incl. failure and re-attach |
| `tab_analysis.py` | summary, all tables, six figures via `report.figures` (shared with stage 08 through `report/assemble.py`), per-scan browser, Export for thesis | Xvfb on synthetic results |

Library modules added for it, all tested without tkinter in `tests/test_gui_support.py`:
`pvc_dlif/status.py`, `pvc_dlif/config_edit.py`, `pvc_dlif/preflight.py`,
`pvc_dlif/report/assemble.py`. Stage 08 now calls `assemble` too, so the GUI and the thesis
figures cannot disagree. `run_batch` gained a `should_stop` hook.

**The threading model is settled — do not rewrite it.** Short jobs: worker thread pushes
strings onto a `queue.Queue`, the Tk thread drains it on `after()`. Long stages: detached
`subprocess.Popen` writing to `<work>/logs/gui_<stage>_<stamp>.log`, state in
`<work>/gui_state.json`, the GUI only tails. One subtlety already handled: garbage is
collected on the Tk thread before a worker starts (`jobs._BaseJob._start`) because a
matplotlib toolbar's `PhotoImage` finalised inside a worker thread stalls the event loop.

What *has not* been exercised, because it needs the owner's machine:

- the Windows-specific paths in `detached.py` (`CREATE_NEW_PROCESS_GROUP`, `taskkill`,
  `OpenProcess` liveness) — written to the documented API, not run;
- the PETPVC backend from the GUI (PETPVC is not installed anywhere it has been run);
- stage 05 from the GUI for real (only a failing stage 05 and a fake long stage were used
  to verify chaining, failure handling and re-attach).

If any of these misbehave, that is the first GUI work to do. Otherwise leave the GUI alone
until the real run has produced results.

## 6. Then: the real full run

Drive a complete run and fix what breaks. Known risks:

- **Stage 02 scale.** 70 scans × 42 frames × 2 methods. `pvc.workers` parallelises over
  frames; tune it. `--sweep` triples the work and is needed for the sensitivity analysis.
- **Measure before optimising stage 05.** `scripts/profile_training.py --variants` times
  each phase of a real epoch (load, transfer, augment, forward, backward, validate) with
  proper CUDA synchronisation and A/Bs a few safe settings. Two attempts to speed training
  up by guessing cost the owner a day: one put the data in VRAM and the card silently
  spilled to system memory (60 s/epoch), the other crashed on a generator/device mismatch.
  Both were reverted. Profile first.
- **Stage 05 belongs on a cluster.** `scripts/cluster/` packs a self-contained bundle
  (`pack_stage05.py`), ships SLURM and Kubernetes templates that run one (condition, fold)
  per GPU — the group's own shape — and merges results back (`collect_stage05.py`). See
  `docs/cluster.md`. The owner has cluster access.
- **Stage 05 cost.** 3 retrained conditions × 10 folds × 10 runs = **300 trainings** of
  1000 epochs each (the published protocol; an earlier config said 300). Every scan is held
  in RAM per condition (~2.6 GB for 70) — without that, an epoch was disk-bound at 28 s on
  the owner's machine — and on a GPU the Poisson/flip augmentation runs on the device
  (`train.augment_on_device`), because NumPy Poisson over 9 M voxels per scan is ~0.8 s and
  would keep the GPU idle for 40 s per epoch. Expect a few seconds per epoch, not tens. Each run writes `progress.json` every epoch; the Pipeline tab shows
  the current epoch, s/epoch, ETA and how long ago it last updated, so "slow" and "dead" can
  be told apart. Pilot checkpoints (5 epochs) are recognised as incomplete and retrained;
  stage 06 re-predicts from checkpoints by default rather than reusing a cached table. Not
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
  (`\textbf{FWHM}_x` — math outside math mode, breaks the build). It says `-p VC` for the
  reblurred Van Cittert, which is PETPVC's actual method code; an earlier draft wrongly changed
  it to `-p RVC`, and the backend once sent that too — fixed and pinned by a test.
- **Abstract still says "approximately 94 scans"** → change to 70.
- Results / Discussion / Conclusion are placeholders. Once a real run completes they can be
  drafted from `results/summary.json`, `comparisons.csv`, `bias_variance.csv`,
  `error_by_time_bin.csv` and `failure_modes.csv`.
- Two references to add if cited: Holm (1979) for the step-down correction, Kerby (2014) for
  the rank-biserial effect size.

---

## 11. Acceptance criteria

Done when the owner can (✅ = built and verified headlessly; ⬜ = needs the real machine):

1. ✅ Double-click `run_gui.bat` and see a preflight page telling him exactly what to install.
2. ✅ Select several scans, set method / iterations / FWHM, click Run, and watch per-frame
   progress without the window freezing — and stop it cleanly if he changes his mind.
3. ✅ Click through stages 00–08, close his laptop mid-run, come back, and see it still going
   (verified on Linux; Windows detach path untested).
4. ✅ Read n = 70 and r = 1.0 as unmissable confirmations, not buried log lines.
5. ✅ Open Analysis, read the headline numbers, look at every plot, and export the figures and
   the LaTeX tables straight into Overleaf.
6. ⬜ Re-run everything from scratch on another machine and get identical numbers — needs
   the first real run to have happened.
7. ✅ Open any file you wrote and follow it without asking what it does.
