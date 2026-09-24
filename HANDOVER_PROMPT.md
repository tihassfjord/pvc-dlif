# Task: finish the pvc-dlif thesis

You are taking over a research repository for a master's thesis (FYS-3941, UiT,
submission December 2026). The owner is Tor-Ivar Hassfjord, a physics student who
reads code comfortably and will maintain everything you write.

**The experiments are finished.** Every stage has been run on the real data, the
complete condition grid is scored, and the thesis has Results, Discussion and
Conclusion chapters written from those numbers. What remains is writing and a
short list of specific fixes.

Read sections 0, 1 and 2 before doing anything. Section 2 is the most important
section in this document: it lists the mistakes that have actually been made on
this project, several of them more than once, and every one of them produced a
confident wrong answer rather than an error message.

---

## 0. Live state — updated 2026-09-24

### Stages

| Stage | State |
|---|---|
| 00 inventory | done — 70 usable of 102 scans |
| 01 convert + window fit | done — 70 of 70 reproduce the distributed inputs at r = 1.000000 |
| 02 PVC (RL, RVC) | done — 70 scans, 2 methods, k ∈ {10,15,20}, PSF ×{0.9,1.0,1.1} |
| 03 network inputs | done |
| 04 pretrained inference | done |
| 05 retrain | done — 3 conditions × 10 folds × 10 runs on Springfield |
| 06 evaluate | done — 21 conditions scored |
| 07 motion | done — whole cohort, both orderings |
| 08 report | done |

Nothing needs to be recomputed. If you find yourself about to launch a training
run or a full stage 02, stop and ask: it is almost certainly not needed, and
stage 05 costs days of GPU time.

### What is left

1. Rewrite the Introduction and Abstract. They still argue the original
   hypothesis — that spill-out biases the predicted input function, and
   correcting it removes the bias. The data say something different and better:
   the governing variable is whether the model and the input representation
   *match*, and most of the gain is a variance reduction rather than a bias
   removal. The Results and Discussion already say this; the front of the thesis
   does not.
2. Three figure improvements listed in section 7.
3. Upload `uit-thesis.cls` to Overleaf. It is not anywhere under `E:\ML4PET`, so
   it has to come from the department template.
4. Push the git commit (see section 1).

That is the whole list. Everything else is prose.

---

## 1. Where everything is

**There are two directories called `pvc-dlif` and only one of them is real.**

| | |
|---|---|
| **The repository — the only one you edit** | `E:\ML4PET\pvc-dlif` |
| A stub copy, two stale files, ignore it | `E:\ML4PET\thesis_work\pvc-dlif` |

Writing to the stub instead of the repository has already happened once. It fails
silently: the pipeline keeps using the old code and the figures come out
unchanged. Before you edit anything, confirm you are in the directory that has a
`.git` folder in it.

| What | Path |
|---|---|
| Generated output tree | `E:\ML4PET\thesis_work` |
| Result tables | `E:\ML4PET\thesis_work\results` |
| Figures | `E:\ML4PET\thesis_work\report` |
| Trained checkpoints (316 files) | `E:\ML4PET\thesis_work\models` |
| Thesis LaTeX | `E:\ML4PET\thesis_work\thesis_draft\thesis_draft.tex` |
| Group's data (AIF / IMG / VOI pickles) | `E:\ML4PET\ML4PET_DLIF\data` |
| Raw DICOM, 102 scans | `E:\ML4PET\ML4PET_DLIF\aif-mice-dicom-pet` |
| Group's private DLIF repo | `E:\ML4PET\DLIF-upstream` |
| Cluster bundle | `E:\ML4PET\stage05_bundle` |
| Backup archive | `E:\OneDrive - UiT Office 365\A_UIT\ML4PET_MASTER\_backup` |

**`configs/thesis.yaml` is gitignored.** It holds local paths, so it is not in
version control — but it also encodes the entire experimental design: every
condition, the arms, the training regime, the evaluation settings. It exists only
on disk and in the backup archive. Do not lose it, and read it before you
form any opinion about how the study is structured. It is heavily commented and
is the best single description of the design that exists.

**Git.** The repository is at `github.com/tihassfjord/pvc-dlif`, branch `main`.
Everything through 2026-09-24 is committed locally as `8522ae9`, but **not
pushed** — the push needs credentials that only the owner's own terminal has. One
command from a normal Windows shell:

```
cd E:\ML4PET\pvc-dlif
git push origin main
```

**Never commit** the mouse data, `DLIFNet.pt`, anything under `thesis_work`, or
any part of `DLIF-upstream`. That last one is S. Kuttner's *private* repository;
it reaches the cluster inside `stage05_bundle`, which must likewise not be copied
anywhere public.

### Stage commands

Every stage is `python scripts/NN_name.py --config configs/thesis.yaml` and takes
`--limit`, `--ids`, `--dry-run`. The two you will actually use:

```
python scripts\06_evaluate.py --config configs\thesis.yaml
python scripts\08_report.py   --config configs\thesis.yaml
python scripts\collect_thesis_figures.py --config configs\thesis.yaml --prune
```

Stage 06 caches predictions per condition, keyed by input tag, source and the
exact list of checkpoint files, so re-running it after adding one condition costs
minutes rather than the ninety it used to. `--repredict` forces a rebuild; you
almost never want it.

`collect_thesis_figures.py` reads the `.tex`, copies every `\includegraphics`
target from `report\` into `thesis_draft\figures\`, and with `--prune` removes
figures the document no longer cites. Run it after every stage 08.

Tests: `python -m pytest tests/` — 130 tests, all green. Keep them that way.

---

## 2. Traps — read this section twice

These are not hypothetical. Each one has produced a confidently wrong statement
about this data.

**2.1 — Paired designs must be read pairwise, never as marginal medians.**
This is the single most expensive mistake available here. The design is fully
paired: every scan contributes one predicted curve under every condition. The
marginal median of a condition can be identical to another condition's while the
paired difference is overwhelming.

Worked example, real numbers. Peak height ratio for the 2024 model, marginal
medians: RL k=10 gives 0.8899, k=15 gives 0.8897, k=20 gives 0.8899. Flat. The
paired difference k=20 minus k=10, same scan against itself, has median −0.00346,
**p = 1.2 × 10⁻⁸**, and is negative in 58 of 70 scans. The effect is real,
consistent and tiny. Reading the marginals would have you report a null.

If you are about to say "the axis is flat", compute the paired differences first.

**2.2 — Aggregate per scan before you take a median.**
The metrics table has one row per (scan, condition, fold, run) — up to 700 rows
per retrained condition. A median over all rows weights scans by how many runs
they happen to have and mixes within-scan variation into a between-scan summary.
Always collapse to one value per scan first, usually by mean over runs.

This changed reported numbers materially once: peak recovery read as 0.949 → 0.954
across all rows, but 0.925 → 0.957 per scan; and a CCC that looked flat was in
fact 0.941 → 0.957 at p = 0.018.

**2.3 — Ratio metrics are right at 1.0, not at zero.**
`auc_ratio` and `peak_height_ratio` are the two. For them, lower is *not* better,
and a condition moving from 0.98 to 1.02 has improved by the same amount as one
moving to 0.94 has worsened. The figure layer knows this — `plot_ratio_agreement`
handles ratios and `plot_paired_metric` handles error metrics — but any new
analysis you write has to be told.

**2.4 — `rl_retrained` *is* the k = 15 condition.**
The naming is not obvious. The main settings carry no suffix: `rl_retrained` is
RL at k = 15 with the measured PSF. The sensitivity variants carry theirs:
`rl_i10_pretrained`, `rl_psf090_pretrained`. Nothing named `rl_i15` exists.

**2.5 — Holm correction is applied within (arm, metric), not across everything.**
The arms answer different questions against different references. Correcting
across all of them made the main hypothesis' p-value depend on how many
sensitivity conditions happened to be run on a different model — it moved from
0.26 to 0.36 purely because PSF conditions were added elsewhere in the study. If
you add a condition, it joins its own arm's family and leaves the others alone.

**2.6 — pyPET's `patlak` takes seconds, not minutes.**
If you compare against the reference implementation (`github.com/Kuttner/pyPET`),
its `patlak` interpolates with `TIME_FRAMES` in whatever unit the time vector
carries and converts to minutes only afterwards. Hand it minutes and you get
2.5-*minute* frames, too few samples for its break-point search, and an
`UnboundLocalError` rather than a diagnostic. Pass seconds. The compartment
models accept either.

**2.7 — Do not count checkpoints with `rglob` without excluding quarantine.**
Superseded runs live in underscore-prefixed directories (`_stale_17fold`,
`_superseded_10fold`). A naive recursive glob descends into them and reports
phantom extra checkpoints. This produced a wrong diagnosis of a "missing folds"
problem that did not exist.

**2.8 — Check which repository you are writing to.** See section 1.

---

## 3. The study design

### The three published papers this work sits on

Read these before forming opinions about the method. PDFs are not in the repo;
the owner has them.

1. **Kuttner et al., Frontiers in Nuclear Medicine 4 (2024)** — the original
   DLIF model. 68 UiT scans, IDIF targets. This is the "deployed" model.
2. **Kuttner et al., EJNMMI Research 16:42 (2026)**, *Variability of the arterial
   input function in small-animal dynamic PET imaging* — the dataset paper. 112
   mice, prospective, one variable varied at a time. Tells you what the
   experimental groups are, how the AIF was measured and calibrated, and what the
   kinetic-modelling conventions are.
3. **Salomonsen et al., EJNMMI Research 16:65 (2026)**, *A robust and versatile
   deep learning model…* — FC-DLIF, the 2026 architecture. 70 FDG scans with
   simultaneously measured AIFs: exactly the 70 this study uses. Gives the
   training protocol this study reproduces.

Facts settled by those papers, so you do not have to ask:

* The 70 scans are **one scan per mouse**. The design is cross-sectional, one
  variable at a time; the age groups are different animals, not a longitudinal
  follow-up. The letter prefix in a scan ID is a batch label, not an animal.
  Plain `KFold` over scan IDs is therefore correct, and is what the group does.
* The measured AIF is **whole blood** from a microfluidic blood counter. Both
  papers convert to plasma before kinetic modelling; this study does not, which
  is a stated limitation.
* The group applies **dispersion correction** to the AIF before kinetic
  modelling. The pickles carry both: `AIF_A_int` (used for DLIF training, by the
  group's own code and by this study) and `AIF_A_int_disp` (the deconvolved one).
* The 10 FDOPA and PSMA scans are the group's own out-of-distribution set, and
  the published result is that the model fails on them. Do not repeat that
  experiment.
* The liver is **not** a kinetic target in the group's pipeline. It is a
  late-time blood surrogate used to build the IDIF.
* The 2026 limitations section names this thesis almost verbatim: *"an
  appropriate investigation of reconstruction parameters, framing scheme, PSF
  modeling, scatter and dead-time corrections, and partial-volume effects, was
  out of scope for the current study."*

### The arms

The study is not one comparison, it is four, and mixing them produces nonsense.
`assemble.condition_arms` builds them; each has its own reference.

| Arm | Model | Reference | Question |
|---|---|---|---|
| `deployed` | DLIFNet (2024), unchanged | `baseline_pretrained` | What does corrected input do to a model trained on uncorrected images? |
| `retrained` | DLIFNet_MAX (2026), trained per representation | `baseline_retrained` | **The main hypothesis.** |
| `fixed_weights` | weights from `baseline_retrained` | `baseline_retrained` | Distribution shift at inference cost only |
| `fixed_weights_rl_retrained`, `…_rvc_retrained` | weights borrowed from those conditions | the condition they came from | The reverse shift |

The deployed and retrained models are **different architectures**, so no
difference between those two arms can separate adaptation from architecture. The
project description asks for a `retrained − pretrained` decomposition; it is not
computed, deliberately, and that belongs in the limitations.

`DLIFNet.pt` is a pickled `nn.Module` from the 2024 lineage: 2 channels,
64 × 48 × 48, 2 203 104 parameters, second channel = mean of frames ≥ 35.
`DLIFNet_MAX` is FC-DLIF: 1 channel, 96 × 48 × 48, 90 124 parameters. Loading one
into the other transfers 2 of 18 tensors. The 64 × 48 × 48 crop for the deployed
model is a centre crop of the 96-slice tree — a working assumption, not confirmed
by the group. It is one of the open questions.

### Findings that must not be re-derived or reverted

**3.1 — The group's preprocessing is a pure crop at native resolution.**
96 × 48 × 48 voxels cut from the 128 × 92 × 92 reconstruction. No resampling, no
smoothing, no rotation, times a global 0.9937. Recovering the per-scan window
reproduces the distributed inputs at r = 1.000000 on all 70. An earlier
crop-and-resample model reached only r ≈ 0.96 and was abandoned.

**3.2 — The crop window is fixed on uncorrected data and reused everywhere.**
`01_prepare_data.py` writes `preprocessing_plan.json`; every later stage reads it.
Recomputed on corrected data, deconvolution would shift the centre of mass and the
conditions would differ by a crop as well as by PVC. **This is the single change
that would invalidate the entire comparison.** Guarded by
`TestPreprocessing::test_corrected_data_gets_the_uncorrected_window`.

**3.3 — n = 70.** 102 exports; 32 removed by the group's `ignore_ids`. Of those,
10 are other tracers (the group's OOD set), 5 are the Sherbrooke cohort
(different site, no CT, no attenuation correction), and 17 are FDG. Twelve of the
17 are almost certainly the withdrawal-rate experiment: their measured AIFs have
median peak 6.79 against 8.00 and FWHM 1.00 min against 0.75 for the 70, which is
exactly the signature the variability paper reports for a low withdrawal rate. A
distorted label is a good reason to exclude. The remaining five (T1, T2, T3, N4,
P1) are unexplained.

**3.4 — PETPVC calls reblurred Van Cittert `VC`, not `RVC`.**
`PETPVC_METHOD_CODE` in `pvc/deconvolution.py` maps the study's names to the
toolbox's. The thesis text was wrong too and has been fixed. Pinned by a test.

**3.5 — Verified dataset facts.** Use these, do not re-measure. Reconstruction
128 × 92 × 92 at 0.5 × 0.5 × 0.59675 mm, SUV. 42 frames = 1×30 s + 24×5 s +
9×20 s + 8×300 s = 2730 s. PSF FWHM 0.864 / 0.874 / 0.994 mm.

---

## 4. What the study found

These numbers are in the thesis. Do not contradict them without recomputing, and
if you do recompute, read section 2 first.

**The main hypothesis holds, weakly but consistently.** Against
`baseline_retrained`, RMSE falls by 0.024 for `rvc_retrained` (95 % CI
[−0.083, −0.009], r = −0.35, p = 0.023) and by 0.020 for `rl_retrained`
(CI [−0.056, +0.004], r = −0.29, p = 0.032). The RL interval crosses zero.

**About two thirds of the gain is variance, not bias.** Squared error
decomposed: baseline 0.569 total / 0.352 bias / 0.217 variance; `rl_retrained`
0.433 / 0.308 / 0.125; `rvc_retrained` 0.431 / 0.322 / 0.109.

**Matching matters more than correcting.** The 2×2: train on original / test on
original 0.580; train original / test RL 0.803; train RL / test RL 0.526; train
RL / test original 0.823. Both mismatched cells are far worse than either matched
one, and the worst cell in the study is a model trained on corrected images shown
uncorrected ones.

**Correcting the input of the deployed model makes it worse, monotonically.**
Every one of the ten conditions is worse than uncorrected, r between 0.91 and
0.94. Peak recovery falls from a median ratio of 0.934 to 0.890 on correction,
then declines further with iteration count (section 2.1). The PSF assumption
matters about three times as much as the iteration count: 0.9× to 1.1× costs
+0.042 RMSE against +0.013 for doubling k.

**PVC-first beats MC-first** by 0.0103 RMSE, p = 2.5 × 10⁻⁶, r = 0.65.
Motion correction alone changes nothing measurable: +0.011, p = 0.445.

**Downstream, Patlak Kᵢ.** Median relative error against the measured AIF:
`baseline_retrained` −5.3 % brain, −1.6 % myocardium; `rvc_retrained` −1.1 % and
−0.1 %. The deployed model runs the other way, −8.8 % to −10.6 % in the brain.
Per-animal absolute error does not improve and the spread does not narrow —
correction shifts the centre of the distribution, it does not tighten it. Say this
explicitly; it is the honest headline.

**A caution that belongs in any downstream discussion.** The reverse-shift
condition produces one of the worst curves in the study (72 % of the peak, nearly
twice the true width) and one of the best group-level Kᵢ biases, −1.4 % in the
brain. Patlak reads an integral, and an integral does not care where in time the
area sits. Never select a correction strategy on a downstream summary statistic.

---

## 5. Kinetic modelling

Rewritten 2026-09-22 to follow pyPET, the group's own library. Three conventions,
all of which change the numbers:

1. **Uniform 2.5 s resampling before fitting.** The framing runs from 5 s frames
   through the bolus to 5 min frames in the tail. Fitted on that grid, least
   squares counts them as one observation each, so the tail — which carries most
   of the area — is under-weighted by two orders of magnitude relative to its
   duration. Resampling supplies frame-duration weighting by construction.
2. **The Patlak break point is fitted per curve**, by regressing two lines either
   side of each candidate split and taking the one with the smallest summed
   residual, searched over the first sixth of the acquisition. It ranges from
   0.2 to 6.7 min in the brain and 0.5 to 7.1 min in the myocardium, so no single
   fixed value serves both.
3. **The two-tissue model is reversible**, k₄ free.

Brain and myocardium only. The liver is excluded for two independent reasons:
the group uses it as a blood surrogate rather than a tissue, and hepatic
FDG-6-phosphate is dephosphorylated by glucose-6-phosphatase and washes out, so it
is not irreversibly trapped and neither Patlak nor a k₄-free model applies.

`tests/test_kinetics_reference.py` checks the implementation against pyPET on the
real curves when both are available, and skips otherwise:

```
git clone https://github.com/Kuttner/pyPET ~/pyPET
PYPET_PATH=~/pyPET DLIF_DATA=E:\ML4PET\ML4PET_DLIF\data pytest tests/test_kinetics_reference.py
```

Agreement is 0.08 % on the Patlak slope and 0.01 % on the rate constants.

**pyPET is GPL-3 and this repository is MIT.** Do not vendor it. The
implementation here is independent and cites pyPET as the reference; keep it that
way, or the whole repository becomes GPL-3.

**Known weakness, already written into the limitations:** 17 % of myocardial
two-tissue fits come to rest on a parameter bound, almost always k₄ at its
ceiling. `curve_fit` calls these converged and they are, but the parameter was set
by the bound and not by the data. The `bounds_hit` flag records it. The individual
myocardial rate constants are reported and not interpreted. No brain fit hits a
bound.

---

## 6. How the owner wants code written

He reads and maintains everything. Match the style already in `src/pvc_dlif/`.

* **Short docstrings on every function and class** — what it does, what the
  parameters mean, what comes back.
* **Comments that say *why*, not what.** A comment explaining a physics or
  statistics choice is worth five explaining syntax. The existing comments carry
  a lot of the project's reasoning; do not strip them.
* **Plain, obvious Python.** No metaclasses, no clever decorators, no dependency
  injection.
* **Keep the GUI thin.** All computation lives in `src/pvc_dlif/`; the GUI
  collects parameters, calls a library function, shows the result.
* No file past a few hundred lines.

In conversation he wants concise, direct, understated prose, plain technically
precise language over academic register. He would rather be told he is wrong than
agreed with. **He has been right and the assistant wrong at least twice on
substantive statistical points in this project** — when he pushes back on a
result, recompute before defending it.

---

## 7. The remaining fixes

**Three figures.**

* `plot_recovery_noise` draws about 2940 points at `alpha=0.7`. It needs a lower
  alpha or a hexbin.
* `iteration_sweep` and `psf_sweep` have independent y-limits, and the caption
  asks the reader to compare their magnitudes. Share the axis.
* `plot_error_by_time_bin` uses unequal categorical bins drawn as a line, which
  implies interpolation between them. A step plot is honest.

**Prose.** Introduction and Abstract, per section 0. The Conclusion repeats
p-values without the confidence-interval caveat the Results carry.

**Orphan files to delete** from `thesis_work\report`: `paired_auc_ratio_*`,
`paired_peak_height_ratio_*`, `crop_agreement.*` (stale and generated by no code
that still exists), `stage06_readout.html`, and
`thesis_work\preprocessing_plan.backup.json`.

---

## 8. Guardrails

Do not do any of these without flagging it explicitly to the owner first.

* Recompute the crop window on corrected data (3.2).
* Derive folds from anything other than scan IDs plus seed. The paired design
  requires the same scan in the same test fold under every condition.
* Choose the iteration count by best test performance. k = 15 is primary; the
  grid is a sensitivity analysis, not a search.
* Average the 10 repeats before the statistical test, or treat repeats as
  independent samples.
* Switch `reference_condition` away from `baseline_retrained`.
* Let the GUI use the built-in numpy PVC backend for real results. It exists for
  development; PETPVC is what the phantom validation used.
* Vendor pyPET (section 5).
* Commit anything from `DLIF-upstream` or `stage05_bundle`.

---

## 9. Open questions — the owner's to resolve, not yours

Surface these; do not answer them.

1. **What exactly is `DLIFNet.pt`?** Two channels, 64 × 48 × 48, second channel =
   mean of frames ≥ 35. No published paper describes that configuration, and the
   2026 paper says its own baseline was the 2024 architecture *retrained on these
   same 70 scans*. So it may be the published 2024 weights or it may be that
   retrained baseline, and the centre crop from 96 to 64 slices is an assumption.
   This sits underneath the entire deployed arm. Asked, unanswered.
2. **The FC-DLIF weights.** Requested from C. Salomonsen; would let the thesis
   separate architecture from training in the deployed-versus-retrained comparison.
3. **Why T1, T2, T3, N4 and P1 are excluded** (3.3). If the reason is not data
   quality, they are the only completely unseen FDG data available — though five
   scans is too few to carry an external validation on their own.

---

## 10. Backup

`E:\OneDrive - UiT Office 365\A_UIT\ML4PET_MASTER\_backup\` holds a dated
archive with the code and history, all 316 checkpoints, every result table and
figure, the thesis source, the ground-truth AIF and VOI curves, and
`DLIFNet.pt` — about 137 MB compressed. Its `MANIFEST.txt` lists what is
deliberately excluded and why, and how to restore.

The 183 GB of derived image volumes under `thesis_work` are *not* backed up.
They are reproducible from the raw data with stages 01–04 and 07, using the
`preprocessing_plan.json` that is in the archive. Without that plan the rerun
would not reproduce these results, which is why it is included.
