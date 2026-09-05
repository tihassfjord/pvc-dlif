# Methods

Why the pipeline is built the way it is. This is the reasoning behind the
design, not a description of the code — for the code, read the stage scripts,
which are short and in order.

---

## The question

A deep-learning-derived input function is read out of the images themselves,
from the left ventricular cavity and the great vessels. In a mouse those are
2–3 mm across, and the scanner's PSF has a FWHM just under 1 mm — so the
structures the network depends on are precisely the ones partial volume effects
distort. Activity spills out of the blood pool into surrounding tissue and
spills in from it, and both errors land on the curve.

The measurable signature is visible before any correction is applied. On scan
AA1 the pretrained model reproduces the arterial curve at r = 0.98 with an AUC
ratio of 1.008 — but the peak height ratio is 0.79. The shape is right and the
total is right; the peak, where the blood pool is smallest relative to its
surroundings and the frames are shortest, is underestimated by a fifth. That is
what a correction has to fix, and it is the reason a single aggregate RMSE is
not an adequate outcome measure.

So: **does correcting the images before the network sees them improve the
predicted input function, and where in the curve does the improvement land?**

---

## Design

Fully paired. Every usable scan passes through every condition, and the
conditions differ only by the correction under test. That makes the paired
non-parametric tests the right ones and removes between-animal variability from
the comparison entirely.

| Condition | Correction | Model |
|---|---|---|
| `baseline_pretrained` | none | deployed |
| `rl_pretrained` | Richardson–Lucy | deployed |
| `rvc_pretrained` | Reblurred Van Cittert | deployed |
| `baseline_retrained` | none | retrained (**reference**) |
| `rl_retrained` | Richardson–Lucy | retrained |
| `rvc_retrained` | Reblurred Van Cittert | retrained |

**The reference is the retrained baseline, not the pretrained one.** Both the
reference and the corrected conditions are then models trained on this
pipeline's own preprocessing, so the comparison isolates the correction and
does not depend on reproducing the group's undocumented preprocessing exactly.
The pretrained conditions stay in the design as a robustness check: they
measure how the deployed model behaves under the distribution shift a
correction introduces, which is a different and also useful question.

---

## Correction

Two iterative deconvolution methods from PETPVC, both applied without an
anatomical prior — there is no co-registered MR or CT for these animals, so
region-based methods are unavailable:

- **Richardson–Lucy**, multiplicative, non-negative by construction, the
  standard choice for Poisson data.
- **Reblurred Van Cittert**, additive with a relaxation parameter α, faster to
  converge and noisier.

Both are run with PETPVC's internal stopping criterion disabled (`-s 0`) so the
iteration count set in the config is the count that runs.

**Correction happens at the reconstructed resolution.** The measured
point-source PSF describes that grid; the network input turned out to be a pure
crop of it with no resampling ([`findings.md`](findings.md) §3), so the
correction reaches the model intact rather than being partly low-pass filtered
back out by an interpolation kernel.

**The iteration count is fixed across all frames of a scan.** This is the
choice most worth defending. The frame schedule spans a 60-fold range in
duration — 5 s early, 300 s late — so counting statistics vary by more than an
order of magnitude along the curve, and the number of iterations that is
optimal for a 5 s frame is not the number that is optimal for a 300 s one.
Adapting the count per frame would apply different amounts of resolution
recovery at different times, and would therefore impose a *time-dependent
processing bias on the very curve being measured*. A fixed count is worse
frame by frame and correct as an experiment. The iteration grid (10, 15, 20) is
reported as a sensitivity analysis, never searched for the best result.

---

## Preprocessing

The transformation from reconstruction to network input is a 96 × 48 × 48 crop
at native resolution, multiplied by a global scalar — no resampling, no
smoothing, no reorientation. Recovering the window per scan and applying it
reproduces the distributed inputs at r = 1.00000.

**The window is fitted once, on uncorrected data, and reused for every
condition.** Deconvolution displaces the centre of mass of the activity
distribution slightly. A window recomputed per condition would mean the
conditions differ by a crop as well as by the correction, and the study would
be measuring both. This is the single change that would invalidate the whole
comparison, and it is covered by a test that fails if the corrected data ever
gets its own window.

---

## Outcome measures

No single number. The failure mode this study is about is localised in time, so
the metrics are chosen to see where the error is, not only how large it is:

- **Curve agreement** — RMSE, nRMSE, MAE, R², concordance correlation, Pearson r.
- **Shape** — peak height ratio, peak time, curve FWHM. The peak height ratio is
  the direct measure of the spill-out effect.
- **Area** — total AUC, and AUC split into peak and tail. Kinetic parameters
  depend on the integral, so a method can improve the peak and still not help.
- **Stratified by frame** — the same metrics within time bins (0–1, 1–2.5,
  2.5–5.5, 5.5–40.5 min), which is where a time-dependent effect shows up.
- **Downstream** — Patlak K_i (t\* = 5 min) and two-tissue-compartment
  parameters computed with each predicted input. This is what an input function
  is *for*, so a correction that improves the curve but not the parameters has
  not helped anyone.

### Bias and variance

MSE is decomposed into bias² and variance across repeated training runs:

```
MSE = ( E[prediction] − truth )²  +  Var[prediction]
        \_______ bias² _______/       \__ variance __/
```

Deconvolution should reduce bias (it recovers the underestimated peak) and
increase variance (it amplifies noise). If both move, an aggregate RMSE can
stay flat while the method is doing exactly what it is supposed to. The
variance term needs repeats, which is why the protocol is 10 folds × 10 runs;
for the deterministic pretrained model the variance is zero by construction and
the decomposition reports only bias.

---

## Statistics

Paired throughout, and non-parametric — with n = 70 and metrics that are
bounded or heavy-tailed, normality is not worth assuming:

- **Wilcoxon signed-rank**, two-sided, each condition against the reference.
- **Matched-pairs rank-biserial correlation** as the effect size, because a
  p-value on 70 paired scans says little about whether a difference matters.
- **Percentile bootstrap confidence intervals** (10 000 resamples) on the median
  paired difference.
- **Holm step-down correction** across the family of comparisons, reported
  alongside the unadjusted values.

Reported as effect size with an interval, with the p-value as supporting
information rather than the headline.

---

## Motion

Some scans show a cyclic inter-frame artifact. This is secondary and
exploratory, and it is kept separate for a reason: motion correction resamples,
and resampling smooths — which partly undoes the resolution recovery that is
the object of study. Stage 07 measures that interpolation cost explicitly, and
runs both orderings (correct-then-deconvolve, deconvolve-then-correct) rather
than assuming one is right.

---

## What this design cannot answer

Worth stating plainly, since a reader will ask:

- **Single tracer.** The exclusion list removes every FDOPA and PSMA scan, so
  all 70 are FDG. Nothing here generalises across tracers.
- **Single scanner, single PSF.** The PSF was measured on a static point source
  at the centre of the field of view, and is assumed spatially invariant. Both
  assumptions are approximations; `sensitivity.psf_scale` bounds how much they
  matter, but does not remove them.
- **Marginal sampling.** The PSF spans 1.67–1.75 voxels on the reconstructed
  grid. Below about two voxels per FWHM, deconvolution is working near the
  limit of what the sampling supports, which bounds what *any* voxelwise method
  can recover here regardless of implementation.
- **No anatomical prior.** Region-based correction methods are not available
  for this dataset, so this study speaks to voxelwise deconvolution only.
