# The condition names

Every experiment in this study is a *condition*: one input representation paired
with one set of network weights. The names are built from components, and once
the grammar is clear the whole grid reads off the name.

This file is the reference. The authoritative definitions are in
`configs/thesis.yaml`, which this was generated from.

---

## The grammar

```
[mc_] <stem> [_setting] [_reverse] [_shift] [_mcfirst|_pvcfirst] _<model>
```

| Component | Means |
|---|---|
| `mc_` | the input was motion corrected with FALCON |
| `baseline` | no partial volume correction |
| `rl` | Richardson–Lucy deconvolution |
| `rvc` | reblurred Van Cittert deconvolution |
| *(no setting suffix)* | the main setting: k = 15, measured PSF |
| `_i10`, `_i20` | deconvolution iteration count, instead of 15 |
| `_psf090`, `_psf110` | the assumed PSF FWHM scaled to 0.9× or 1.1× the measured value |
| `_shift` | **borrowed weights.** The network was trained on something else and is only *shown* this input. Isolates the inference-time cost of a distribution shift from the benefit of retraining. |
| `_reverse_shift` | borrowed weights, shift running the other way: trained on corrected, shown uncorrected |
| `_mcfirst`, `_pvcfirst` | which of motion correction and PVC was applied first |
| `_pretrained` | DLIFNet (2024), the published model, weights unchanged |
| `_retrained` | DLIFNet\_MAX (2026), trained here on the representation in the name |

**The one genuine trap.** In `rl_shift_retrained` the `rl` names the *input*: the
weights come from `baseline_retrained` and are shown RL-corrected images. In
`rl_reverse_shift_retrained` the `rl` names the *weights*: they come from
`rl_retrained` and are shown uncorrected images. The prefix always names the
non-baseline half of the pair, but which half that is flips between the two.

**A second thing worth knowing.** `rl_retrained` *is* the k = 15 condition.
Nothing called `rl_i15` exists; the main settings carry no suffix.

---

## Arm 1 — `deployed`: DLIFNet (2024) on each input

Reference: `baseline_pretrained`. The published 2024 weights, unchanged, shown
each representation in turn. This arm carries the sensitivity analyses.

| Condition | Input | Weights |
|---|---|---|
| `baseline_pretrained` | uncorrected | 2024, published |
| `rl_pretrained` | RL, k = 15 | 2024, published |
| `rvc_pretrained` | RVC, k = 15 | 2024, published |
| `rl_i10_pretrained` | RL, k = 10 | 2024, published |
| `rl_i20_pretrained` | RL, k = 20 | 2024, published |
| `rvc_i10_pretrained` | RVC, k = 10 | 2024, published |
| `rvc_i20_pretrained` | RVC, k = 20 | 2024, published |
| `rl_psf090_pretrained` | RL, k = 15, PSF ×0.9 | 2024, published |
| `rl_psf110_pretrained` | RL, k = 15, PSF ×1.1 | 2024, published |
| `rvc_psf090_pretrained` | RVC, k = 15, PSF ×0.9 | 2024, published |
| `rvc_psf110_pretrained` | RVC, k = 15, PSF ×1.1 | 2024, published |

Note that this arm cannot separate PSF accuracy from correction strength: a
wider assumed PSF means a stronger deconvolution, and both axes act as strength
axes here.

## Arm 2 — `retrained`: the main hypothesis

Reference: `baseline_retrained`. Each condition is a model trained from scratch
on its own representation, 10 folds × 10 runs.

| Condition | Trained on | Shown |
|---|---|---|
| `baseline_retrained` | uncorrected | uncorrected |
| `rl_retrained` | RL, k = 15 | RL, k = 15 |
| `rvc_retrained` | RVC, k = 15 | RVC, k = 15 |

Training and inference match in all three. This is the comparison the thesis
turns on, and it is the primary endpoint: its Holm family is these two
comparisons and nothing else.

## Arm 2b — `motion`: a separate family

Reference: `baseline_retrained`, the same unregistered baseline. Registration is
a different intervention from deconvolution, so these are corrected among
themselves. Folded into the arm above, adding one of them would move the main
hypothesis' adjusted p-value without anything about the PVC experiment having
changed.

| Condition | Trained on | Ordering |
|---|---|---|
| `mc_baseline_retrained` | registered, uncorrected | — |
| `mc_rl_retrained` | registered then RL | MC first |
| `mc_pvcfirst_rl_retrained` | RL then registered | PVC first |

The last two are the same pipeline with the corrections swapped. The borrowed
pair below ranks the orderings under a model trained on uncorrected images,
where both double-correct; these two rank them for a model trained on what it
is shown, which is the question that bears on how a pipeline should be built.

The name follows the input tree each reads (`mc_rl_i15`,
`mc_pvcfirst_rl_i15`), so `mc_pvcfirst_rl_retrained` is not the same thing as
the borrowed `mc_rl_pvcfirst_retrained`. That is an unfortunate near-collision;
check `checkpoints_from` if in doubt — a borrowed condition has one.

## Arm 3 — `fixed_weights`: the forward shift

Reference: `baseline_retrained`. The weights of `baseline_retrained`, trained on
uncorrected images, shown other inputs. Nothing is retrained, so any difference
is the cost of the shift alone.

| Condition | Weights trained on | Shown |
|---|---|---|
| `rl_shift_retrained` | uncorrected | RL, k = 15 |
| `rvc_shift_retrained` | uncorrected | RVC, k = 15 |
| `mc_baseline_shift_retrained` | uncorrected | uncorrected + motion correction |
| `mc_rl_mcfirst_retrained` | uncorrected | motion correction, then RL |
| `mc_rl_pvcfirst_retrained` | uncorrected | RL, then motion correction |

`mc_baseline_shift_retrained` is the one that isolates registration itself:
motion correction and nothing else. Comparing the reference against a series
that had been both registered *and* deconvolved would confound the two.

## Arms 4 and 5 — the reverse shift

Each has its own reference: the condition its weights came from. These are what
make the double-correction account testable.

| Condition | Weights trained on | Shown | Reference |
|---|---|---|---|
| `rl_reverse_shift_retrained` | RL, k = 15 | uncorrected | `rl_retrained` |
| `rvc_reverse_shift_retrained` | RVC, k = 15 | uncorrected | `rvc_retrained` |

A model fitted on images where partial volume had already been corrected has
learned to do less of that correction itself. Shown uncorrected images it
under-corrects, which is the prediction these two conditions confirmed.

---

## Defined but never run

None. `mc_baseline_retrained`, `mc_rl_retrained` and
`mc_pvcfirst_rl_retrained` were trained on 2026-09-24, after it was noticed
that inferring the matched case from a borrowed one is the inference this
study elsewhere shows to be wrong: for deconvolution, the mismatched cells of
the 2×2 are far worse than either matched cell.

---

## Which arm a condition belongs to

Read the suffixes:

* ends in `_pretrained` → arm 1, reference `baseline_pretrained`
* ends in `_retrained` with no `shift` and no `mc_` → arm 2, the primary
  family, reference `baseline_retrained`
* starts with `mc_` and has no `checkpoints_from` → the `motion` family, same
  reference, corrected separately
* contains `_shift` but not `_reverse` → arm 3, reference `baseline_retrained`
* contains `_reverse_shift` → its own arm, reference is the condition named in
  the prefix

Holm correction is applied within (arm, metric), never across arms. The arms
answer different questions against different references, so a condition added to
one must not move another's p-value.
