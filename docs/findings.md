# Findings

Non-obvious things established empirically about this dataset and this model. Each one cost
real investigation and each is load-bearing — read this before changing the data or model
layers, or you will rediscover them the hard way.

---

## 1. The distributed checkpoint does not match the repository's model code

`DLIF-main/src/models/pretrained_weigths/DLIFNet.pt` is a **pickled `nn.Module`**, not a state
dict, and it carries the architecture it was trained with:

| | Checkpoint | Repository's current `models.py` |
|---|---|---|
| Input channels | 2 | 1 |
| Encoder widths | 8 / 16 / 32 / 64 / 128 | 8 / 8 / 16 / 16 / 32 |
| Batch normalisation | present | commented out |
| TCN | 256 → 128 → 64 → 32 | 32 → 16 → 8 → 4 |
| Parameters | 2 203 104 | 90 124 |

Building the architecture from today's `models.py` and loading the weights into it transfers
**2 of 18 parameter tensors**. The remaining 16 keep their random initialisation, so the
"pretrained baseline" would be a mostly untrained network — and it would fail quietly, looking
like a poor result rather than a broken one.

**Consequence.** Load the checkpoint as the module it is:

```python
from pvc_dlif.dlif.adapter import DlifRepo, load_pretrained_module
model, spec, info = load_pretrained_module(DlifRepo(repo_path))
```

`load_pretrained_model()` (the state-dict path) still exists for other checkpoints, and logs an
error when transfer is poor. Do not resolve this by editing the group's repository.

---

## 2. The pretrained model needs 64 × 48 × 48 and two channels

Its bottleneck is `Conv3d(kernel=(4, 3, 3))` with no padding, after four pooling stages. That
collapses to a single voxel only when the input is 64 × 48 × 48 — **not** the 96 × 48 × 48 tree
distributed alongside it. Feeding it 96 slices raises a shape error from inside torch.

The second channel is the `add_average` augmentation: the mean of **frames 35 onwards**, not
the mean of the whole series. Using the whole-series mean changes what the channel means and
shifts every prediction.

`adapter.infer_input_spec()` recovers all of this from the weights, and
`ModelInputSpec.check_shape()` turns a mismatch into a readable error instead of a torch crash.

**Sanity value.** On scan AA1: r = 0.98 against the arterial curve, CCC = 0.97, AUC ratio 1.008,
peak height ratio **0.79** — the peak underestimated by about a fifth, which is the spill-out
signature this study is about.

---

## 3. The preprocessing is a pure crop at native resolution

The transformation from reconstruction to network input involves **no resampling, no smoothing
and no reorientation**. It is a 96 × 48 × 48 voxel window cut from the 128 × 92 × 92
reconstruction, multiplied by a global factor of 0.9937.

The arithmetic gives it away: 96 × 0.59675 mm = 57.3 mm axially, 48 × 0.5 mm = 24 mm in plane.
Recovering the window position per scan and applying it reproduces the distributed
`IMG_*.pkl` at **r = 1.00000**.

An earlier attempt modelled it as crop-and-resample and reached only r ≈ 0.96, with the fitted
crop wandering several voxels between animals. That approach was abandoned. Do not reintroduce
resampling.

**Why this matters beyond tidiness.**

- The network's input voxels *are* reconstruction voxels, so the measured point-source PSF
  applies to them directly, with no interpolation kernel to account for.
- A correction applied at the reconstructed resolution maps into the network input one-to-one.
  Downsampling would low-pass filter it and partly undo the resolution recovery.
- The uncorrected condition reproduces the distributed inputs, so the pretrained model is
  evaluated on exactly the data it was trained on.

Nine of ten scans tested reproduce exactly. AB3 reaches only 0.97 — its distributed input
evidently came from a different reconstruction. Stage 01 names any such scan.

---

## 4. The crop window must come from uncorrected data

Deconvolution slightly displaces the centre of mass of the activity distribution. If the crop
window were derived per condition, the corrected and uncorrected conditions would differ by a
crop *as well as* by the correction — and the study would be measuring both.

Stage 01 therefore derives the window once, from the uncorrected series, writes it to
`<work>/preprocessing_plan.json`, and every later stage reads it back unchanged.

**This is the single change that would invalidate the entire comparison.** It is covered by
`TestPreprocessing::test_corrected_data_gets_the_uncorrected_window`.

---

## 5. The usable dataset is 70 scans, not 94

| | Scans |
|---|---|
| Dynamic exports in the archive | 102 |
| Excluded by the group's `ignore_ids` | −32 |
| No paired arterial curve (Sherbrooke series) | −5 |
| **Usable, fully paired** | **70** |

All 70 are [¹⁸F]FDG. The exclusion list removes **all ten** scans acquired with the alternative
tracers (FDOPA, PSMA), so the analysis set is single-tracer and no claim about tracer
generalisation is available without revisiting that list.

Two naming variants exist for the same animals (`Sherbrooke_A` in the DICOM directory,
`Sherbrooke-A` in the exclusion list); the manifest normalises them, otherwise those scans slip
through the exclusion silently.

---

## 6. Acquisition facts worth not re-measuring

| | |
|---|---|
| Reconstruction | 128 × 92 × 92 voxels |
| Voxel size | 0.5 × 0.5 × 0.59675 mm |
| Field of view | 76.4 × 46 × 46 mm |
| Units | SUV |
| Corrections applied | decay, attenuation, scatter, dead time, randoms, radial, normalisation |
| Frames | 42 |
| Frame schedule | 1 × 30 s, 24 × 5 s, 9 × 20 s, 8 × 300 s |
| Total duration | 2730 s (45.5 min) |
| PSF FWHM | 0.864 / 0.874 / 0.994 mm |
| PSF in voxels | 1.73 / 1.75 / 1.67 |

Frame timing lives in PMOD private DICOM tags `(0055,1001)` start times in seconds,
`(0055,1004)` durations in milliseconds, `(0055,1005)` per-frame rescale slopes — not in the
standard timing attributes.

Two consequences. The 60-fold range in frame duration means counting statistics vary by more
than an order of magnitude along the curve, so the recovery–noise trade-off cannot be assumed
uniform — hence the frame-stratified analysis. And the PSF spans fewer than two voxels on every
axis, which is marginal sampling: it bounds what any voxelwise method can recover, and it is
why correction happens on the reconstructed grid rather than anything coarser.

Five Sherbrooke scans use a different matrix (128 × 120 × 120). They are excluded anyway, but
the manifest checks for this because a single crop fitted across mixed matrix sizes would cut
the wrong region out of the minority.
