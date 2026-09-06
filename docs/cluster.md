# Running stage 05 on a cluster

Stage 05 is 3 conditions × 10 folds × 10 runs × 1000 epochs. The group ran
their own training as one job per fold on a Kubernetes cluster
(`docs/DLIF.yaml` in their repository), and the same split works here: every
(condition, fold) is independent, writes to its own folder, and the resume
logic merges whatever exists. Nothing scientific changes when the folds run in
parallel.

## 1. Pack

On your machine, after stages 00–03 have run:

```bash
python scripts/cluster/pack_stage05.py --out E:/ML4PET/cluster_bundle
```

(or *Pack for cluster…* on the Pipeline tab). This writes a self-contained
folder — this package, the group's model code with its commit recorded, the
arterial curves, the manifest, the three input trees, a config with relative
paths, and job templates. Inputs are downcast to float32, which is what
training uses anyway, so the bundle is roughly 8 GB for 70 scans rather than 16.
`bundle.json` in the folder says exactly what went in.

Copy it over with `rsync -av` or `scp -r`.

## 2. Environment on the cluster

Python ≥ 3.10 with a CUDA build of torch. From the bundle root:

```bash
pip install -e pvc-dlif          # this package; numpy, scipy, pandas, pyyaml, nibabel, pydicom, scikit-learn
python -c "import torch; print(torch.cuda.is_available())"
```

PETPVC is **not** needed on the cluster — stage 02 already ran locally.

## 3. Time one run first

```bash
bash run/stage05_one_fold.sh baseline_retrained 1 --runs 1
```

Watch `work/models/baseline_retrained/fold_01/run_01/progress.json` or the log:
it prints s/epoch and an ETA from the first epochs. Multiply by
`runs × epochs` for the job time limit. One fold is ten runs.

## 4. Submit

**SLURM** — one array job, index → (condition, fold):

```bash
mkdir -p run/logs
sbatch run/stage05_slurm.sh        # edit --time and the module/venv lines first
```

`--array` must be `conditions × folds − 1` (29 for 3 × 10); the script reads
`run/conditions.txt` and `run/n_folds.txt` to map the index.

**Kubernetes** — one job per (condition, fold), the group's own shape:

```bash
for c in $(cat run/conditions.txt); do
  for f in $(seq 1 $(cat run/n_folds.txt)); do
    sed -e "s/__COND__/$c/" -e "s/__FOLD__/$f/" run/stage05_k8s.yaml | kubectl apply -f -
  done
done
```

Edit the image and the volume claim in `run/stage05_k8s.yaml` to match the
cluster. Both templates call `run/stage05_one_fold.sh`, so extra flags
(`--runs 3`, `--epochs 300`) go in one place.

Jobs are resumable: a killed job resubmitted with the same arguments skips the
runs that completed their epochs and redoes the rest.

## 5. Collect

Copy `work/models` back from the cluster into the bundle folder, then:

```bash
python scripts/cluster/collect_stage05.py --from E:/ML4PET/cluster_bundle/work/models
```

(or *Collect from cluster…* on the Pipeline tab). Complete runs are merged
into `<work>/models`; incomplete ones are skipped and listed; the report ends
with exactly which (condition, fold, run) are still missing, so a second
submission can target those. Then run stage 06 locally as usual — it
re-predicts from the checkpoints.

## What to keep in mind

- `folds.json` is written by every job identically (folds come from the scan
  IDs and the seed), so parallel jobs cannot disagree about the partition.
- Each job holds its condition's 70 scans in RAM (~2.6 GB); ask for 16 GB.
- The commit of the group's repository travels in `dlif_repo/COMMIT` and lands
  in the provenance of every run, so the cluster runs are as traceable as local
  ones.
