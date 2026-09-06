#!/usr/bin/env bash
#SBATCH --job-name=pvc-dlif-05
#SBATCH --array=0-29              # conditions x folds - 1  (3 x 10 = 30 jobs)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00         # one fold = 10 runs x 1000 epochs; adjust after a timed run
#SBATCH --output=run/logs/%x_%a.out
#
# SLURM array over (condition, fold).  Submit from the bundle root:
#
#     mkdir -p run/logs
#     sbatch run/stage05_slurm.sh
#
# Array index -> (condition, fold) so all folds of one condition sit next to
# each other in the queue.  --array must equal (#conditions x #folds) - 1; the
# script computes the mapping from run/conditions.txt and run/n_folds.txt so
# nothing else needs editing when either changes.
set -euo pipefail
cd "$(dirname "$0")/.."

mapfile -t CONDITIONS < run/conditions.txt
N_FOLDS=$(cat run/n_folds.txt)
INDEX="${SLURM_ARRAY_TASK_ID:-0}"
CONDITION="${CONDITIONS[$((INDEX / N_FOLDS))]}"
FOLD=$((INDEX % N_FOLDS + 1))

echo "job $INDEX -> condition $CONDITION fold $FOLD on $(hostname)"
# Activate your environment here, e.g.:
#   module load Python/3.11 CUDA
#   source ~/pvc-dlif-env/bin/activate
bash run/stage05_one_fold.sh "$CONDITION" "$FOLD"
