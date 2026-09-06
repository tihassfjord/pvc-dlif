#!/usr/bin/env bash
# Train one (condition, fold) of stage 05 inside a bundle made by pack_stage05.py.
#
#   bash run/stage05_one_fold.sh <condition> <fold>
#
# Run from the bundle root (the folder holding configs/cluster.yaml).  Both the
# SLURM and the Kubernetes templates call this, so the two never drift apart.
# Resumable: a run that already finished its epochs is skipped.
set -euo pipefail

CONDITION="${1:?condition name, e.g. rl_retrained}"
FOLD="${2:?fold index, 1-based}"

cd "$(dirname "$0")/.."                   # bundle root
export PYTHONUNBUFFERED=1

python pvc-dlif/scripts/05_retrain.py \
    --config configs/cluster.yaml \
    --conditions "$CONDITION" \
    --folds "$FOLD" \
    --device cuda \
    "${@:3}"                              # extra flags, e.g. --runs 3 --epochs 300
