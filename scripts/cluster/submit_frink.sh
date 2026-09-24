#!/usr/bin/env bash
# Generate and submit one frink job per (condition, fold) of stage 05.
#
# Run from your own machine - the one that has `frink` - with the bundle
# already copied to the cluster's storage.  This script only reads
# run/conditions.txt and run/n_folds.txt locally; the training itself reads
# everything from the copy on the cluster.
#
#   bash run/submit_frink.sh --dry-run          # write the YAML, submit nothing
#   bash run/submit_frink.sh --runs 1           # one run per fold: a complete
#                                               # paired result, ten times sooner
#   bash run/submit_frink.sh                    # the config's full n_runs
#   bash run/submit_frink.sh --gpu-type rtx-a6000
#   bash run/submit_frink.sh --node flanders      # one named machine
#
# --node pins by hostname, --gpu-type by the cluster's GPU label.  Use
# --node when the machines differ in something the label does not capture:
# speed (2.1 s/epoch on flanders against ~10 s on a 2080 Ti, worse again on
# the 1080 Ti nodes where AMP buys nothing), or a driver fault on one node
# that shows up as an NVML version mismatch.
#
# Why one file per job: `frink run` takes a file path, not stdin, and it
# deletes any job with the same name before scheduling a new one.  Piping a
# generated manifest per job, the way `kubectl apply -f -` allows, would submit
# thirty jobs that each replace the last.
#
# Start with --runs 1.  Thirty jobs of one run each give every condition on
# every fold - a complete paired comparison you can put through stage 06 - in a
# tenth of the time.  Submitting again without --runs then fills in runs 2..10
# around them, because a finished run is skipped on resume.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
BUNDLE_LOCAL="$(cd "$HERE/.." && pwd)"

# Where the bundle lives *on the cluster*, and what image to run.
BUNDLE_REMOTE="${BUNDLE_REMOTE:-/storage/stage05_bundle}"
IMAGE="${IMAGE:-pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime}"
OUT="${OUT:-$BUNDLE_LOCAL/run/jobs}"

# A nodeSelector block, or nothing.  Written by both the queue path and the
# one-job-per-fold path, from here rather than from two copies.
emit_node_selector() {
    local file="$1" indent="      "
    if [ -z "$GPU_TYPE" ] && [ -z "$NODE" ]; then return 0; fi
    printf '%snodeSelector:\n' "$indent" >> "$file"
    if [ -n "$NODE" ]; then
        printf '%s  kubernetes.io/hostname: %s\n' "$indent" "$NODE" >> "$file"
    fi
    if [ -n "$GPU_TYPE" ]; then
        printf '%s  springfield.uit.no/gpu-type: %s\n' "$indent" "$GPU_TYPE" >> "$file"
    fi
}

DRY_RUN=0
GPU_TYPE=""
NODE=""
EXTRA=""
QUEUE=""

while [ $# -gt 0 ]; do
    case "$1" in
        --queue)     QUEUE="${2:?how many GPUs to hold at once}"; shift 2 ;;
        --dry-run)   DRY_RUN=1; shift ;;
        --gpu-type)  GPU_TYPE="${2:?gpu type, e.g. rtx-a6000}"; shift 2 ;;
        --node)      NODE="${2:?node hostname, e.g. flanders}"; shift 2 ;;
        --out)       OUT="${2:?output directory}"; shift 2 ;;
        --image)     IMAGE="${2:?container image}"; shift 2 ;;
        --bundle)    BUNDLE_REMOTE="${2:?bundle path on the cluster}"; shift 2 ;;
        --runs)      EXTRA="$EXTRA --runs ${2:?number of runs}"; shift 2 ;;
        --epochs)    EXTRA="$EXTRA --epochs ${2:?number of epochs}"; shift 2 ;;
        --conditions) CONDITIONS_OVERRIDE="${2:?space-separated condition names}"; shift 2 ;;
        -h|--help)   sed -n '2,30p' "$0"; exit 0 ;;
        *)           echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

TEMPLATE="$HERE/stage05_k8s.yaml"
[ -f "$TEMPLATE" ] || { echo "missing $TEMPLATE" >&2; exit 1; }

# tr -d '\r': a bundle packed on Windows carries CRLF in these two files, and
# "10\r" is not a number to seq or to $(( )).
CONDITIONS="${CONDITIONS_OVERRIDE:-$(tr -d '\r' < "$HERE/conditions.txt")}"
N_FOLDS="$(tr -d '\r' < "$HERE/n_folds.txt" | tr -d '[:space:]')"

mkdir -p "$OUT"
count=0

# --queue N: one indexed job that holds N GPUs and refills as pieces finish,
# instead of every (condition, fold) queueing for a GPU at once.
if [ -n "$QUEUE" ]; then
    QUEUE_TEMPLATE="$HERE/stage05_k8s_queue.yaml"
    [ -f "$QUEUE_TEMPLATE" ] || { echo "missing $QUEUE_TEMPLATE" >&2; exit 1; }

    n_conditions="$(printf '%s\n' $CONDITIONS | wc -l | tr -d ' ')"
    completions=$((n_conditions * N_FOLDS))
    name="pvc-dlif-05"
    file="$OUT/$name.yaml"

    sed -e "s|__NAME__|$name|g" \
        -e "s|__COMPLETIONS__|$completions|g" \
        -e "s|__PARALLELISM__|$QUEUE|g" \
        -e "s|__EXTRA__|$EXTRA|g" \
        -e "s|__IMAGE__|$IMAGE|g" \
        -e "s|__BUNDLE__|$BUNDLE_REMOTE|g" \
        "$QUEUE_TEMPLATE" > "$file"

    emit_node_selector "$file"

    # The pods read run/conditions.txt in this order; the index maps into it.
    printf '%s\n' $CONDITIONS > "$HERE/conditions.txt"

    echo "queue: $completions pieces of work, $QUEUE at a time  ($file)"
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "nothing submitted (--dry-run)"
    else
        frink run "$file"
        echo "watch with:  frink ls    |    frink logs $name"
    fi
    exit 0
fi

for condition in $CONDITIONS; do
    # Kubernetes names are RFC 1123: lowercase alphanumeric and '-' only.
    # rl_retrained would be rejected outright.
    safe="$(printf '%s' "$condition" | tr '_' '-' | tr '[:upper:]' '[:lower:]')"
    for fold in $(seq 1 "$N_FOLDS"); do
        name="$(printf 'pvc-dlif-05-%s-f%02d' "$safe" "$fold")"
        file="$OUT/$name.yaml"

        sed -e "s|__NAME__|$name|g" \
            -e "s|__COND__|$condition|g" \
            -e "s|__FOLD__|$fold|g" \
            -e "s|__EXTRA__|$EXTRA|g" \
            -e "s|__IMAGE__|$IMAGE|g" \
            -e "s|__BUNDLE__|$BUNDLE_REMOTE|g" \
            "$TEMPLATE" > "$file"

        # Sits under spec.template.spec, level with restartPolicy.
        emit_node_selector "$file"

        count=$((count + 1))
        if [ "$DRY_RUN" -eq 1 ]; then
            echo "would submit $name  ($file)"
        else
            echo "submitting $name"
            frink run "$file"
        fi
    done
done

echo
echo "$count jobs written to $OUT"
[ "$DRY_RUN" -eq 1 ] && echo "nothing submitted (--dry-run)"
echo "watch with:  frink ls    |    frink logs <name>"
exit 0
