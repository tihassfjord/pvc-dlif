#!/usr/bin/env bash
# How far along is stage 05 on the cluster, and how fast is it going.
#
#   bash run/status.sh                 # everything
#   bash run/status.sh --watch         # refresh every 60 s
#
# Reads two independent sources, because they answer different questions and
# either one alone can mislead:
#
#   summary.json files   what has actually finished and is safe to score.
#                        The only number that cannot be optimistic.
#   progress.json files  where each running piece is right now - epoch, loss,
#                        seconds per epoch, ETA.  Written every epoch by the
#                        training loop, so a stale timestamp means a pod died
#                        without the job noticing.
#
# Both live on the cluster's storage, so this works from anywhere with ssh,
# with or without kubectl.
set -euo pipefail

REMOTE="${REMOTE:-springfield}"
BUNDLE="${BUNDLE:-stage05_bundle}"
JOB="${JOB:-pvc-dlif-05}"
WATCH=0

while [ $# -gt 0 ]; do
    case "$1" in
        --watch)  WATCH=1; shift ;;
        --remote) REMOTE="${2:?ssh host}"; shift 2 ;;
        --bundle) BUNDLE="${2:?bundle directory on the remote}"; shift 2 ;;
        --job)    JOB="${2:?kubernetes job name}"; shift 2 ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

report() {
    echo "=============================================================="
    date '+%Y-%m-%d %H:%M'
    echo

    if command -v kubectl >/dev/null 2>&1; then
        echo "--- job ---"
        kubectl get job "$JOB" 2>/dev/null || echo "(no job $JOB)"
        echo
        echo "--- pods ---"
        kubectl get pods -l "job-name=$JOB" \
            -o custom-columns=NAME:.metadata.name,STATUS:.status.phase,NODE:.spec.nodeName,AGE:.metadata.creationTimestamp \
            2>/dev/null || true
        echo
    fi

    ssh "$REMOTE" "cd $BUNDLE/work/models 2>/dev/null || exit 0; \
        printf '%s\n' '--- finished runs ---'; \
        for c in */; do \
            n=\$(find \"\$c\" -name summary.json 2>/dev/null | wc -l); \
            printf '%-24s %3d\n' \"\${c%/}\" \"\$n\"; \
        done; \
        total=\$(find . -name summary.json 2>/dev/null | wc -l); \
        printf '%-24s %3d\n' TOTAL \"\$total\"; \
        printf '\n%s\n' '--- running now ---'; \
        find . -name progress.json -mmin -30 2>/dev/null | sort | while read -r p; do \
            python3 -c \"
import json,sys,time
p=sys.argv[1]
d=json.load(open(p))
age=time.time()-d.get('updated',0)
where=p.replace('./','').replace('/progress.json','')
print(f\\\"{where:<44} epoch {d['epoch']:>4}/{d['epochs']}  {d['seconds_per_epoch']:5.1f} s/ep  best {d['best_val_loss']:.4f} @ {d['best_epoch']+1}  ETA {d['eta_seconds']/3600:4.1f} h  ({age:.0f}s ago)\\\")
\" \"\$p\" 2>/dev/null; \
        done"
}

if [ "$WATCH" -eq 1 ]; then
    while true; do report; sleep 60; done
else
    report
fi
