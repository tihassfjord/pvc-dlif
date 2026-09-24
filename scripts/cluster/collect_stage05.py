"""Merge stage-05 output copied back from a cluster into the local work tree.

    python scripts/cluster/collect_stage05.py --from E:/ML4PET/cluster_bundle/work/models

Copies ``<condition>/fold_XX/run_YY/`` folders whose run finished (summary.json
present and epochs complete) into ``<work>/models``, never overwriting a local
run that is itself complete, and reports what is still missing so a second
batch of jobs can be submitted for exactly those.  After this, stage 06 runs
locally as usual.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pvc_dlif.config import load_config                      # noqa: E402


def _epochs_trained(run_dir: Path) -> int:
    summary = run_dir / "summary.json"
    if not summary.exists():
        return 0
    try:
        return int(json.loads(summary.read_text(encoding="utf-8")).get("epochs_trained", 0))
    except (json.JSONDecodeError, OSError):
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--from", dest="source", required=True, help="the bundle's work/models folder")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    source = Path(args.source)
    target = config.dir_models
    epochs = int(config.get("dlif.train.epochs", 1000))
    n_folds = int(config.get("dlif.cv.n_folds", 10))
    n_runs = int(config.get("dlif.cv.n_runs", 10))

    copied, kept, incomplete = [], [], []
    for condition_dir in sorted(p for p in source.iterdir() if p.is_dir()):
        for run_dir in sorted(condition_dir.glob("fold_*/run_*")):
            rel = run_dir.relative_to(source)
            trained = _epochs_trained(run_dir)
            if trained < epochs:
                incomplete.append((str(rel), trained))
                continue
            local = target / rel
            if _epochs_trained(local) >= epochs:
                kept.append(str(rel))
                continue
            if not args.dry_run:
                if local.exists():
                    shutil.rmtree(local)
                shutil.copytree(run_dir, local)
            copied.append(str(rel))
        # folds.json and provenance travel with the condition
        for name in ("folds.json", "training.json.provenance.json"):
            if (condition_dir / name).exists() and not args.dry_run:
                (target / condition_dir.name).mkdir(parents=True, exist_ok=True)
                shutil.copy2(condition_dir / name, target / condition_dir.name / name)

    # What is still missing, per condition, so the next submission is exact
    missing: dict[str, list[str]] = {}
    # Borrowed-checkpoint conditions train nothing, so every run would count as
    # missing and the report would never reach zero.  They are stage 06's work.
    for condition in (c for c in config.conditions
                      if c.model == "retrained" and not c.motion and not c.borrows_checkpoints):
        for fold in range(1, n_folds + 1):
            for run in range(1, n_runs + 1):
                if _epochs_trained(target / condition.name / f"fold_{fold:02d}" / f"run_{run:02d}") < epochs:
                    missing.setdefault(condition.name, []).append(f"fold {fold} run {run}")

    report = {
        "copied": len(copied), "already_complete_locally": len(kept),
        "skipped_incomplete_on_cluster": incomplete[:20],
        "still_missing": {k: (v[:8] + ["..."] if len(v) > 8 else v) for k, v in missing.items()},
        "still_missing_count": sum(len(v) for v in missing.values()),
        "dry_run": args.dry_run,
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
