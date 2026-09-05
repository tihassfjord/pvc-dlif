"""Run the pipeline end to end, in order, stopping at the first failure.

Each stage is resumable, so re-running after a fix picks up where it left off
rather than recomputing everything.

    python scripts/run_all.py                       # everything
    python scripts/run_all.py --from 02 --to 06     # a range of stages
    python scripts/run_all.py --dry-run             # what would run
    python scripts/run_all.py --pilot               # small, fast, for a check

``--pilot`` limits the number of scans, folds and epochs so a complete pass
takes minutes instead of days.  Use it to confirm the wiring before committing
to a full run; its numbers are not results.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# (stage id, script, extra arguments, whether a non-zero exit should stop the run)
STAGES: list[tuple[str, str, list[str], bool]] = [
    ("00", "00_build_manifest.py", [], True),
    ("01", "01_prepare_data.py", [], True),
    ("02", "02_run_pvc.py", [], True),
    ("03", "03_make_dlif_inputs.py", [], True),
    ("04", "04_infer_baseline.py", [], True),
    ("05", "05_retrain.py", [], True),
    ("06", "06_evaluate.py", [], True),
    # The motion subset is exploratory and is skipped cleanly when the affected
    # IDs have not been confirmed yet, so it must not stop the run.
    ("07", "07_motion_subset.py", ["--screen"], False),
    ("08", "08_report.py", [], True),
]

PILOT_ARGS: dict[str, list[str]] = {
    "00": [],
    "01": ["--limit", "8", "--diagnose-scans", "3"],
    "02": ["--limit", "8"],
    "03": ["--limit", "8"],
    "04": ["--limit", "8"],
    "05": ["--limit", "8", "--folds", "1", "--runs", "1", "--epochs", "5"],
    "06": [],
    "07": ["--limit", "8"],
    "08": [],
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--from", dest="start", default="00", help="first stage id")
    parser.add_argument("--to", dest="stop", default="08", help="last stage id")
    parser.add_argument("--skip", nargs="*", default=[], help="stage ids to skip")
    parser.add_argument("--pilot", action="store_true",
                        help="small, fast pass to check the wiring (not results)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    selected = [
        s for s in STAGES
        if args.start <= s[0] <= args.stop and s[0] not in set(args.skip)
    ]
    if not selected:
        print("No stages selected", file=sys.stderr)
        return 1

    results: list[tuple[str, int, float]] = []
    for stage_id, script, extra, fatal in selected:
        command = [sys.executable, str(HERE / script), *extra]
        if args.config:
            command += ["--config", args.config]
        if args.pilot:
            command += PILOT_ARGS.get(stage_id, [])
        if args.dry_run:
            command.append("--dry-run")

        print(f"\n{'=' * 72}\n  stage {stage_id}: {script}\n{'=' * 72}", flush=True)
        started = time.perf_counter()
        code = subprocess.call(command)
        elapsed = time.perf_counter() - started
        results.append((stage_id, code, elapsed))

        if code != 0:
            print(f"\nstage {stage_id} exited with {code} after {elapsed:.1f} s", file=sys.stderr)
            if fatal and not args.continue_on_error:
                break
            print("continuing", file=sys.stderr)

    print(f"\n{'=' * 72}")
    for stage_id, code, elapsed in results:
        status = "ok" if code == 0 else f"exit {code}"
        print(f"  stage {stage_id}: {status:>8}  {elapsed / 60:6.1f} min")
    print("=" * 72)

    return 0 if all(code == 0 for _, code, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())
