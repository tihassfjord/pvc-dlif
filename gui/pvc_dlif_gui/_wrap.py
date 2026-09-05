"""Run a command and append its exit code to the log.

A detached process cannot be waited on by a GUI that was restarted in the
meantime, so the exit status has to be written somewhere.  This wrapper runs
the real stage with inherited stdout/stderr (both already pointing at the log
file) and prints a marker line the GUI looks for.

    python _wrap.py -- python scripts/02_run_pvc.py --config ...
"""

import subprocess
import sys

MARKER = "__EXIT_CODE__"


def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] == "--":
        argv = argv[1:]
    if not argv:
        print(f"{MARKER} 2", flush=True)
        return 2
    try:
        code = subprocess.call(argv)
    except Exception as exc:                 # noqa: BLE001
        print(f"wrapper: could not start {argv[0]}: {exc}", flush=True)
        code = 127
    print(f"{MARKER} {code}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
