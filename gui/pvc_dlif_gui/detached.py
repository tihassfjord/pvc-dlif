"""Stages that outlive the window.

Stage 05 runs for days.  A thread dies with the GUI, so long stages are
started as *detached* subprocesses that write to their own log file, and the
GUI merely tails that file.  What is running is recorded in
``<work>/gui_state.json``; when the GUI is reopened it reads that, checks
whether the process is still alive, and re-attaches to the log instead of
starting the stage again.

Everything here is plain functions on plain files so it can be tested
without a window.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

STATE_NAME = "gui_state.json"
MARKER = "__EXIT_CODE__"        # must match _wrap.py


# ==================================================================== #
# Process liveness / termination, on both platforms
# ==================================================================== #
def pid_alive(pid: int) -> bool:
    """Whether a process with this PID is still running."""
    if pid <= 0:
        return False
    if sys.platform.startswith("win"):
        import ctypes
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32                       # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # Reap a finished child of this process, otherwise it stays a zombie
    # that os.kill(pid, 0) reports as alive.
    try:
        finished, _ = os.waitpid(pid, os.WNOHANG)
        if finished == pid:
            return False
    except ChildProcessError:
        pass
    return True


def terminate(pid: int) -> None:
    """Stop a detached stage. Windows needs taskkill to take the tree with it."""
    if not pid_alive(pid):
        return
    if sys.platform.startswith("win"):
        subprocess.call(["taskkill", "/PID", str(pid), "/T", "/F"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        import signal
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            os.kill(pid, signal.SIGTERM)


# ==================================================================== #
# State file
# ==================================================================== #
def state_path(work: Path) -> Path:
    return Path(work) / STATE_NAME


def read_state(work: Path) -> dict[str, Any] | None:
    path = state_path(work)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def write_state(work: Path, state: dict[str, Any] | None) -> None:
    path = state_path(work)
    path.parent.mkdir(parents=True, exist_ok=True)
    if state is None:
        path.unlink(missing_ok=True)
    else:
        path.write_text(json.dumps(state, indent=2), encoding="utf-8")


# ==================================================================== #
# Launching
# ==================================================================== #
def launch(command: Sequence[str], work: Path, stage_id: str, cwd: Path | None = None,
           queue: Sequence[str] = ()) -> dict[str, Any]:
    """Start a stage detached from this process and record it in the state file.

    ``queue`` is the list of stage IDs to run after this one, kept in the
    state so a reopened GUI can carry on where the closed one left off.
    """
    logs = Path(work) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs / f"gui_{stage_id}_{stamp}.log"

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    popen_kwargs: dict[str, Any] = dict(
        cwd=str(cwd) if cwd else None, env=env,
        stdout=open(log_path, "w", encoding="utf-8"), stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
    )
    if sys.platform.startswith("win"):
        # DETACHED_PROCESS: the child gets no console at all, so closing the
        # window run_gui.bat opened (or the GUI) cannot take it down with it.
        # A new process group so Ctrl-C in that console is not delivered here.
        DETACHED_PROCESS = 0x00000008
        popen_kwargs["creationflags"] = (subprocess.CREATE_NEW_PROCESS_GROUP   # type: ignore[attr-defined]
                                         | DETACHED_PROCESS)
        popen_kwargs["close_fds"] = True
    else:
        popen_kwargs["start_new_session"] = True

    # Wrapped so the exit code lands in the log (see _wrap.py).
    wrapper = Path(__file__).resolve().parent / "_wrap.py"
    full = [sys.executable, "-u", str(wrapper), "--", *[str(c) for c in command]]
    proc = subprocess.Popen(full, **popen_kwargs)
    state = {
        "pid": proc.pid,
        "stage": stage_id,
        "command": [str(c) for c in command],
        "log_path": str(log_path),
        "started": datetime.now().isoformat(timespec="seconds"),
        "queue": list(queue),
        "queue_command_template": None,
    }
    write_state(work, state)
    return state


class LogTail:
    """Incremental reader for a log file another process is writing."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._offset = 0

    def read_new(self) -> list[str]:
        """Lines appended since the last call."""
        if not self.path.exists():
            return []
        with open(self.path, "r", encoding="utf-8", errors="replace") as handle:
            handle.seek(self._offset)
            chunk = handle.read()
            self._offset = handle.tell()
        return chunk.splitlines()


def exit_code_from_log(log_path: Path, pid: int, timeout_s: float = 2.0) -> int | None:
    """Exit status of a detached stage, read from the marker the wrapper wrote.

    Waits briefly for the process to disappear so the last lines are flushed.
    Returns None when no marker is present (log missing, or the process was
    killed before the wrapper could write it).
    """
    deadline = time.time() + timeout_s
    while pid_alive(pid) and time.time() < deadline:
        time.sleep(0.1)
    path = Path(log_path)
    if not path.exists():
        return None
    tail = path.read_text(encoding="utf-8", errors="replace")[-2000:]
    for line in reversed(tail.splitlines()):
        if line.startswith(MARKER):
            try:
                return int(line.split()[1])
            except (IndexError, ValueError):
                return None
    return None
