"""Running work off the Tk thread.

Tkinter is not thread-safe, so background work never touches a widget.  Both
helpers here follow the same rule: the worker thread pushes strings onto a
queue, and the Tk thread drains that queue on a timer.

Two kinds of job:

* :class:`ThreadJob`   - a Python function (PVC, loading tables).
* :class:`ProcessJob`  - a ``scripts/NN_*.py`` subprocess, output streamed live.
"""

from __future__ import annotations

import gc
import logging
import queue
import subprocess
import sys
import threading
import traceback
from pathlib import Path
from typing import Callable, Sequence


class _BaseJob:
    """Shared queue plumbing: the worker puts lines in, the UI takes them out."""

    def __init__(self, on_line: Callable[[str], None], on_done: Callable[[bool], None], widget):
        self._queue: queue.Queue = queue.Queue()
        self._on_line = on_line
        self._on_done = on_done
        self._widget = widget            # any widget, only used for .after()
        self._thread: threading.Thread | None = None
        self.cancelled = False

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def cancel(self) -> None:
        """Ask the job to stop. Cooperative - the worker has to check."""
        self.cancelled = True

    def _pump(self) -> None:
        """Drain the queue into the UI, then reschedule until the worker exits."""
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(item, tuple) and item and item[0] == "__done__":
                self._on_done(bool(item[1]))
                return
            self._on_line(str(item))
        self._widget.after(80, self._pump)

    def _start(self, target) -> None:
        # Collect garbage on the Tk thread before the worker starts, so no Tk
        # object (a PhotoImage, say) gets finalised from the wrong thread.
        gc.collect()
        self._thread = threading.Thread(target=target, daemon=True)
        self._thread.start()
        self._widget.after(80, self._pump)


class _QueueHandler(logging.Handler):
    """Forwards log records from the library into the job's queue."""

    def __init__(self, put):
        super().__init__(level=logging.INFO)
        self._put = put

    def emit(self, record: logging.LogRecord) -> None:
        self._put(self.format(record))


class ThreadJob(_BaseJob):
    """Run a Python callable in a thread.

    The callable is given a ``log`` function it can call from the worker
    thread; anything it logs appears in the UI on the next pump.  With
    ``capture_logger`` set, everything the named logger (and its children)
    emits while the job runs is forwarded too - which is how the library's own
    progress messages reach the log pane without the library knowing about
    the GUI.
    """

    def start(self, work: Callable[[Callable[[str], None]], None],
              capture_logger: str | None = None) -> None:
        def runner():
            ok = True
            handler = None
            logger = None
            if capture_logger:
                logger = logging.getLogger(capture_logger)
                handler = _QueueHandler(self._queue.put)
                handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
                logger.addHandler(handler)
            try:
                work(lambda line: self._queue.put(line))
            except Exception as exc:                      # noqa: BLE001 - shown to the user
                ok = False
                self._queue.put(f"ERROR: {exc}")
                for line in traceback.format_exc().splitlines()[-8:]:
                    self._queue.put("  " + line)
            finally:
                if logger is not None and handler is not None:
                    logger.removeHandler(handler)
            self._queue.put(("__done__", ok))

        self._start(runner)


class ProcessJob(_BaseJob):
    """Run a subprocess and stream its combined output line by line."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._proc: subprocess.Popen | None = None

    def start(self, command: Sequence[str], cwd: Path | None = None) -> None:
        def runner():
            ok = True
            try:
                self._queue.put("$ " + " ".join(str(c) for c in command))
                self._proc = subprocess.Popen(
                    [str(c) for c in command],
                    cwd=str(cwd) if cwd else None,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1,
                    # Unbuffered child so progress appears while it runs.
                    env={**_env(), "PYTHONUNBUFFERED": "1"},
                )
                for line in self._proc.stdout:            # type: ignore[union-attr]
                    if self.cancelled:
                        self._proc.terminate()
                        self._queue.put("cancelled")
                        break
                    self._queue.put(line.rstrip())
                code = self._proc.wait()
                ok = code == 0 and not self.cancelled
                self._queue.put(f"exit code {code}")
            except Exception as exc:                      # noqa: BLE001
                ok = False
                self._queue.put(f"ERROR: {exc}")
            self._queue.put(("__done__", ok))

        self._start(runner)

    def cancel(self) -> None:
        super().cancel()
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()


def _env() -> dict:
    import os
    return dict(os.environ)


def python_exe() -> str:
    """The interpreter running the GUI - so stages use the same environment."""
    return sys.executable or "python"
