"""Logging and small run-bookkeeping helpers.

Every stage writes a log file into ``<work>/logs`` and a JSON provenance
record next to its output, so a result in the thesis can be traced back to the
exact configuration and code revision that produced it.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

__all__ = ["get_logger", "setup_logging", "provenance", "write_provenance", "timed"]

_CONFIGURED = False
_FORMAT = "%(asctime)s %(levelname)-7s %(name)-28s %(message)s"


def setup_logging(log_dir: Path | None = None, stage: str = "run", level: int = logging.INFO) -> Path | None:
    """Configure root logging once; return the log file path if one is used."""
    global _CONFIGURED
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    log_file: Path | None = None

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{stage}_{stamp}.log"
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(level=level, format=_FORMAT, handlers=handlers, force=True)
    logging.captureWarnings(True)
    _CONFIGURED = True
    return log_file


def get_logger(name: str) -> logging.Logger:
    if not _CONFIGURED:
        logging.basicConfig(level=logging.INFO, format=_FORMAT, stream=sys.stdout)
    return logging.getLogger(name)


def _git_revision(repo: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def provenance(stage: str, config: Any = None, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Build a provenance record describing how an output was produced."""
    here = Path(__file__).resolve().parents[2]
    record: dict[str, Any] = {
        "stage": stage,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "pipeline_git": _git_revision(here),
        "cwd": os.getcwd(),
    }
    if config is not None:
        record["config_source"] = str(getattr(config, "source", "")) or None
        record["config"] = getattr(config, "raw", None)
    if extra:
        record.update(dict(extra))
    return record


def write_provenance(path: Path, stage: str, config: Any = None, **extra: Any) -> Path:
    """Write a ``<name>.provenance.json`` next to an output file or directory."""
    # Never write over the output itself -- always a sibling ".provenance.json".
    target = (
        path if path.name.endswith(".provenance.json")
        else path.with_name(path.name + ".provenance.json")
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(provenance(stage, config, extra), handle, indent=2, default=str)
    return target


@contextmanager
def timed(logger: logging.Logger, what: str) -> Iterator[None]:
    """Log how long a block took."""
    start = time.perf_counter()
    logger.info("%s ...", what)
    try:
        yield
    finally:
        logger.info("%s done in %.1f s", what, time.perf_counter() - start)
