"""Shared argument parsing and setup for the stage scripts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the package importable when the scripts are run directly from a checkout.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pvc_dlif.config import Config, load_config  # noqa: E402
from pvc_dlif.logging_utils import setup_logging  # noqa: E402


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", default=None, help="path to thesis.yaml")
    parser.add_argument("--limit", type=int, default=None, help="process at most this many scans")
    parser.add_argument("--ids", nargs="*", default=None, help="only these scan IDs")
    parser.add_argument("--dry-run", action="store_true", help="report what would run, change nothing")
    parser.add_argument("--verbose", action="store_true")
    return parser


def start(args, stage: str) -> Config:
    """Load the config and set up logging for one stage."""
    import logging

    config = load_config(args.config)
    config.ensure_work_tree()
    setup_logging(config.dir_logs, stage, logging.DEBUG if args.verbose else logging.INFO)
    return config


def select_ids(candidates, args) -> list[str]:
    """Apply ``--ids`` and ``--limit`` to a list of scan IDs."""
    ids = list(candidates)
    if args.ids:
        wanted = set(args.ids)
        missing = wanted - set(ids)
        if missing:
            raise SystemExit(f"Unknown scan IDs: {sorted(missing)}")
        ids = [i for i in ids if i in wanted]
    if args.limit:
        ids = ids[: args.limit]
    return ids
