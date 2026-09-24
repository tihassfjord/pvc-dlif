"""Certify an existing prediction table so stage 06 can reuse it.

Stage 06 keeps a per-condition cache of retrained predictions, and reuses a
condition only when a stored signature -- its input tree, the condition its
weights came from, and the exact set of checkpoint files behind it -- still
matches what is on disk.  A table written before that mechanism existed has no
signature, so stage 06 has no grounds to trust it and predicts everything
again.

This writes the missing signature, but only for conditions it can verify.  The
check is the one that matters: the (fold, run) pairs present in the stored
predictions must be exactly the (fold, run) pairs that have a checkpoint on
disk.  If a run was collected after the table was written, or a checkpoint is
missing, or the condition was never predicted at all, the sets differ and that
condition is left out -- stage 06 will then predict it, which is the correct
outcome rather than a failure.

This is a one-off.  Once stage 06 has written a table and its signature
together, the cache maintains itself.

    python scripts/bootstrap_prediction_cache.py --config configs/thesis.yaml
    python scripts/bootstrap_prediction_cache.py --config configs/thesis.yaml --dry-run
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
from _common import base_parser, start

from pvc_dlif.logging_utils import get_logger

LOGGER = get_logger("qc.cache")

CHECKPOINT = re.compile(r"fold_(\d+)[/\\]run_(\d+)")


def _pairs_on_disk(root: Path) -> set[tuple[int, int]]:
    """The (fold, run) pairs that have a checkpoint file under ``root``."""
    pairs: set[tuple[int, int]] = set()
    for path in root.rglob("*.pt"):
        relative = path.relative_to(root).as_posix()
        # Anything quarantined under a "_"-prefixed folder is deliberately out
        # of the analysis and must not count towards the signature.
        if any(part.startswith("_") for part in Path(relative).parts):
            continue
        match = CHECKPOINT.search(relative)
        if match:
            pairs.add((int(match.group(1)), int(match.group(2))))
    return pairs


def main() -> int:
    # --dry-run comes from base_parser, which every stage shares.
    parser = base_parser(__doc__ or "")
    args = parser.parse_args()
    config = start(args, "bootstrap_cache")

    table_path = config.dir_predictions / "retrained.parquet"
    if not table_path.exists():
        table_path = table_path.with_suffix(".csv")
    if not table_path.exists():
        raise SystemExit("No retrained prediction table to certify; run stage 06 first.")

    table = (pd.read_parquet(table_path) if table_path.suffix == ".parquet"
             else pd.read_csv(table_path))
    if not {"condition", "fold", "run"} <= set(table.columns):
        raise SystemExit(f"{table_path.name} has no fold/run columns; it cannot be verified.")

    stored = table.groupby("condition").apply(
        lambda g: {(int(f), int(r)) for f, r in zip(g["fold"], g["run"])},
        include_groups=False,
    ).to_dict()

    signatures: dict[str, dict] = {}
    for condition in config.conditions:
        if condition.model != "retrained":
            continue
        name = condition.name
        source = condition.checkpoints_from or name
        root = config.dir_models / source
        if not root.exists():
            LOGGER.info("%-32s no checkpoints on disk", name)
            continue
        if name not in stored:
            LOGGER.info("%-32s not in the table; stage 06 will predict it", name)
            continue

        on_disk = _pairs_on_disk(root)
        in_table = stored[name]
        if on_disk != in_table:
            missing, extra = sorted(on_disk - in_table), sorted(in_table - on_disk)
            LOGGER.warning(
                "%-32s NOT certified: %d checkpoint(s) absent from the table, "
                "%d row-group(s) with no checkpoint; it will be predicted again",
                name, len(missing), len(extra),
            )
            continue

        # Must be built exactly as stage 06 builds it, quarantine exclusion
        # included: a signature computed differently would never match, and the
        # certificate would silently do nothing.
        signatures[name] = {
            "input_tag": condition.input_tag,
            "source": source,
            "checkpoints": sorted(
                str(p.relative_to(root).as_posix()) for p in root.rglob("*.pt")
                if not any(part.startswith("_") for part in p.relative_to(root).parts)
            ),
        }
        LOGGER.info("%-32s certified (%d fold/run pairs)", name, len(on_disk))

    if not signatures:
        LOGGER.warning("Nothing could be certified; stage 06 will predict everything.")
        return 0

    path = config.dir_predictions / "retrained.signature.json"
    if args.dry_run:
        LOGGER.info("would write %d certified condition(s) to %s", len(signatures), path.name)
        return 0

    path.write_text(json.dumps(signatures, indent=2), encoding="utf-8")
    LOGGER.info("Wrote %s: %d condition(s) stage 06 can now reuse", path.name, len(signatures))
    return 0


if __name__ == "__main__":
    sys.exit(main())
