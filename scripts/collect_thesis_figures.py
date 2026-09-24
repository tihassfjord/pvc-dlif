"""Copy the figures a LaTeX document references into a folder beside it.

The thesis is written against figures that stage 08 regenerates, and it has to
be uploadable to Overleaf as one self-contained folder.  Those two facts pull
against each other: keeping the document pointed at the report directory means
it cannot be uploaded, and copying the figures by hand means they go stale the
next time stage 08 runs.

This reads the document, finds every ``\\includegraphics``, and copies exactly
those files from the report directory into ``<document folder>/figures``.  Run
it after every stage 08 and the uploaded folder is current; nothing else is
copied, so the folder does not accumulate figures the text stopped using.

A figure the document names but stage 08 has not produced is reported rather
than passed over, because the usual cause is that the document was edited to
reference a figure before the stage that makes it was re-run.

    python scripts/collect_thesis_figures.py --config configs/thesis.yaml
    python scripts/collect_thesis_figures.py --config configs/thesis.yaml --dry-run
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

from _common import base_parser, start

from pvc_dlif.logging_utils import get_logger

LOGGER = get_logger("qc.figures")

INCLUDE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")


def main() -> int:
    parser = base_parser(__doc__ or "")
    parser.add_argument("--document", default=None,
                        help="the .tex file (default: <work>/thesis_draft/thesis_draft.tex)")
    parser.add_argument("--into", default="figures",
                        help="folder name beside the document (default: figures)")
    parser.add_argument("--prune", action="store_true",
                        help="delete files in the target folder the document no longer names")
    args = parser.parse_args()
    config = start(args, "collect_figures")

    document = Path(args.document) if args.document else (
        config.work / "thesis_draft" / "thesis_draft.tex")
    if not document.exists():
        raise SystemExit(f"No document at {document}")

    target = document.parent / args.into
    text = document.read_text(encoding="utf-8")
    wanted = sorted(set(INCLUDE.findall(text)))
    if not wanted:
        LOGGER.warning("%s references no figures", document.name)
        return 0

    LOGGER.info("%s references %d figure(s)", document.name, len(wanted))
    copied, missing = [], []
    for name in wanted:
        source = config.dir_report / name
        if not source.exists():
            # LaTeX allows the extension to be omitted; try the usual ones.
            for suffix in (".pdf", ".png"):
                candidate = config.dir_report / (name + suffix)
                if candidate.exists():
                    source = candidate
                    break
        if not source.exists():
            missing.append(name)
            continue
        destination = target / source.name
        if args.dry_run:
            LOGGER.info("would copy %s", source.name)
            copied.append(source.name)
            continue
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(source.name)

    for name in missing:
        LOGGER.warning("%s: not in %s -- run stage 08 first if the document was "
                       "edited to reference a new figure", name, config.dir_report.name)

    if args.prune and target.exists() and not args.dry_run:
        keep = set(copied)
        for path in target.iterdir():
            if path.is_file() and path.name not in keep:
                LOGGER.info("removing %s, no longer referenced", path.name)
                path.unlink()

    LOGGER.info("%d figure(s) %s %s", len(copied),
                "would go to" if args.dry_run else "in", target)
    if missing:
        LOGGER.error("%d figure(s) missing; the document will not compile cleanly",
                     len(missing))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
