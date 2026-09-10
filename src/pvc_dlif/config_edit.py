"""Editing a handful of values in thesis.yaml without losing its comments.

The config file is heavily commented on purpose - the comments are where the
reasoning for each setting lives - so a round trip through a YAML dumper, which
throws comments away, is not acceptable.  Instead the few keys the GUI exposes
are replaced line by line with a regular expression, and the result is loaded
back through :func:`pvc_dlif.config.load_config` so an invalid edit is refused
before it is written.

Keys are located by their full dotted path, not by their bare name.  Matching
on the bare name was fine until the config grew a ``dlif.regimes`` block, where
``n_folds`` and ``n_runs`` appear once per regime as well as in ``dlif.cv`` -
at which point every GUI save failed with "occurs 3 times".  The scanner below
tracks indentation to reconstruct each line's path, so ``dlif.cv.n_folds``
matches only the real one.  A path that still resolves to more than one line is
an error rather than a guess.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .config import load_config

__all__ = ["EDITABLE", "resolve_key", "set_value", "yaml_scalar", "save_values"]

#: Dotted key -> the bare YAML key, kept for readability of what is exposed.
#: Matching uses the dotted key itself; the value is documentation.
EDITABLE: dict[str, str] = {
    "paths.dicom_root": "dicom_root",
    "paths.dlif_data_root": "dlif_data_root",
    "paths.dlif_repo": "dlif_repo",
    "paths.work": "work",
    "paths.petpvc_exe": "petpvc_exe",
    "pvc.iterations_primary": "iterations_primary",
    "pvc.methods": "methods",
    "pvc.workers": "workers",
    "dlif.cv.n_folds": "n_folds",
    "dlif.cv.n_runs": "n_runs",
    "dlif.train.device": "device",
    "motion.affected_ids": "affected_ids",
}


def resolve_key(config: Any, dotted: str) -> str:
    """Where an edit to ``dotted`` has to land so the regime does not undo it.

    ``dlif.cv.*`` and ``dlif.train.*`` are overwritten at load by the selected
    regime's patch, so writing ``dlif.cv.n_runs`` while a regime names that key
    changes a value nothing reads.  Redirect the edit into the active regime,
    which is what the person adjusting it means: change the protocol I am
    running, not the one I am not.
    """
    regime = getattr(config, "regime", "base")
    if regime == "base":
        return dotted
    parts = dotted.split(".")
    if len(parts) == 3 and parts[0] == "dlif" and parts[1] in ("cv", "train"):
        patch = config.get(f"dlif.regimes.{regime}.{parts[1]}") or {}
        if parts[2] in patch:
            return f"dlif.regimes.{regime}.{parts[1]}.{parts[2]}"
    return dotted


def yaml_scalar(value: Any) -> str:
    """Render a Python value as a single-line YAML scalar or flow list."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(yaml_scalar(v) for v in value) + "]"
    text = str(value).replace("\\", "/")
    return '"' + text.replace('"', '\\"') + '"'


#: "  key:" at the start of a line.  Keys inside a flow mapping ("{a: 1}") and
#: list items are deliberately not matched: they carry no block structure.
_KEY_LINE = re.compile(r"^(\s*)([A-Za-z_][\w.\-]*|\"[^\"]+\"|'[^']+')\s*:(\s|$)")


def _find_lines(text: str, dotted: str) -> list[int]:
    """Indices of the lines whose reconstructed path equals ``dotted``.

    Indentation is the only structure YAML block mappings have, so the path of
    a line is the stack of keys still open above it.
    """
    parts = dotted.split(".")
    stack: list[tuple[int, str]] = []
    hits: list[int] = []
    for i, line in enumerate(text.splitlines()):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _KEY_LINE.match(line)
        if match is None:
            continue
        indent, key = len(match.group(1)), match.group(2).strip("\"'")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        stack.append((indent, key))
        if [k for _, k in stack] == parts:
            hits.append(i)
    return hits


def set_value(text: str, dotted: str, value: Any) -> str:
    """Return ``text`` with one key's value replaced, comments intact."""
    hits = _find_lines(text, dotted)
    if len(hits) != 1:
        raise ValueError(f"key {dotted!r} matches {len(hits)} lines; expected exactly one")

    lines = text.splitlines(keepends=True)
    index = hits[0]
    # Keep the indentation and the key, replace the rest of the line - which
    # includes any trailing comment, since that comment described the old value.
    head = re.match(r"^(\s*[^:]+:\s*)", lines[index]).group(1)
    ending = "\n" if lines[index].endswith("\n") else ""
    lines[index] = head + yaml_scalar(value) + ending
    return "".join(lines)


def save_values(path: Path, values: dict[str, Any]) -> Path:
    """Apply several edits, validate by loading, then write.

    The validated text is written to a temporary file first and loaded from
    there, so a config that fails validation never replaces the good one.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    current = load_config(path)
    for dotted, value in values.items():
        text = set_value(text, resolve_key(current, dotted), value)

    probe = path.with_suffix(".yaml.tmp")
    probe.write_text(text, encoding="utf-8")
    try:
        load_config(probe)            # raises with a readable message on error
    finally:
        probe.unlink(missing_ok=True)

    path.write_text(text, encoding="utf-8")
    return path
