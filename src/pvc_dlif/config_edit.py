"""Editing a handful of values in thesis.yaml without losing its comments.

The config file is heavily commented on purpose - the comments are where the
reasoning for each setting lives - so a round trip through a YAML dumper, which
throws comments away, is not acceptable.  Instead the few keys the GUI exposes
are replaced line by line with a regular expression, and the result is loaded
back through :func:`pvc_dlif.config.load_config` so an invalid edit is refused
before it is written.

Only keys that occur exactly once in the file are editable this way.  Every
key in :data:`EDITABLE` satisfies that in the shipped configs; :func:`set_value`
raises if it does not.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .config import load_config

__all__ = ["EDITABLE", "set_value", "yaml_scalar", "save_values"]

#: Dotted key -> the bare YAML key the line starts with.
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


def set_value(text: str, dotted: str, value: Any) -> str:
    """Return ``text`` with one key's value replaced, comments intact."""
    key = EDITABLE[dotted]
    # Match "  key:  <anything>" at line start; keep indentation and key,
    # replace the rest of the line including any trailing comment.
    pattern = re.compile(rf"^(\s*{re.escape(key)}\s*:\s*).*$", re.MULTILINE)
    matches = pattern.findall(text)
    if len(matches) != 1:
        raise ValueError(f"key {key!r} occurs {len(matches)} times; expected exactly once")
    return pattern.sub(lambda m: m.group(1) + yaml_scalar(value), text, count=1)


def save_values(path: Path, values: dict[str, Any]) -> Path:
    """Apply several edits, validate by loading, then write.

    The validated text is written to a temporary file first and loaded from
    there, so a config that fails validation never replaces the good one.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    for dotted, value in values.items():
        text = set_value(text, dotted, value)

    probe = path.with_suffix(".yaml.tmp")
    probe.write_text(text, encoding="utf-8")
    try:
        load_config(probe)            # raises with a readable message on error
    finally:
        probe.unlink(missing_ok=True)

    path.write_text(text, encoding="utf-8")
    return path
