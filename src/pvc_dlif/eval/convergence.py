"""Which training runs left the loss plateau.

Every retrained run starts on a plateau near validation loss 4 and leaves it at
a stochastic epoch; a few never do.  A run that never leaves it has not fitted
the task, so its predictions measure the optimiser, not the input
representation.  The criterion is training-side only -- the lowest validation
loss in the stored history -- and never looks at test performance, so excluding
on it does not select on the outcome.

The threshold sits in a wide empty gap: on the thesis runs the five that stay on
the plateau have best validation loss 3.36-4.37, and the best of the remaining
595 is 1.56, so any threshold between those gives the same five
(audit/a14_convergence.py).

Conditions evaluated on another condition's checkpoints (the shift conditions)
inherit that condition's runs; ``signature`` maps each evaluated condition to
the checkpoint set it used.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

__all__ = ["best_val_loss", "run_convergence"]


def _val_loss(summary: Mapping[str, Any]) -> list[float]:
    history = summary["history"]
    series = (history.get("val_loss") if isinstance(history, Mapping)
              else [e.get("val_loss") for e in history])
    return [float(x) for x in series if x is not None]


def best_val_loss(summary_path: Path) -> float:
    with open(summary_path, encoding="utf-8") as handle:
        series = _val_loss(json.load(handle))
    return min(series) if series else float("nan")


def run_convergence(models_root: Path, conditions_runs: pd.DataFrame,
                    signature: Mapping[str, Mapping[str, Any]] | None,
                    threshold: float) -> pd.DataFrame:
    """One row per (condition, fold, run) in ``conditions_runs``.

    ``conditions_runs`` needs columns ``condition, fold, run``.  A run whose
    summary cannot be found is reported with ``converged = NaN`` and is not
    excluded -- a missing file is a problem to fix, not a reason to drop data.
    """
    rows = []
    for (condition, fold, run), _ in conditions_runs.groupby(["condition", "fold", "run"]):
        source = str((signature or {}).get(condition, {}).get("source", condition))
        path = (Path(models_root) / source / f"fold_{int(fold):02d}"
                / f"run_{int(run):02d}" / "summary.json")
        best = best_val_loss(path) if path.exists() else float("nan")
        rows.append({
            "condition": condition, "fold": int(fold), "run": int(run), "source": source,
            "best_val_loss": best,
            "converged": (best < threshold) if best == best else float("nan"),
            "summary_found": path.exists(),
        })
    return pd.DataFrame(rows)
