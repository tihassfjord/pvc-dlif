"""What each stage has produced, read from the work tree.

Status is computed from the filesystem every time it is asked for, never held
in memory.  That way it survives a crashed GUI, a closed laptop, or a stage run
from a terminal while the GUI was open: the files are the truth.

Each stage has a simple rule for "done" and, where it is per scan, a count
against the number of usable scans in the manifest.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Config

__all__ = ["StageStatus", "stage_statuses", "manifest_summary", "prepare_summary"]


@dataclass
class StageStatus:
    """One stage's progress as far as the files can tell."""

    stage_id: str
    label: str
    done: int                 # items produced
    total: int | None         # items expected, None when not per-item
    detail: str = ""          # short human-readable note

    @property
    def state(self) -> str:
        """'done', 'partial' or 'missing' - what the status light shows."""
        if self.total is None:
            return "done" if self.done else "missing"
        if self.done == 0:
            return "missing"
        return "done" if self.done >= self.total else "partial"

    @property
    def progress_text(self) -> str:
        if self.total is None:
            return "done" if self.done else "not run"
        return f"{self.done} / {self.total}"


# -------------------------------------------------------------------- #
# Small readers for the two JSON summaries the GUI shows as banners
# -------------------------------------------------------------------- #
def manifest_summary(config: Config) -> dict[str, Any] | None:
    """The summary block of manifest.json, or None before stage 00 has run."""
    path = config.work / "manifest.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = dict(payload.get("summary", {}))

    # Exclusion breakdown by reason, so the banner can say *why* n is what it is.
    reasons: dict[str, int] = {}
    for scan in payload.get("scans", []):
        if scan.get("excluded"):
            reason = scan.get("exclusion_reason") or "ignore_ids"
            reasons[reason] = reasons.get(reason, 0) + 1
    summary["exclusion_reasons"] = reasons
    summary["usable_ids"] = [
        s["scan_id"] for s in payload.get("scans", [])
        if not s.get("excluded") and s.get("has_dicom") and s.get("has_dlif_img") and s.get("has_aif")
    ]
    return summary


def prepare_summary(config: Config) -> dict[str, Any] | None:
    """Stage 01's agreement figures, plus every scan whose fit is below 0.999."""
    path = config.dir_results / "prepare_summary.json"
    if not path.exists():
        return None
    summary = json.loads(path.read_text(encoding="utf-8"))

    plan_path = config.work / "preprocessing_plan.json"
    below: list[tuple[str, float]] = []
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        for scan_id, fit in plan.get("fits", {}).items():
            r = fit.get("correlation")
            if r is not None and r < 0.999:
                below.append((scan_id, float(r)))
    summary["scans_below_0999"] = sorted(below, key=lambda item: item[1])
    return summary


# -------------------------------------------------------------------- #
def _count(pattern_dir: Path, glob: str) -> int:
    return len(list(pattern_dir.glob(glob))) if pattern_dir.exists() else 0


def _usable_count(config: Config) -> int | None:
    summary = manifest_summary(config)
    return int(summary["n_usable"]) if summary else None


def stage_statuses(config: Config) -> list[StageStatus]:
    """Progress of every stage, in order."""
    work = config.work
    n = _usable_count(config)
    statuses: list[StageStatus] = []

    # 00 - the manifest either exists or it does not
    manifest = manifest_summary(config)
    statuses.append(StageStatus(
        "00", "Inventory", 1 if manifest else 0, None,
        f"{manifest['n_usable']} usable of {manifest['n_total']}" if manifest else ""))

    # 01 - one native NIfTI per usable scan, plus the plan
    native = _count(config.dir_native, "*.nii.gz")
    plan_ok = (work / "preprocessing_plan.json").exists()
    statuses.append(StageStatus(
        "01", "Convert + crop", native, n,
        "crop plan written" if plan_ok else "no preprocessing_plan.json"))

    # 02 - a scan counts as done when every primary (method, k) exists for it
    primary_tags = [f"{m.lower()}_i{config.iterations_primary}" for m in config.pvc_methods]
    per_scan_done = 0
    if n and native:
        for path in config.dir_native.glob("*.nii.gz"):
            scan_id = path.name.replace(".nii.gz", "")
            if all((config.dir_pvc / tag / f"{scan_id}.nii.gz").exists() for tag in primary_tags):
                per_scan_done += 1
    sweep_tags = sorted(p.name for p in config.dir_pvc.iterdir()) if config.dir_pvc.exists() else []
    statuses.append(StageStatus(
        "02", "PVC", per_scan_done, n,
        f"tags on disk: {', '.join(sweep_tags)}" if sweep_tags else "primary: " + ", ".join(primary_tags)))

    # 03 - one input tree per distinct input tag, each with one pkl per scan
    tags = sorted({c.input_tag for c in config.conditions if not c.motion})
    trees_done = 0
    for tag in tags:
        root = config.dir_dlif_inputs / tag
        if root.exists() and any(root.rglob("IMG_*.pkl")):
            trees_done += 1
    statuses.append(StageStatus("03", "Network inputs", trees_done, len(tags),
                                "input trees: " + ", ".join(tags)))

    # 04 - a predictions table from the deployed model
    pre = config.dir_predictions / "pretrained"
    has_pre = pre.with_suffix(".parquet").exists() or pre.with_suffix(".csv").exists()
    statuses.append(StageStatus("04", "Pretrained inference", 1 if has_pre else 0, None,
                                "predictions/pretrained" if has_pre else ""))

    # 05 - folds x runs x retrained conditions, each leaving a summary.json
    retrained = [c for c in config.conditions if c.model == "retrained" and not c.motion]
    folds = int(config.get("dlif.cv.n_folds", 10))
    runs = int(config.get("dlif.cv.n_runs", 10))
    trainings_done = sum(
        len(list((config.dir_models / c.name).glob("fold_*/run_*/summary.json")))
        for c in retrained if (config.dir_models / c.name).exists()
    )
    statuses.append(StageStatus(
        "05", "Retrain", trainings_done, len(retrained) * folds * runs,
        f"{len(retrained)} conditions x {folds} folds x {runs} runs"))

    # 06 - the headline summary
    has_summary = (config.dir_results / "summary.json").exists()
    tables = sorted(p.stem for p in config.dir_results.glob("*.csv")) if config.dir_results.exists() else []
    statuses.append(StageStatus("06", "Evaluate", 1 if has_summary else 0, None,
                                ", ".join(tables) if tables else ""))

    # 07 - motion QC table (exploratory, so absence is normal)
    qc = config.dir_motion / "motion_qc.csv"
    statuses.append(StageStatus("07", "Motion subset", 1 if qc.exists() else 0, None,
                                "motion_qc.csv" if qc.exists() else "exploratory; skipped is fine"))

    # 08 - figures and LaTeX
    pdfs = _count(config.dir_report, "*.pdf")
    texs = _count(config.dir_report, "*.tex")
    statuses.append(StageStatus("08", "Report", pdfs + texs, None,
                                f"{pdfs} figures, {texs} tables" if pdfs + texs else ""))

    return statuses
