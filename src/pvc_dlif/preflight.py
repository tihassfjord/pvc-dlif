"""Is this machine ready to run the pipeline?

Each check returns a :class:`Check` with a pass/fail flag and, on failure, the
exact thing to do about it.  The GUI shows them as a list; the command line
can print them with ``python -m pvc_dlif.preflight``.

Nothing here is clever.  It exists because the two most likely reasons for a
first run to fail - PETPVC not installed, CUDA not available - are both
invisible until a stage is minutes in, and both have a one-line fix.
"""

from __future__ import annotations

import importlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["Check", "run_checks", "disk_estimate_gb"]


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""       # what was found
    fix: str = ""          # what to do if not ok
    severity: str = "error"   # "error" blocks the run, "warn" does not


# Bytes per scan for the intermediate products, measured on the real data.
_NATIVE_MB_PER_SCAN = 26.0          # 42 frames of 128x92x92 float32, gzipped


def disk_estimate_gb(config: Any, sweep: bool = False) -> float:
    """Rough space a full run needs under <work>, in GB."""
    n = 70
    try:
        from .status import manifest_summary
        summary = manifest_summary(config)
        if summary:
            n = int(summary["n_usable"])
    except Exception:              # noqa: BLE001 - estimate only
        pass
    settings = len(config.pvc_methods) * (len(config.iterations_grid) if sweep else 1)
    per_scan = _NATIVE_MB_PER_SCAN * (1 + settings)        # native + each corrected series
    inputs = len({c.input_tag for c in config.conditions}) * 4.5   # 96x48x48 x 42 float64 pkl
    return (n * (per_scan + inputs)) / 1024.0


def _free_gb(path: Path) -> float | None:
    try:
        probe = path if path.exists() else next(p for p in path.parents if p.exists())
        return shutil.disk_usage(probe).free / 1024 ** 3
    except Exception:              # noqa: BLE001
        return None


def run_checks(config: Any | None, config_error: str = "") -> list[Check]:
    """All checks, in the order they matter."""
    checks: list[Check] = []

    # --- Python dependencies ------------------------------------------ #
    for module, package, needed_for in (
        ("numpy", "numpy", "everything"),
        ("scipy", "scipy", "everything"),
        ("pandas", "pandas", "everything"),
        ("nibabel", "nibabel", "NIfTI I/O"),
        ("pydicom", "pydicom", "stage 01"),
        ("sklearn", "scikit-learn", "cross-validation folds"),
        ("matplotlib", "matplotlib", "figures"),
        ("yaml", "pyyaml", "the config"),
    ):
        try:
            importlib.import_module(module)
            checks.append(Check(f"python: {package}", True, "installed"))
        except ImportError:
            checks.append(Check(f"python: {package}", False, f"missing ({needed_for})",
                                f'pip install -e ".[all]"'))

    # pyarrow is optional: without it tables fall back to CSV.
    try:
        importlib.import_module("pyarrow")
        checks.append(Check("python: pyarrow", True, "installed (parquet tables)"))
    except ImportError:
        checks.append(Check("python: pyarrow", False, "missing - tables written as CSV only",
                            "pip install pyarrow", severity="warn"))

    # --- PETPVC --------------------------------------------------------- #
    exe = None
    if config is not None:
        try:
            exe = config.petpvc_exe
        except Exception:          # noqa: BLE001
            exe = None
    found = str(exe) if exe and Path(exe).exists() else shutil.which("petpvc")
    checks.append(Check(
        "PETPVC toolbox", bool(found),
        found or "not on PATH",
        "conda install -c conda-forge petpvc   (or set paths.petpvc_exe). Stage 02 refuses to "
        "run without it; the numpy fallback is for development only."))

    # --- torch / CUDA --------------------------------------------------- #
    try:
        torch = importlib.import_module("torch")
        cuda = bool(torch.cuda.is_available())
        name = torch.cuda.get_device_name(0) if cuda else ""
        checks.append(Check("torch", True, f"torch {torch.__version__}"))
        checks.append(Check(
            "CUDA GPU", cuda, name or "no CUDA device visible",
            "Stage 05 is 300 trainings and is not feasible on CPU. Install a CUDA build of "
            "torch (see pytorch.org) or run stage 05 on a machine with a GPU.",
            severity="warn"))
    except ImportError:
        checks.append(Check("torch", False, "not installed",
                            "pip install torch   (CUDA build recommended; stages 04 and 05 need it)"))

    # --- config and paths ------------------------------------------------ #
    if config is None:
        checks.append(Check("config", False, config_error or "not loaded",
                            "Fix the config file; the error above says what is wrong."))
        return checks
    checks.append(Check("config", True, str(config.source)))

    for name, must_exist in (("dicom_root", True), ("dlif_data_root", True),
                             ("dlif_repo", True), ("work", False)):
        try:
            path = getattr(config, name)
        except Exception as exc:   # noqa: BLE001
            checks.append(Check(f"path: {name}", False, str(exc), "Set it on this page."))
            continue
        exists = Path(path).exists()
        if must_exist:
            checks.append(Check(f"path: {name}", exists, str(path),
                                "Point this at the folder described in docs/data.md."))
        else:
            parent_ok = exists or any(p.exists() for p in Path(path).parents)
            checks.append(Check(f"path: {name}", parent_ok,
                                f"{path} ({'exists' if exists else 'will be created'})",
                                "Choose a folder on a drive that exists."))

    # The three things the DLIF repo must contain
    try:
        repo = config.dlif_repo
        weights = repo / config.pretrained_weights
        checks.append(Check("pretrained weights", weights.exists(), str(weights),
                            "The DLIF repository must contain src/models/pretrained_weigths/DLIFNet.pt."))
        models_py = repo / "src" / "models" / "models.py"
        checks.append(Check("DLIF models.py", models_py.exists(), str(models_py),
                            "Clone the group's DLIF repository at paths.dlif_repo."))
    except Exception as exc:       # noqa: BLE001
        checks.append(Check("DLIF repository", False, str(exc), "Set paths.dlif_repo."))

    # --- disk ------------------------------------------------------------- #
    try:
        need = disk_estimate_gb(config, sweep=True)
        free = _free_gb(config.work)
        ok = free is None or free > need
        checks.append(Check(
            "disk space", ok,
            f"need ~{need:.0f} GB with the iteration sweep; {free:.0f} GB free" if free is not None
            else f"need ~{need:.0f} GB; free space unknown",
            "Point paths.work at a drive with more room, or skip --sweep.",
            severity="warn"))
    except Exception as exc:       # noqa: BLE001
        checks.append(Check("disk space", True, f"estimate unavailable: {exc}", severity="warn"))

    return checks


def main() -> int:               # pragma: no cover - convenience entry point
    import sys
    from .config import load_config
    try:
        config = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
        error = ""
    except Exception as exc:       # noqa: BLE001
        config, error = None, str(exc)
    worst = 0
    for check in run_checks(config, error):
        mark = "ok  " if check.ok else ("WARN" if check.severity == "warn" else "FAIL")
        print(f"[{mark}] {check.name:22s} {check.detail}")
        if not check.ok:
            print(f"       -> {check.fix}")
            worst = max(worst, 1 if check.severity == "warn" else 2)
    return 0 if worst < 2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
