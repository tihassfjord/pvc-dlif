"""Tests for the GUI's worker functions.

The widgets themselves are not tested - clicking buttons in CI is not worth the
machinery.  What is tested is the code that does work: the PVC tab's batch
worker and the pipeline tab's command building, both plain functions kept
free of widget references.  The library modules the GUI sits on are covered in
``test_gui_support.py`` without needing tkinter at all.

Skipped where tkinter is unavailable (headless containers, some conda builds),
because the tab modules import it at the top level.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tkinter", reason="GUI tab modules import tkinter at module level")
nib = pytest.importorskip("nibabel")

GUI_DIR = Path(__file__).resolve().parents[1] / "gui"
if str(GUI_DIR) not in sys.path:
    sys.path.insert(0, str(GUI_DIR))


def _write_blurred_sphere(path: Path, frames: int = 3) -> None:
    """A 4D series of a hot sphere, Gaussian-blurred - something to deconvolve."""
    from scipy.ndimage import gaussian_filter

    grid = np.zeros((24, 24, 24), np.float32)
    z, y, x = np.ogrid[:24, :24, :24]
    grid[((z - 12) ** 2 + (y - 12) ** 2 + (x - 12) ** 2) < 9] = 100.0
    blurred = gaussian_filter(grid, 1.2).astype(np.float32)
    data = np.stack([blurred * s for s in np.linspace(1.0, 0.4, frames)], axis=-1)
    nib.save(nib.Nifti1Image(data, np.diag([0.5, 0.5, 0.6, 1])), str(path))


PARAMS = dict(methods=["RL"], counts=[5], alpha=1.5, fwhm=(0.864, 0.874, 0.994),
              backend="numpy", executable=None, workers=1, resume=True)


def test_worker_runs_through_run_batch_and_writes_diagnostics(tmp_path):
    from pvc_dlif_gui.tab_pvc import _run_batch_job

    for name in ("a", "b"):
        _write_blurred_sphere(tmp_path / f"{name}.nii.gz")
    pairs = [("a", tmp_path / "a.nii.gz"), ("b", tmp_path / "b.nii.gz")]
    lines: list[str] = []
    report = _run_batch_job(pairs, tmp_path / "out", PARAMS, lines.append, threading.Event())

    assert len(report["written"]) == 2 and not report["failed"]
    assert (tmp_path / "out" / "rl_i5" / "a.nii.gz").exists()
    assert (tmp_path / "out" / "rl_i5" / "a.diagnostics.json").exists()
    assert lines.count("__progress__") == 2

    # Resume: nothing recomputed on a second call
    report = _run_batch_job(pairs, tmp_path / "out", PARAMS, lines.append, threading.Event())
    assert len(report["skipped"]) == 2 and not report["written"]


def test_worker_recovers_the_peak(tmp_path):
    from pvc_dlif_gui.tab_pvc import _run_batch_job

    _write_blurred_sphere(tmp_path / "a.nii.gz")
    _run_batch_job([("a", tmp_path / "a.nii.gz")], tmp_path / "out", PARAMS, lambda _l: None,
                   threading.Event())
    before = np.asarray(nib.load(str(tmp_path / "a.nii.gz")).dataobj)
    after = np.asarray(nib.load(str(tmp_path / "out" / "rl_i5" / "a.nii.gz")).dataobj)
    assert after.max() > before.max()
    assert after.min() >= -1e-6


def test_worker_stops_at_a_scan_boundary(tmp_path):
    from pvc_dlif_gui.tab_pvc import _run_batch_job

    for name in ("a", "b", "c"):
        _write_blurred_sphere(tmp_path / f"{name}.nii.gz")
    pairs = [(n, tmp_path / f"{n}.nii.gz") for n in ("a", "b", "c")]
    stop = threading.Event()

    def log(line: str) -> None:
        if line.startswith("--- scan 2"):       # ask to stop while scan 2 is announced
            stop.set()

    report = _run_batch_job(pairs, tmp_path / "out", PARAMS, log, stop)
    assert report["stopped"]
    assert len(report["written"]) == 1           # scan 1 completed, 2 and 3 never started


def test_pilot_arguments_come_from_run_all():
    from pvc_dlif_gui.tab_pipeline import _pilot_args

    assert _pilot_args("05") == ["--limit", "8", "--folds", "1", "--runs", "1", "--epochs", "5"]
    assert _pilot_args("00") == []
