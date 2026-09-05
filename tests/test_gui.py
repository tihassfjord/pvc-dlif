"""Tests for the GUI's non-interactive parts.

The widgets themselves are not tested - clicking buttons in CI is not worth the
machinery.  What is tested is the code that does work: the PVC worker function
and the pipeline tab's command building, both of which are plain functions
deliberately kept free of widget references.

Skipped entirely where tkinter is unavailable (headless containers, some conda
builds), because the modules import it at the top level.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tkinter", reason="GUI modules import tkinter at module level")
nib = pytest.importorskip("nibabel")

GUI_DIR = Path(__file__).resolve().parents[1] / "gui"
if str(GUI_DIR) not in sys.path:
    sys.path.insert(0, str(GUI_DIR))


def _write_blurred_sphere(path: Path, frames: int | None = None) -> None:
    """A hot sphere, Gaussian-blurred - something deconvolution can act on."""
    from scipy.ndimage import gaussian_filter

    grid = np.zeros((24, 24, 24), np.float32)
    z, y, x = np.ogrid[:24, :24, :24]
    grid[((z - 12) ** 2 + (y - 12) ** 2 + (x - 12) ** 2) < 9] = 100.0
    blurred = gaussian_filter(grid, 1.2).astype(np.float32)
    data = (np.stack([blurred * s for s in np.linspace(1.0, 0.4, frames)], axis=-1)
            if frames else blurred)
    nib.save(nib.Nifti1Image(data, np.diag([0.5, 0.5, 0.6, 1])), str(path))


SETTINGS = dict(method="RL", alpha=1.5, fwhm=(0.864, 0.874, 0.994),
                disable_stopping=True, backend="numpy", workers=1)


def test_worker_corrects_3d_and_4d_and_writes_sidecars(tmp_path):
    """One 3D volume and one 4D series, two iteration counts, in one call."""
    from pvc_dlif_gui.tab_pvc import _correct_all

    _write_blurred_sphere(tmp_path / "vol.nii.gz")
    _write_blurred_sphere(tmp_path / "series.nii.gz", frames=3)

    lines: list[str] = []
    _correct_all([tmp_path / "vol.nii.gz", tmp_path / "series.nii.gz"],
                 tmp_path / "out", [5, 10], SETTINGS, lines.append)

    written = sorted(p.name for p in (tmp_path / "out").iterdir())
    assert "vol_rl_i5.nii.gz" in written and "series_rl_i10.nii.gz" in written

    # The sidecar is what makes a result traceable; check it is complete.
    sidecar = json.loads((tmp_path / "out" / "vol_rl_i10.pvc.json").read_text())
    assert sidecar["method"] == "RL"
    assert sidecar["iterations"] == 10
    assert sidecar["psf_fwhm_mm"] == [0.864, 0.874, 0.994]
    assert sidecar["backend"] == "numpy"

    # The 4D file keeps its fourth dimension.
    assert nib.load(str(tmp_path / "out" / "series_rl_i5.nii.gz")).shape == (24, 24, 24, 3)

    # One progress tick per (file, iteration count).
    assert sum(1 for line in lines if line == "__progress__") == 4


def test_worker_recovers_the_peak(tmp_path):
    """Deconvolution should raise the peak of a blurred sphere, not lower it."""
    from pvc_dlif_gui.tab_pvc import _correct_all

    _write_blurred_sphere(tmp_path / "vol.nii.gz")
    _correct_all([tmp_path / "vol.nii.gz"], tmp_path / "out", [10], SETTINGS, lambda _l: None)

    before = np.asarray(nib.load(str(tmp_path / "vol.nii.gz")).dataobj)
    after = np.asarray(nib.load(str(tmp_path / "out" / "vol_rl_i10.nii.gz")).dataobj)
    assert after.max() > before.max()
    assert after.min() >= -1e-6            # RL is non-negative by construction


def test_worker_stops_when_asked(tmp_path):
    """The cancel check is honoured between files."""
    from pvc_dlif_gui.tab_pvc import _correct_all

    for name in ("a.nii.gz", "b.nii.gz"):
        _write_blurred_sphere(tmp_path / name)

    lines: list[str] = []
    _correct_all([tmp_path / "a.nii.gz", tmp_path / "b.nii.gz"], tmp_path / "out",
                 [5], SETTINGS, lines.append, should_stop=lambda: True)
    assert "cancelled" in lines
    assert not (tmp_path / "out").exists() or not list((tmp_path / "out").glob("*.nii.gz"))


def test_pilot_arguments_come_from_run_all():
    """The GUI's pilot mode must be the same pilot mode as the command line."""
    from pvc_dlif_gui.tab_pipeline import _pilot_args

    assert _pilot_args("05") == ["--limit", "8", "--folds", "1", "--runs", "1", "--epochs", "5"]
    assert _pilot_args("00") == []
