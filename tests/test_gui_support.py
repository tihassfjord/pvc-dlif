"""Tests for the library modules the GUI is built on.

None of these need tkinter: they cover the filesystem status scanner, the
comment-preserving config editor, the preflight checks, the shared figure
assembly, and the detached-process helpers.  The GUI itself is a thin layer
over these, so this is where its behaviour is actually pinned down.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pvc_dlif.config import load_config
from pvc_dlif.config_edit import save_values, set_value, yaml_scalar
from pvc_dlif.preflight import disk_estimate_gb, run_checks
from pvc_dlif.report import assemble
from pvc_dlif.status import manifest_summary, prepare_summary, stage_statuses, training_progress

REPO = Path(__file__).resolve().parents[1]
GUI_DIR = REPO / "gui"


# ==================================================================== #
# Fixtures: a config pointing at an empty work tree
# ==================================================================== #
@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    """A copy of example.yaml with all four paths under tmp_path."""
    text = (REPO / "configs" / "example.yaml").read_text(encoding="utf-8")
    target = tmp_path / "thesis.yaml"
    target.write_text(text, encoding="utf-8")
    for dotted, sub in (("paths.dicom_root", "dicom"), ("paths.dlif_data_root", "data"),
                        ("paths.dlif_repo", "repo"), ("paths.work", "work")):
        (tmp_path / sub).mkdir(exist_ok=True)
    save_values(target, {
        "paths.dicom_root": str(tmp_path / "dicom"),
        "paths.dlif_data_root": str(tmp_path / "data"),
        "paths.dlif_repo": str(tmp_path / "repo"),
        "paths.work": str(tmp_path / "work"),
    })
    return target


def _fake_manifest(work: Path, usable: list[str], excluded: dict[str, str]) -> None:
    scans = [{"scan_id": s, "excluded": False, "has_dicom": True, "has_dlif_img": True, "has_aif": True}
             for s in usable]
    scans += [{"scan_id": s, "excluded": True, "exclusion_reason": r, "has_dicom": True,
               "has_dlif_img": False, "has_aif": True} for s, r in excluded.items()]
    payload = {"meta": {}, "summary": {"n_total": len(scans), "n_usable": len(usable),
                                       "n_excluded": len(excluded), "n_missing_dicom": 0,
                                       "n_missing_dlif_img": 0, "n_missing_aif": 0},
               "scans": scans}
    (work / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")


# ==================================================================== #
# status
# ==================================================================== #
class TestStatus:
    def test_empty_work_tree_is_all_missing(self, config_path):
        statuses = stage_statuses(load_config(config_path))
        assert [s.stage_id for s in statuses] == [f"{i:02d}" for i in range(9)]
        assert all(s.state == "missing" for s in statuses)

    def test_counts_follow_the_files(self, config_path):
        config = load_config(config_path)
        work = config.work
        _fake_manifest(work, ["A1", "A2", "A3"], {"T1": "ignore_ids"})
        config.dir_native.mkdir(parents=True)
        for scan in ("A1", "A2"):
            (config.dir_native / f"{scan}.nii.gz").touch()
        # A1 has every primary PVC tag, A2 only one of them
        for method in config.pvc_methods:
            tag = f"{method.lower()}_i{config.iterations_primary}"
            (config.dir_pvc / tag).mkdir(parents=True)
            (config.dir_pvc / tag / "A1.nii.gz").touch()
        (config.dir_pvc / f"{config.pvc_methods[0].lower()}_i{config.iterations_primary}" / "A2.nii.gz").touch()

        by_id = {s.stage_id: s for s in stage_statuses(config)}
        assert by_id["00"].state == "done"
        assert (by_id["01"].done, by_id["01"].total, by_id["01"].state) == (2, 3, "partial")
        assert (by_id["02"].done, by_id["02"].total) == (1, 3)      # only A1 complete
        assert by_id["05"].total == 3 * 10 * 10                        # conditions x folds x runs

    def test_manifest_summary_has_reasons_and_usable_ids(self, config_path):
        config = load_config(config_path)
        _fake_manifest(config.work, ["A1"], {"T1": "ignore_ids", "S-A": "no arterial curve"})
        summary = manifest_summary(config)
        assert summary["usable_ids"] == ["A1"]
        assert summary["exclusion_reasons"] == {"ignore_ids": 1, "no arterial curve": 1}

    def test_training_progress_reads_the_newest_heartbeat(self, config_path):
        """Stage 05 rewrites progress.json every epoch; the GUI shows the newest."""
        import time
        config = load_config(config_path)
        old = config.dir_models / "rl_retrained" / "fold_01" / "run_01"
        new = config.dir_models / "rl_retrained" / "fold_01" / "run_02"
        for folder, epoch in ((old, 1000), (new, 137)):
            folder.mkdir(parents=True)
            (folder / "progress.json").write_text(json.dumps({
                "epoch": epoch, "epochs": 1000, "train_loss": 1.0, "val_loss": 2.0,
                "best_val_loss": 2.0, "best_epoch": 0, "seconds_per_epoch": 1.5,
                "eta_seconds": 1.5 * (1000 - epoch), "updated": time.time()}))
        (old / "summary.json").write_text("{}")
        time.sleep(0.05)
        (new / "progress.json").touch()                    # newest
        progress = training_progress(config)
        assert (progress["condition"], progress["run"], progress["epoch"]) == ("rl_retrained", "run_02", 137)
        assert progress["finished"] is False and progress["age_seconds"] < 5

    def test_prepare_summary_names_scans_below_threshold(self, config_path):
        config = load_config(config_path)
        config.dir_results.mkdir(parents=True)
        (config.dir_results / "prepare_summary.json").write_text(
            json.dumps({"agreement_with_distributed_median_r": 0.9999}), encoding="utf-8")
        (config.work / "preprocessing_plan.json").write_text(json.dumps({
            "fits": {"AA1": {"correlation": 1.0}, "AB3": {"correlation": 0.97}}}), encoding="utf-8")
        summary = prepare_summary(config)
        assert summary["scans_below_0999"] == [("AB3", 0.97)]


# ==================================================================== #
# config_edit
# ==================================================================== #
class TestConfigEdit:
    def test_scalar_rendering(self):
        assert yaml_scalar(None) == "null"
        assert yaml_scalar(True) == "true"
        assert yaml_scalar(15) == "15"
        assert yaml_scalar(["RL", "RVC"]) == '["RL", "RVC"]'
        assert yaml_scalar("E:\\ML4PET\\x") == '"E:/ML4PET/x"'

    def test_set_value_keeps_everything_else(self):
        text = "pvc:\n  # why 15\n  iterations_primary: 15   # trailing\n  workers: 4\n"
        out = set_value(text, "pvc.iterations_primary", 20)
        assert "  # why 15\n" in out and "  workers: 4\n" in out
        assert "iterations_primary: 20\n" in out and "trailing" not in out

    def test_save_preserves_comments_and_round_trips(self, config_path):
        before = config_path.read_text(encoding="utf-8").count("\n#")
        save_values(config_path, {"pvc.iterations_primary": 20, "dlif.cv.n_runs": 3,
                                  "dlif.train.device": "cpu", "motion.affected_ids": ["AA1", "AB2"]})
        config = load_config(config_path)
        assert config.iterations_primary == 20
        assert config.get("dlif.cv.n_runs") == 3
        assert config.get("dlif.train.device") == "cpu"
        assert config.motion_affected_ids == ["AA1", "AB2"]
        assert config_path.read_text(encoding="utf-8").count("\n#") == before

    def test_invalid_edit_is_refused_and_file_untouched(self, config_path):
        original = config_path.read_text(encoding="utf-8")
        # Dropping RVC from the methods while RVC conditions exist is invalid.
        with pytest.raises(ValueError, match="RVC"):
            save_values(config_path, {"pvc.methods": ["RL"]})
        assert config_path.read_text(encoding="utf-8") == original
        assert not config_path.with_suffix(".yaml.tmp").exists()


# ==================================================================== #
# preflight
# ==================================================================== #
class TestPreflight:
    def test_reports_missing_repo_contents_with_a_fix(self, config_path):
        checks = {c.name: c for c in run_checks(load_config(config_path))}
        assert checks["pretrained weights"].ok is False
        assert "DLIFNet.pt" in checks["pretrained weights"].fix
        assert checks["path: dicom_root"].ok is True
        assert checks["config"].ok is True

    def test_no_config_is_a_single_failed_check(self):
        checks = run_checks(None, "boom")
        assert any(c.name == "config" and not c.ok and "boom" in c.detail for c in checks)
        assert not any(c.name.startswith("path:") for c in checks)

    def test_disk_estimate_scales_with_sweep(self, config_path):
        config = load_config(config_path)
        assert disk_estimate_gb(config, sweep=True) > disk_estimate_gb(config, sweep=False) > 0


# ==================================================================== #
# assemble
# ==================================================================== #
class TestAssemble:
    def test_frame_diagnostics_ratios(self, tmp_path):
        tag_dir = tmp_path / "rl_i15"
        tag_dir.mkdir()
        (tag_dir / "AA1.diagnostics.json").write_text(json.dumps({
            "scan_id": "AA1",
            "frames": [{"frame": 0, "time_min": 0.2, "counts_proxy": 10.0,
                        "noise_pct_std_before": 10.0, "noise_pct_std_after": 15.0,
                        "blood_peak_before": 2.0, "blood_peak_after": 2.5,
                        "negative_fraction_after": 0.0}]}), encoding="utf-8")
        table = assemble.frame_diagnostics_table(tmp_path)
        assert len(table) == 1
        assert table.loc[0, "noise_amplification"] == pytest.approx(1.5)
        assert table.loc[0, "peak_recovery"] == pytest.approx(1.25)
        assert table.loc[0, "tag"] == "rl_i15"

    def test_curves_for_scan_averages_runs(self):
        rows = []
        for run in (0, 1):
            for frame, (pred, truth) in enumerate(((1.0, 1.0), (3.0, 2.0))):
                rows.append({"scan_id": "AA1", "condition": "c", "run": run, "frame": frame,
                             "time_min": frame * 0.1, "predicted": pred + run, "truth": truth})
        curves, truth = assemble.curves_for_scan(pd.DataFrame(rows), "AA1")
        np.testing.assert_allclose(curves["c"][1], [1.5, 3.5])
        np.testing.assert_allclose(truth[1], [1.0, 2.0])

    def test_iteration_sweep_uses_condition_settings(self, config_path):
        config = load_config(config_path)
        metrics = pd.DataFrame({
            "condition": ["rl_retrained"] * 2 + ["baseline_retrained"] * 2,
            "scan_id": ["A", "B", "A", "B"], "rmse": [1.0, 2.0, 3.0, 4.0]})
        sweep = assemble.iteration_sweep_table(metrics, config.conditions)
        assert list(sweep["condition"]) == ["rl_retrained"]           # baseline has no PVC
        assert sweep.loc[0, "iterations"] == config.iterations_primary
        assert sweep.loc[0, "rmse"] == pytest.approx(1.5)


# ==================================================================== #
# detached (imports without tkinter)
# ==================================================================== #
class TestDetached:
    @pytest.fixture(autouse=True)
    def _gui_on_path(self):
        if str(GUI_DIR) not in sys.path:
            sys.path.insert(0, str(GUI_DIR))

    def _wait(self, pid: int, timeout: float = 10.0) -> None:
        from pvc_dlif_gui import detached
        deadline = time.time() + timeout
        while detached.pid_alive(pid) and time.time() < deadline:
            time.sleep(0.05)

    def test_launch_records_state_and_exit_code(self, tmp_path):
        from pvc_dlif_gui import detached
        state = detached.launch([sys.executable, "-c", "print('hi')"], tmp_path, "00", queue=["01"])
        assert detached.read_state(tmp_path)["queue"] == ["01"]
        self._wait(state["pid"])
        assert detached.exit_code_from_log(state["log_path"], state["pid"]) == 0
        tail = detached.LogTail(Path(state["log_path"]))
        assert "hi" in tail.read_new()

    def test_failure_exit_code_is_reported(self, tmp_path):
        from pvc_dlif_gui import detached
        state = detached.launch([sys.executable, "-c", "raise SystemExit(4)"], tmp_path, "05")
        self._wait(state["pid"])
        assert detached.exit_code_from_log(state["log_path"], state["pid"]) == 4

    def test_terminate_stops_a_running_stage(self, tmp_path):
        from pvc_dlif_gui import detached
        state = detached.launch([sys.executable, "-c", "import time; time.sleep(30)"], tmp_path, "05")
        time.sleep(0.3)
        assert detached.pid_alive(state["pid"])
        detached.terminate(state["pid"])
        self._wait(state["pid"], timeout=5)
        assert not detached.pid_alive(state["pid"])
        assert detached.exit_code_from_log(state["log_path"], state["pid"]) is None   # killed: no marker
