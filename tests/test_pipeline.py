"""Tests for the parts of the pipeline where a silent error would be costly.

The emphasis is on the things that would corrupt a result without crashing:
a PSF converted to the wrong units, folds that differ between conditions, a
statistical test applied to unpaired data, a bias/variance decomposition that
does not add up.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pvc_dlif.data.grid import GridTransform, calibrate_grid
from pvc_dlif.dlif.dataset import make_folds, stratified_val_split
from pvc_dlif.eval.bias_variance import decompose
from pvc_dlif.eval.curve_metrics import auc, compute_metrics
from pvc_dlif.eval.kinetics import patlak
from pvc_dlif.eval.stats import adjust_pvalues, paired_test, rank_biserial
from pvc_dlif.pvc.deconvolution import PETPVC_METHOD_CODE, NumpyBackend, PetpvcBackend, PVCSettings
from pvc_dlif.pvc.psf import FWHM_TO_SIGMA, PSF


# --------------------------------------------------------------------------- #
# PSF
# --------------------------------------------------------------------------- #
class TestPSF:
    def test_fwhm_to_sigma_constant(self):
        assert FWHM_TO_SIGMA == pytest.approx(1 / (2 * np.sqrt(2 * np.log(2))), rel=1e-12)

    def test_fwhm_in_voxels_uses_voxel_size(self):
        psf = PSF((0.864, 0.874, 0.994))
        # 0.5 mm voxels in-plane, 0.59675 mm axially: the measured LabPET8 grid.
        fx, fy, fz = psf.fwhm_voxels((0.5, 0.5, 0.59675))
        assert fx == pytest.approx(1.728, rel=1e-3)
        assert fy == pytest.approx(1.748, rel=1e-3)
        assert fz == pytest.approx(1.6657, rel=1e-3)

    def test_undersampled_grid_is_flagged(self):
        psf = PSF((0.864, 0.874, 0.994))
        # A 1 mm grid puts the PSF under 1.5 voxels on every axis.
        problems = psf.check_sampling((1.0, 1.0, 1.0))
        assert len(problems) == 3
        assert all("undersampled" in p for p in problems)

    def test_marginal_sampling_is_noted_but_not_flagged(self):
        """0.86 mm FWHM at 0.5 mm voxels is about 1.7 voxels: marginal, and the
        normal situation on this scanner. Flagging it every frame would bury the
        cases that actually need attention."""
        psf = PSF((0.864, 0.874, 0.994))
        assert psf.check_sampling((0.5, 0.5, 0.59675)) == []

    def test_units_mistake_is_flagged(self):
        # An FWHM given in the wrong units shows up as an absurdly wide PSF.
        problems = PSF((86.4, 87.4, 99.4)).check_sampling((0.5, 0.5, 0.6))
        assert len(problems) == 3
        assert all("units" in p for p in problems)

    def test_well_sampled_grid_is_not_flagged(self):
        assert PSF((3.0, 3.0, 3.0)).check_sampling((1.0, 1.0, 1.0)) == []

    def test_blur_conserves_total_activity(self):
        rng = np.random.default_rng(0)
        volume = rng.random((16, 16, 16)).astype(np.float32)
        blurred = PSF((2.0, 2.0, 2.0)).blur(volume, (1.0, 1.0, 1.0))
        assert blurred.sum() == pytest.approx(volume.sum(), rel=1e-3)

    def test_scaled_psf(self):
        assert PSF((1.0, 2.0, 4.0)).scaled(0.5).fwhm_mm == (0.5, 1.0, 2.0)

    def test_rejects_nonpositive_fwhm(self):
        with pytest.raises(ValueError):
            PSF((1.0, 0.0, 1.0))


# --------------------------------------------------------------------------- #
# Deconvolution
# --------------------------------------------------------------------------- #
class TestDeconvolution:
    @staticmethod
    def _phantom():
        volume = np.zeros((24, 24, 24), dtype=np.float32)
        volume[10:14, 10:14, 10:14] = 10.0
        return volume

    def test_petpvc_command_uses_the_toolbox_method_codes(self, tmp_path):
        """PETPVC has no 'RVC' method: its reblurred Van Cittert is '-p VC'.

        Sending 'RVC' fails with an unknown-method error on the first scan, so
        the mapping is pinned here.  RL stays RL.  '-k' is the deconvolution
        iteration count, '-a' the VC relaxation, '-s 0' disables the stopping
        criterion so exactly -k iterations run.
        """
        backend = PetpvcBackend.__new__(PetpvcBackend)      # skip the PATH lookup
        backend.executable = "petpvc"
        psf = PSF((0.864, 0.874, 0.994))

        rvc = backend.build_command(tmp_path / "in.nii", tmp_path / "out.nii",
                                    PVCSettings("RVC", 15, psf, alpha=1.5))
        assert rvc[rvc.index("-p") + 1] == "VC"
        assert rvc[rvc.index("-k") + 1] == "15"
        assert rvc[rvc.index("-a") + 1] == "1.5"
        assert rvc[rvc.index("-s") + 1] == "0"
        assert "RVC" not in rvc

        rl = backend.build_command(tmp_path / "in.nii", tmp_path / "out.nii",
                                   PVCSettings("RL", 15, psf))
        assert rl[rl.index("-p") + 1] == "RL"
        assert "-a" not in rl                                # alpha is a VC parameter

        assert PETPVC_METHOD_CODE == {"RL": "RL", "RVC": "VC", "VC": "VC"}

    def test_output_tags_keep_the_thesis_name(self):
        """Files and conditions are named rvc_i15, whatever PETPVC calls it."""
        assert PVCSettings("RVC", 15, PSF((1, 1, 1))).tag == "rvc_i15"

    def test_numpy_vc_and_rvc_are_the_same_algorithm(self):
        """In PETPVC 'VC' is the reblurred variant, so the fallback must agree."""
        psf = PSF((2.0, 2.0, 2.0))
        blurred = psf.blur(self._phantom(), (1.0, 1.0, 1.0))
        a = NumpyBackend().correct_volume(blurred, (1.0, 1.0, 1.0), PVCSettings("RVC", 5, psf))
        b = NumpyBackend().correct_volume(blurred, (1.0, 1.0, 1.0), PVCSettings("VC", 5, psf))
        np.testing.assert_allclose(a, b)

    def test_rl_improves_the_recovery_coefficient(self):
        """The phantom metric: mean activity in the object over its true value."""
        psf = PSF((2.0, 2.0, 2.0))
        truth = self._phantom()
        blurred = psf.blur(truth, (1.0, 1.0, 1.0))
        corrected = NumpyBackend().correct_volume(
            blurred, (1.0, 1.0, 1.0), PVCSettings("RL", 20, psf)
        )
        roi = np.s_[10:14, 10:14, 10:14]
        rc_before = blurred[roi].mean() / truth[roi].mean()
        rc_after = corrected[roi].mean() / truth[roi].mean()
        assert rc_before < 1.0                       # spill-out lowers recovery
        assert abs(rc_after - 1.0) < abs(rc_before - 1.0)

    def test_rl_stays_non_negative(self):
        psf = PSF((2.0, 2.0, 2.0))
        blurred = psf.blur(self._phantom(), (1.0, 1.0, 1.0))
        corrected = NumpyBackend().correct_volume(
            blurred, (1.0, 1.0, 1.0), PVCSettings("RL", 20, psf)
        )
        assert corrected.min() >= 0.0

    def test_rvc_also_recovers_the_peak(self):
        psf = PSF((2.0, 2.0, 2.0))
        truth = self._phantom()
        blurred = psf.blur(truth, (1.0, 1.0, 1.0))
        corrected = NumpyBackend().correct_volume(
            blurred, (1.0, 1.0, 1.0), PVCSettings("RVC", 15, psf, alpha=1.5)
        )
        assert corrected.max() > blurred.max()

    def test_more_iterations_amplify_noise(self):
        """The recovery-noise trade-off the phantom work characterised."""
        rng = np.random.default_rng(1)
        psf = PSF((2.0, 2.0, 2.0))
        blurred = psf.blur(self._phantom(), (1.0, 1.0, 1.0))
        noisy = blurred + rng.normal(0, 0.05, blurred.shape).astype(np.float32)

        backend = NumpyBackend()
        background = np.s_[0:6, 0:6, 0:6]
        few = backend.correct_volume(noisy, (1.0, 1.0, 1.0), PVCSettings("RVC", 3, psf))
        many = backend.correct_volume(noisy, (1.0, 1.0, 1.0), PVCSettings("RVC", 25, psf))
        assert many[background].std() > few[background].std()

    def test_series_applies_frame_by_frame(self):
        psf = PSF((2.0, 2.0, 2.0))
        series = np.stack([self._phantom() * scale for scale in (0.5, 1.0, 2.0)])
        corrected = NumpyBackend().correct_series(
            series, (1.0, 1.0, 1.0), PVCSettings("RVC", 5, psf)
        )
        assert corrected.shape == series.shape
        # Scaling the input scales the output: the correction is applied per frame
        # with no coupling between frames.
        assert corrected[2].max() > corrected[1].max() > corrected[0].max()

    def test_settings_reject_bad_input(self):
        psf = PSF((1.0, 1.0, 1.0))
        with pytest.raises(ValueError):
            PVCSettings("GTM", 10, psf)
        with pytest.raises(ValueError):
            PVCSettings("RL", 0, psf)

    def test_settings_tag(self):
        assert PVCSettings("rl", 15, PSF((1.0, 1.0, 1.0))).tag == "rl_i15"


# --------------------------------------------------------------------------- #
# Grid transform
# --------------------------------------------------------------------------- #
class TestGrid:
    def test_apply_produces_the_requested_shape(self):
        transform = GridTransform(crop=((10, 106), (10, 82), (10, 82)), out_shape=(96, 48, 48))
        assert transform.apply(np.random.rand(128, 92, 92)).shape == (96, 48, 48)

    def test_out_voxel_size_follows_from_the_crop(self):
        transform = GridTransform(
            crop=((0, 96), (0, 96), (0, 96)), out_shape=(96, 48, 48),
            native_voxel_mm=(0.5, 0.5, 0.6),
        )
        vx, vy, vz = transform.out_voxel_mm
        assert vx == pytest.approx(1.0)      # 96 voxels -> 48
        assert vy == pytest.approx(1.0)
        assert vz == pytest.approx(0.6)      # 96 -> 96, unchanged

    def test_calibration_recovers_a_known_crop(self):
        rng = np.random.default_rng(3)
        native = np.zeros((8, 64, 48, 48), dtype=np.float32)
        for t in range(8):
            native[t, 12:44, 8:40, 8:40] = rng.random((32, 32, 32)) * (t + 1)

        truth = GridTransform(crop=((12, 44), (8, 40), (8, 40)), out_shape=(16, 16, 16))
        reference = np.stack([truth.apply(native[t]) for t in range(8)])

        result = calibrate_grid(native, reference, (0.5, 0.5, 0.6), scan_id="TEST")
        (z0, z1), (y0, y1), (x0, x1) = result.transform.crop
        assert abs(z0 - 12) <= 2 and abs(z1 - 44) <= 2
        assert abs(y0 - 8) <= 2 and abs(y1 - 40) <= 2
        assert result.volume_correlation > 0.9

    def test_calibration_reports_a_bad_fit(self):
        rng = np.random.default_rng(4)
        native = rng.random((6, 32, 32, 32)).astype(np.float32)
        unrelated = rng.random((6, 16, 16, 16)).astype(np.float32)
        result = calibrate_grid(native, unrelated, (0.5, 0.5, 0.6), scan_id="NOISE")
        assert not result.is_trustworthy
        assert result.notes


# --------------------------------------------------------------------------- #
# Curve metrics
# --------------------------------------------------------------------------- #
class TestCurveMetrics:
    @staticmethod
    def _curve(peak: float = 10.0, shift: int = 0):
        t = np.array([0, 0.5, 0.6, 0.7, 0.9, 1.2, 1.6, 2.2, 3.0, 5.5, 10.5, 20.5, 40.5])
        y = np.array([0, 0.2, 2.0, peak, peak * 0.7, 4.0, 3.2, 2.8, 2.4, 1.8, 1.2, 0.7, 0.4])
        if shift:
            y = np.roll(y, shift)
        return t, y

    def test_perfect_prediction(self):
        t, y = self._curve()
        m = compute_metrics(y, y, t)
        assert m.rmse == pytest.approx(0.0)
        assert m.auc_ratio == pytest.approx(1.0)
        assert m.peak_height_ratio == pytest.approx(1.0)
        assert m.peak_time_error == pytest.approx(0.0)
        assert m.r2 == pytest.approx(1.0)

    def test_auc_uses_real_frame_times_not_indices(self):
        """The frames are 8 s early and 5 min late; index-based integration
        would weight them equally and be badly wrong."""
        t = np.array([0.0, 0.1, 10.0])
        y = np.array([1.0, 1.0, 1.0])
        assert auc(y, t) == pytest.approx(10.0)
        assert auc(y, np.arange(3.0)) == pytest.approx(2.0)

    def test_underestimated_peak_is_detected(self):
        t, truth = self._curve(peak=10.0)
        _, predicted = self._curve(peak=6.0)
        m = compute_metrics(predicted, truth, t)
        assert m.peak_height_ratio < 1.0
        assert m.bias < 0          # under-prediction gives negative signed bias

    def test_early_and_late_rmse_split(self):
        t, truth = self._curve()
        predicted = truth.copy()
        predicted[t < 2.5] += 1.0        # error confined to the early frames
        m = compute_metrics(predicted, truth, t, early_late_split_min=2.5)
        assert m.rmse_early > m.rmse_late
        assert m.rmse_late == pytest.approx(0.0, abs=1e-9)

    def test_metrics_frame_over_a_table(self):
        import pandas as pd

        t, y = self._curve()
        rows = []
        for condition in ("a", "b"):
            for frame in range(len(t)):
                rows.append(
                    {
                        "scan_id": "S1", "condition": condition, "model": "pretrained",
                        "fold": None, "run": None, "frame": frame,
                        "time_min": t[frame], "predicted": y[frame], "truth": y[frame],
                    }
                )
        table = metrics_table = pd.DataFrame(rows)
        from pvc_dlif.eval.curve_metrics import metrics_frame

        out = metrics_frame(table)
        assert len(out) == 2
        assert out["rmse"].max() == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Bias / variance
# --------------------------------------------------------------------------- #
class TestBiasVariance:
    def test_decomposition_adds_up(self):
        rng = np.random.default_rng(5)
        truth = np.linspace(1, 5, 20)
        predictions = truth[None, :] + rng.normal(0.3, 0.5, size=(40, 20))
        result = decompose(predictions, truth)
        assert result.mse == pytest.approx(result.bias_squared + result.variance, rel=1e-9)

    def test_pure_bias_has_no_variance(self):
        truth = np.linspace(1, 5, 10)
        predictions = np.tile(truth + 2.0, (7, 1))
        result = decompose(predictions, truth)
        assert result.variance == pytest.approx(0.0)
        assert result.bias_squared == pytest.approx(4.0)
        assert result.signed_bias == pytest.approx(2.0)

    def test_single_repeat_is_not_estimable(self):
        truth = np.linspace(1, 5, 10)
        result = decompose(truth[None, :] + 1.0, truth)
        assert result.n_repeats == 1
        assert not result.is_estimable

    def test_a_flat_mse_can_hide_a_trade(self):
        """The case the project description warns about: bias down, variance up,
        aggregate error unchanged."""
        rng = np.random.default_rng(6)
        truth = np.linspace(1, 5, 30)
        biased = np.tile(truth + 0.5, (30, 1)) + rng.normal(0, 0.05, (30, 30))
        noisy = truth[None, :] + rng.normal(0, 0.5, (30, 30))

        a, b = decompose(biased, truth), decompose(noisy, truth)
        assert a.mse == pytest.approx(b.mse, rel=0.3)
        assert a.bias_squared > b.bias_squared
        assert b.variance > a.variance


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
class TestStats:
    def test_consistent_improvement_is_detected(self):
        rng = np.random.default_rng(7)
        reference = rng.random(40) + 1.0
        condition = reference - 0.2
        result = paired_test(condition, reference, metric="rmse")
        assert result.p_value < 0.001
        assert result.median_difference < 0
        assert result.effect_size == pytest.approx(-1.0)
        assert result.improves is True

    def test_no_difference_gives_a_large_p(self):
        rng = np.random.default_rng(8)
        values = rng.random(40)
        result = paired_test(values, values, metric="rmse")
        assert result.p_value >= 0.05
        assert result.n_zero_differences == 40

    def test_unpaired_lengths_are_rejected(self):
        with pytest.raises(ValueError):
            paired_test(np.zeros(10), np.zeros(9))

    def test_missing_values_are_dropped_and_reported(self):
        a = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        b = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        result = paired_test(a, b)
        assert result.n_pairs == 4
        assert "dropped" in (result.note or "")

    def test_rank_biserial_bounds(self):
        assert rank_biserial(np.array([1.0, 2.0, 3.0])) == pytest.approx(1.0)
        assert rank_biserial(np.array([-1.0, -2.0, -3.0])) == pytest.approx(-1.0)
        assert rank_biserial(np.array([0.0, 0.0])) == pytest.approx(0.0)

    def test_bootstrap_interval_brackets_the_median(self):
        rng = np.random.default_rng(9)
        differences = rng.normal(-0.5, 0.2, 200)
        from pvc_dlif.eval.stats import bootstrap_ci

        low, high = bootstrap_ci(differences, n_boot=2000, seed=1)
        assert low < np.median(differences) < high

    def test_holm_is_monotone_and_conservative(self):
        raw = [0.01, 0.02, 0.03, 0.04]
        adjusted = adjust_pvalues(raw, "holm")
        assert all(a >= r for a, r in zip(adjusted, raw))
        assert adjusted == sorted(adjusted)

    def test_holm_is_more_conservative_than_bh(self):
        raw = [0.001, 0.01, 0.02, 0.04, 0.2]
        holm = adjust_pvalues(raw, "holm")
        bh = adjust_pvalues(raw, "bh")
        assert all(h >= b - 1e-12 for h, b in zip(holm, bh))

    def test_correction_ignores_nan(self):
        adjusted = adjust_pvalues([0.01, float("nan"), 0.02], "holm")
        assert np.isnan(adjusted[1])
        assert np.isfinite(adjusted[0]) and np.isfinite(adjusted[2])


# --------------------------------------------------------------------------- #
# Cross-validation
# --------------------------------------------------------------------------- #
class TestFolds:
    def test_folds_partition_the_dataset(self):
        ids = [f"S{i:03d}" for i in range(80)]
        folds = make_folds(ids, 10, seed=42)
        assert len(folds) == 10
        test_ids = [i for fold in folds for i in fold.test_ids]
        assert sorted(test_ids) == sorted(ids)
        assert len(set(test_ids)) == len(ids)

    def test_incomplete_runs_are_not_reused(self, tmp_path):
        """A 5-epoch pilot checkpoint must not pass for a 1000-epoch run."""
        import json
        from pvc_dlif.dlif.train import TrainSettings, _run_is_complete
        summary = tmp_path / "summary.json"
        summary.write_text(json.dumps({"epochs_trained": 5}))
        assert _run_is_complete(summary, TrainSettings(epochs=1000)) == (False, "incomplete (5 of 1000 epochs)")
        summary.write_text(json.dumps({"epochs_trained": 1000}))
        assert _run_is_complete(summary, TrainSettings(epochs=1000))[0] is True
        # Early stopping past min_epochs is a legitimate short run
        summary.write_text(json.dumps({"epochs_trained": 60}))
        assert _run_is_complete(summary, TrainSettings(epochs=1000, early_stopping=True, min_epochs=20))[0] is True

    def test_a_longer_run_is_not_a_finished_shorter_one(self, tmp_path):
        """Switching regime must not silently inherit the other one's models.

        The 2026 regime trains 1000 epochs, the 2024 one 200.  Under a plain
        ``trained >= epochs`` test every 1000-epoch checkpoint would pass as a
        finished 200-epoch run and the whole grid would be reused unchanged.
        """
        import json
        from pvc_dlif.dlif.train import TrainSettings, _run_is_complete, protocol_of

        summary = tmp_path / "summary.json"
        summary.write_text(json.dumps({"epochs_trained": 1000}))
        complete, why = _run_is_complete(summary, TrainSettings(epochs=200))
        assert complete is False and "different protocol" in why

    def test_a_recorded_protocol_must_match(self, tmp_path):
        import json
        from pvc_dlif.dlif.train import TrainSettings, _run_is_complete, protocol_of

        settings = TrainSettings(epochs=200, learning_rate=2e-4, loss="MSELoss")
        summary = tmp_path / "summary.json"
        summary.write_text(json.dumps(
            {"epochs_trained": 200, "protocol": protocol_of(settings)}))
        assert _run_is_complete(summary, settings)[0] is True

        # Same epoch budget, different learning rate and loss: not the same run.
        other = TrainSettings(epochs=200, learning_rate=1e-4, loss="WeightedMSELoss")
        complete, why = _run_is_complete(summary, other)
        assert complete is False
        assert "learning_rate" in why and "loss" in why

    def test_a_different_partition_is_not_reused(self, tmp_path):
        """n_folds 10 -> 17 renumbers the folds; fold_01 is other scans now."""
        import json
        from pvc_dlif.dlif.train import TrainSettings, _run_is_complete
        from pvc_dlif.dlif.dataset import Fold

        summary = tmp_path / "summary.json"
        summary.write_text(json.dumps({"epochs_trained": 200, "test_ids": ["A1", "A2"]}))
        settings = TrainSettings(epochs=200)
        same = Fold(index=1, train_ids=["B1"], test_ids=["A1", "A2"])
        assert _run_is_complete(summary, settings, same)[0] is True
        moved = Fold(index=1, train_ids=["B1"], test_ids=["A3", "A4"])
        complete, why = _run_is_complete(summary, settings, moved)
        assert complete is False and "partition" in why

    def test_folds_round_trip_through_json(self, tmp_path):
        """Stage 06 must score with the folds stage 05 trained on."""
        import json
        from pvc_dlif.dlif.dataset import load_folds
        folds = make_folds([f"S{i}" for i in range(12)], 3, seed=7)
        path = tmp_path / "folds.json"
        path.write_text(json.dumps([f.as_dict() for f in folds]))
        assert load_folds(path) == folds

    def test_folds_are_numbered_from_one(self):
        """The group's convention, and what ``--folds`` on stage 05 expects.
        ``--folds 0`` once selected nothing and reported success."""
        folds = make_folds([f"S{i}" for i in range(20)], 4, seed=1)
        assert [f.index for f in folds] == [1, 2, 3, 4]

    def test_train_and_test_never_overlap(self):
        folds = make_folds([f"S{i}" for i in range(50)], 5, seed=1)
        for fold in folds:
            assert not (set(fold.train_ids) & set(fold.test_ids))

    def test_folds_are_identical_across_conditions(self):
        """The paired design depends on this: the same scan must be held out in
        the same fold no matter which input representation is used."""
        ids = [f"S{i:03d}" for i in range(70)]
        a = make_folds(ids, 10, seed=42)
        b = make_folds(list(reversed(ids)), 10, seed=42)
        assert [f.test_ids for f in a] == [f.test_ids for f in b]

    def test_seed_changes_the_partition(self):
        ids = [f"S{i:03d}" for i in range(70)]
        assert [f.test_ids for f in make_folds(ids, 10, 42)] != [
            f.test_ids for f in make_folds(ids, 10, 7)
        ]

    def test_validation_split_is_disjoint(self):
        ids = [f"S{i:02d}" for i in range(40)]
        group_of = {s: f"g{i % 5}" for i, s in enumerate(ids)}
        train, val = stratified_val_split(ids, group_of, 0.15, seed=3)
        assert not (set(train) & set(val))
        assert sorted(train + val) == sorted(ids)


# --------------------------------------------------------------------------- #
# Kinetics
# --------------------------------------------------------------------------- #
class TestKinetics:
    def test_patlak_recovers_a_known_slope(self):
        t = np.linspace(0, 40, 60)
        plasma = 5 * np.exp(-0.3 * t) + 0.5
        integral = np.concatenate([[0], np.cumsum(0.5 * (plasma[1:] + plasma[:-1]) * np.diff(t))])
        ki, vb = 0.02, 0.4
        tissue = ki * integral + vb * plasma

        result = patlak(tissue, plasma, t, t_star_min=5.0)
        assert result.ki == pytest.approx(ki, rel=0.05)
        assert result.intercept == pytest.approx(vb, rel=0.1)
        assert result.r_squared > 0.99

    def test_underestimated_input_inflates_ki(self):
        """Why the input function matters: scaling it down scales K_i up."""
        t = np.linspace(0, 40, 60)
        plasma = 5 * np.exp(-0.3 * t) + 0.5
        integral = np.concatenate([[0], np.cumsum(0.5 * (plasma[1:] + plasma[:-1]) * np.diff(t))])
        tissue = 0.02 * integral + 0.4 * plasma

        true_ki = patlak(tissue, plasma, t).ki
        biased_ki = patlak(tissue, plasma * 0.8, t).ki
        assert biased_ki > true_ki

    def test_two_tissue_fit_runs(self):
        from pvc_dlif.eval.kinetics import fit_two_tissue

        t = np.linspace(0, 40, 40)
        plasma = 8 * np.exp(-0.5 * t) + 0.6
        tissue = 0.3 * np.cumsum(plasma) * (t[1] - t[0]) * 0.1 + 0.05 * plasma
        result = fit_two_tissue(tissue, plasma, t)
        assert result.converged
        assert np.isfinite(result.ki)


# --------------------------------------------------------------------------- #
# Motion
# --------------------------------------------------------------------------- #
class TestMotion:
    @staticmethod
    def _moving_series(shift_pattern):
        from scipy.ndimage import shift as ndshift

        base = np.zeros((20, 20, 20), dtype=np.float32)
        base[8:12, 8:12, 8:12] = 10.0
        return np.stack([ndshift(base, (0, 0, s), order=1) for s in shift_pattern])

    def test_static_series_shows_no_motion(self):
        from pvc_dlif.motion.detect import detect_motion

        series = self._moving_series([0.0] * 12)
        trace = detect_motion("STATIC", series, (0.5, 0.5, 0.6), com_threshold_mm=0.5)
        assert trace.max_displacement_mm < 0.05
        assert not trace.flagged

    def test_cyclic_motion_is_flagged(self):
        from pvc_dlif.motion.detect import detect_motion

        pattern = [2.0 * np.sin(i * np.pi / 3) for i in range(24)]
        series = self._moving_series(pattern)
        trace = detect_motion("CYCLIC", series, (0.5, 0.5, 0.6), com_threshold_mm=0.3)
        assert trace.flagged
        assert trace.max_displacement_mm > 0.3

    def test_rigid_correction_reduces_displacement(self):
        from pvc_dlif.motion.correct import rigid_translation_correct
        from pvc_dlif.motion.detect import detect_motion

        pattern = [0, 1, 2, 1, 0, -1, -2, -1, 0, 1, 2, 1]
        series = self._moving_series(pattern)
        before = detect_motion("M", series, (0.5, 0.5, 0.6), skip_early_frames=0)
        corrected, result = rigid_translation_correct(series, (0.5, 0.5, 0.6), reference=0)
        after = detect_motion("M", corrected, (0.5, 0.5, 0.6), skip_early_frames=0)
        assert after.max_displacement_mm < before.max_displacement_mm
        assert result.method == "rigid_translation"

    def test_smoothing_cost_detects_interpolation_blur(self):
        from pvc_dlif.motion.correct import interpolation_smoothing_cost
        from scipy.ndimage import gaussian_filter

        rng = np.random.default_rng(11)
        original = rng.random((4, 16, 16, 16)).astype(np.float32)
        smoothed = np.stack([gaussian_filter(v, 1.2) for v in original])
        cost = interpolation_smoothing_cost(original, smoothed, (0.5, 0.5, 0.6))
        assert cost["gradient_energy_ratio"] < 1.0
        assert cost["high_frequency_ratio"] < 1.0


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
class TestConfig:
    @staticmethod
    def _write(tmp_path: Path, overrides: str = "") -> Path:
        text = f"""
project: {{name: t, seed: 42}}
paths:
  dicom_root: /tmp/dicom
  dlif_data_root: /tmp/data
  dlif_repo: /tmp/repo
  work: /tmp/work
psf: {{fwhm_mm: [0.864, 0.874, 0.994]}}
pvc: {{methods: [RL, RVC], iterations_primary: 15, iterations_grid: [10, 15, 20], domain: native}}
dlif: {{model_name: DLIFNet_MAX, n_frames: 42}}
conditions:
  - {{name: baseline_pretrained, pvc: null, motion: false, model: pretrained}}
  - {{name: rl_pretrained, pvc: {{method: RL, iterations: 15}}, motion: false, model: pretrained}}
reference_condition: baseline_pretrained
{overrides}
"""
        path = tmp_path / "configs" / "thesis.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def test_loads_and_exposes_typed_values(self, tmp_path):
        from pvc_dlif.config import load_config

        cfg = load_config(self._write(tmp_path))
        assert cfg.fwhm_mm == (0.864, 0.874, 0.994)
        assert cfg.iterations_grid == [10, 15, 20]
        assert cfg.reference_condition.name == "baseline_pretrained"

    def test_condition_input_tags(self, tmp_path):
        from pvc_dlif.config import load_config

        cfg = load_config(self._write(tmp_path))
        assert cfg.condition("baseline_pretrained").input_tag == "orig"
        assert cfg.condition("rl_pretrained").input_tag == "rl_i15"

    def test_unknown_reference_is_rejected(self, tmp_path):
        from pvc_dlif.config import load_config

        with pytest.raises(ValueError, match="reference_condition"):
            load_config(self._write(tmp_path, "\n"), **{"reference_condition": "nope"})

    def _write_with_retrained(self, tmp_path):
        """The thesis config's shape: three study conditions plus a motion subset."""
        path = self._write(tmp_path)
        text = path.read_text(encoding="utf-8").replace(
            "reference_condition: baseline_pretrained",
            "  - {name: baseline_retrained, pvc: null, motion: false, model: retrained}\n"
            "  - {name: rl_retrained, pvc: {method: RL}, motion: false, model: retrained}\n"
            "  - {name: rvc_retrained, pvc: {method: RVC}, motion: false, model: retrained}\n"
            "  - {name: mc_baseline_retrained, pvc: null, motion: true, model: retrained}\n"
            "  - {name: mc_rl_retrained, pvc: {method: RL}, motion: true, model: retrained}\n"
            "reference_condition: baseline_retrained",
        )
        path.write_text(text, encoding="utf-8")
        return path

    def test_motion_conditions_are_not_in_the_retraining_grid(self, tmp_path):
        """Stage 05 trains the study's conditions; motion is stage 07's subset.

        Stage 05 has no notion of `subset`, so a motion condition trained with
        the main grid is fitted to every scan rather than the affected ones -
        and it inflates the grid by two conditions x folds x runs.
        """
        from pvc_dlif.config import load_config

        cfg = load_config(self._write_with_retrained(tmp_path))
        assert [c.name for c in cfg.retrained_conditions()] == [
            "baseline_retrained", "rl_retrained", "rvc_retrained",
        ]

    def test_motion_conditions_can_be_asked_for(self, tmp_path):
        from pvc_dlif.config import load_config

        cfg = load_config(self._write_with_retrained(tmp_path))
        assert len(cfg.retrained_conditions(include_motion=True)) == 5
        # ... and naming one explicitly overrides the default exclusion.
        named = cfg.retrained_conditions(names=["mc_rl_retrained"])
        assert [c.name for c in named] == ["mc_rl_retrained"]

    def test_status_and_stage_05_count_the_same_grid(self, tmp_path):
        """The GUI's progress fraction must denominate what stage 05 trains.

        These were two separate list comprehensions that had drifted apart:
        the status report excluded motion conditions and stage 05 did not, so
        the fraction read 5/30 while 50 trainings were queued.
        """
        from pvc_dlif.config import load_config
        from pvc_dlif.status import stage_statuses

        cfg = load_config(self._write_with_retrained(tmp_path))
        stage_05 = next(s for s in stage_statuses(cfg) if s.stage_id == "05")
        expected = len(cfg.retrained_conditions()) * cfg.get("dlif.cv.n_folds", 10) \
            * cfg.get("dlif.cv.n_runs", 10)
        assert stage_05.total == expected

    def _write_with_regimes(self, tmp_path, regime: str):
        path = self._write(tmp_path)
        text = path.read_text(encoding="utf-8").replace(
            "dlif: {model_name: DLIFNet_MAX, n_frames: 42}",
            "dlif:\n"
            "  model_name: DLIFNet_MAX\n"
            "  n_frames: 42\n"
            f"  regime: \"{regime}\"\n"
            "  regimes:\n"
            "    \"2024\":\n"
            "      cv: {n_folds: 17, n_runs: 1}\n"
            "      train: {epochs: 200, learning_rate: 0.0002, loss: MSELoss}\n"
            "    \"2026\":\n"
            "      cv: {n_folds: 10, n_runs: 10}\n"
            "      train: {epochs: 1000, learning_rate: 0.0001, loss: WeightedMSELoss}\n"
            "  cv: {n_folds: 10, n_runs: 10, validation_size: 0.15}\n"
            "  train: {batch_size: 8, device: cpu, amp: false}\n",
        )
        path.write_text(text, encoding="utf-8")
        return path

    def test_regime_2024_overrides_the_base_values(self, tmp_path):
        """The signed project description asks for the 2024 baseline protocol.

        Selecting it must reach every reader of dlif.cv.* and dlif.train.*,
        because those are spread across five scripts and the GUI.
        """
        from pvc_dlif.config import load_config

        cfg = load_config(self._write_with_regimes(tmp_path, "2024"))
        assert cfg.regime == "2024"
        assert cfg.get("dlif.cv.n_folds") == 17
        assert cfg.get("dlif.cv.n_runs") == 1
        assert cfg.get("dlif.train.epochs") == 200
        assert cfg.get("dlif.train.learning_rate") == 0.0002
        assert cfg.get("dlif.train.loss") == "MSELoss"
        # Keys the regime does not name keep the base value.
        assert cfg.get("dlif.train.batch_size") == 8
        assert cfg.get("dlif.cv.validation_size") == 0.15

    def test_regime_reaches_train_settings(self, tmp_path):
        from pvc_dlif.config import load_config
        from pvc_dlif.dlif.train import TrainSettings, _build_loss

        cfg = load_config(self._write_with_regimes(tmp_path, "2024"))
        settings = TrainSettings.from_config(cfg)
        assert (settings.epochs, settings.learning_rate) == (200, 0.0002)
        assert type(_build_loss(settings)).__name__ == "MSELoss"

    def test_regime_2026_is_the_other_protocol(self, tmp_path):
        from pvc_dlif.config import load_config

        cfg = load_config(self._write_with_regimes(tmp_path, "2026"))
        assert (cfg.get("dlif.cv.n_folds"), cfg.get("dlif.cv.n_runs")) == (10, 10)
        assert cfg.get("dlif.train.epochs") == 1000
        assert cfg.get("dlif.train.loss") == "WeightedMSELoss"

    def test_unknown_regime_is_rejected(self, tmp_path):
        from pvc_dlif.config import load_config

        with pytest.raises(ValueError, match="no such regime"):
            load_config(self._write_with_regimes(tmp_path, "2027"))

    def test_non_deconvolution_method_is_rejected(self, tmp_path):
        from pvc_dlif.config import load_config

        path = self._write(tmp_path)
        text = path.read_text(encoding="utf-8").replace("method: RL", "method: GTM")
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match="deconvolution"):
            load_config(path)


# --------------------------------------------------------------------------- #
# Model input specification
# --------------------------------------------------------------------------- #
torch = pytest.importorskip("torch")


class TestInputSpec:
    """The distributed checkpoint takes two channels and a 64x48x48 volume while
    the repository's data directory is 96x48x48 and its code defaults to one
    channel. Recovering the requirement from the weights is what stops that
    mismatch from producing a silently wrong baseline."""

    @staticmethod
    def _fixed_model(in_channels=2, kernel=(4, 3, 3), stages=5):
        import torch.nn as nn

        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.Maxpool = nn.MaxPool3d(2, 2)
                channels = [in_channels] + [8] * stages
                for i in range(stages):
                    setattr(self, f"Conv{i + 1}", nn.Conv3d(channels[i], channels[i + 1], 3, padding=1))

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = Encoder()
                self.conv3d = nn.Conv3d(8, 16, kernel_size=kernel, bias=False)

        return Model()

    @staticmethod
    def _adaptive_model():
        import torch.nn as nn

        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.Maxpool = nn.MaxPool3d(2, 2)
                self.Conv1 = nn.Conv3d(1, 8, 3, padding=1)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = Encoder()
                self.conv3d = nn.Sequential(
                    nn.Conv3d(8, 8, kernel_size=(4, 2, 2), bias=False),
                    nn.AdaptiveAvgPool3d((1, 1, 1)),
                )

        return Model()

    def test_recovers_the_fixed_input_size(self):
        from pvc_dlif.dlif.adapter import infer_input_spec

        spec = infer_input_spec(self._fixed_model())
        assert spec.in_channels == 2
        assert spec.add_average is True
        assert spec.fixed_spatial is True
        # (4, 3, 3) kernel after four pooling stages -> 64 x 48 x 48
        assert spec.spatial_shape == (64, 48, 48)

    def test_adaptive_pooling_accepts_any_size(self):
        from pvc_dlif.dlif.adapter import infer_input_spec

        spec = infer_input_spec(self._adaptive_model())
        assert spec.fixed_spatial is False
        assert spec.spatial_shape is None
        assert spec.in_channels == 1

    def test_predict_rejects_the_wrong_spatial_shape(self):
        from pvc_dlif.dlif.adapter import ModelInputSpec, predict_series

        spec = ModelInputSpec(2, (64, 48, 48), True, True, "note")
        with pytest.raises(ValueError, match="64, 48, 48"):
            predict_series(self._fixed_model(), np.zeros((4, 96, 48, 48), np.float32), spec=spec)

    def test_add_average_uses_the_late_frames(self):
        """The group's add_average averages frames 35 onward, not the whole
        series; using the whole-series mean would change the second channel."""
        from pvc_dlif.dlif.adapter import add_average_channel

        image = np.zeros((42, 4, 4, 4), np.float32)
        image[:35] = 100.0          # early frames, excluded from the average
        image[35:] = 2.0
        stacked = add_average_channel(image)
        assert stacked.shape == (2, 42, 4, 4, 4)
        assert stacked[1].mean() == pytest.approx(2.0)

    def test_add_average_is_broadcast_over_time(self):
        from pvc_dlif.dlif.adapter import add_average_channel

        rng = np.random.default_rng(12)
        image = rng.random((42, 3, 3, 3)).astype(np.float32)
        stacked = add_average_channel(image)
        assert np.allclose(stacked[1][0], stacked[1][-1])
        assert np.allclose(stacked[0], image)


# --------------------------------------------------------------------------- #
# LaTeX output
# --------------------------------------------------------------------------- #
class TestTables:
    @staticmethod
    def _comparisons():
        import pandas as pd

        return pd.DataFrame([
            {"metric": "rmse", "condition": "rl_retrained", "reference": "baseline_pretrained",
             "n_pairs": 80, "median_reference": 1.2, "median_condition": 1.05,
             "median_difference": -0.15, "mean_difference": -0.14, "ci_low": -0.22,
             "ci_high": -0.08, "statistic": 300.0, "p_value": 0.0002, "p_adjusted": 0.0008,
             "effect_size": -0.62, "effect_size_name": "rank_biserial", "test": "wilcoxon",
             "n_zero_differences": 0, "note": None, "correction": "holm"},
            {"metric": "rmse", "condition": "rvc_retrained", "reference": "baseline_pretrained",
             "n_pairs": 80, "median_reference": 1.2, "median_condition": 1.19,
             "median_difference": -0.01, "mean_difference": -0.01, "ci_low": -0.05,
             "ci_high": 0.03, "statistic": 1500.0, "p_value": 0.6, "p_adjusted": 0.6,
             "effect_size": -0.05, "effect_size_name": "rank_biserial", "test": "wilcoxon",
             "n_zero_differences": 2, "note": None, "correction": "holm"},
        ])

    def test_percent_signs_are_escaped(self):
        """A bare % starts a LaTeX comment and would swallow the rest of the line."""
        from pvc_dlif.report.tables import comparison_table

        latex = comparison_table(self._comparisons(), n_boot=10000)
        for line in latex.splitlines():
            stripped = line.replace(r"\%", "")
            assert "%" not in stripped, line

    def test_underscores_in_names_are_escaped(self):
        from pvc_dlif.report.tables import comparison_table

        latex = comparison_table(self._comparisons())
        assert r"rl\_retrained" in latex
        assert "rl_retrained" not in latex.replace(r"rl\_retrained", "")

    def test_significant_rows_are_marked(self):
        from pvc_dlif.report.tables import comparison_table

        latex = comparison_table(self._comparisons(), alpha=0.05)
        assert r"\textbf{rl\_retrained}" in latex
        assert r"\textbf{rvc\_retrained}" not in latex

    def test_bootstrap_count_reflects_what_was_run(self):
        from pvc_dlif.report.tables import comparison_table

        assert "500 resamples" in comparison_table(self._comparisons(), n_boot=500)
        assert "resamples" not in comparison_table(self._comparisons())

    def test_p_value_formatting(self):
        from pvc_dlif.report.tables import format_p

        assert format_p(0.0001) == r"$<0.001$"
        assert format_p(0.0432) == "0.043"
        assert format_p(float("nan")) == "--"

    def test_bias_variance_table_marks_unestimable_variance(self):
        import pandas as pd

        from pvc_dlif.report.tables import bias_variance_table

        frame = pd.DataFrame([
            {"condition": "baseline_pretrained", "n_scans": 80, "n_repeats": 1, "mse": 0.5,
             "bias_squared": 0.5, "variance": 0.0, "signed_bias": -0.1,
             "variance_share": float("nan"), "estimable": False},
            {"condition": "rl_retrained", "n_scans": 80, "n_repeats": 10, "mse": 0.4,
             "bias_squared": 0.25, "variance": 0.15, "signed_bias": -0.02,
             "variance_share": 0.375, "estimable": True},
        ])
        latex = bias_variance_table(frame)
        # The single deterministic model gets a dash, not a misleading zero.
        assert "0.0000" not in latex.split(r"\midrule")[1].splitlines()[0]
        assert "--" in latex


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #
class TestManifest:
    """Checks driven by what the real dataset turned out to contain: two
    reconstruction matrix sizes, two frame schedules, ``_SUV_`` unit tags, and
    Sherbrooke IDs spelled with an underscore in one place and a hyphen in
    another."""

    @staticmethod
    def _entry(scan_id, **kwargs):
        from pvc_dlif.data.manifest import ScanEntry

        base = dict(has_dicom=True, has_dlif_img=True, has_aif=True)
        base.update(kwargs)
        return ScanEntry(scan_id=scan_id, **base)

    def test_id_matching_ignores_separator_and_case(self):
        from pvc_dlif.data.manifest import _canonical

        assert _canonical("Sherbrooke-A") == _canonical("Sherbrooke_A")
        assert _canonical(" aa1 ") == _canonical("AA1")

    def test_usable_requires_every_ingredient(self):
        assert self._entry("A").usable
        assert not self._entry("B", has_dicom=False).usable
        assert not self._entry("C", has_aif=False).usable
        assert not self._entry("D", excluded=True).usable

    def test_summary_counts(self):
        from pvc_dlif.data.manifest import Manifest

        manifest = Manifest([
            self._entry("A", group="Reference"),
            self._entry("B", group="Reference"),
            self._entry("C", excluded=True, exclusion_reason="ignore list"),
            self._entry("D", has_aif=False, group="15s"),
        ])
        summary = manifest.summary()
        assert summary["n_total"] == 4
        assert summary["n_usable"] == 2
        assert summary["n_excluded"] == 1
        assert summary["n_missing_aif"] == 1
        assert summary["groups"] == {"Reference": 2}

    def test_dataframe_drops_the_per_frame_schedules(self):
        from pvc_dlif.data.manifest import Manifest

        manifest = Manifest([self._entry("A", frame_start_s=[0.0, 30.0], frame_duration_s=[30.0, 5.0])])
        assert "frame_start_s" not in manifest.to_dataframe().columns
        assert "frame_start_s" in manifest.to_dataframe(with_schedules=True).columns

    def test_round_trip_through_json(self, tmp_path):
        from pvc_dlif.data.manifest import Manifest, load_manifest

        original = Manifest(
            [self._entry("A", native_shape_zyx=[128, 92, 92], units="_SUV_", n_frames=42)],
            meta={"note": "x"},
        )
        reloaded = load_manifest(original.save(tmp_path / "manifest.json"))
        assert reloaded["A"].native_shape_zyx == [128, 92, 92]
        assert reloaded["A"].n_frames == 42
        assert reloaded.meta["note"] == "x"


class TestWindowsThatRunOffTheVolume:
    """The group's own crop windows do not always fit inside the reconstruction.

    Seven of the seventy real scans need a 96-slice axial window starting 1 to 8
    slices too late to fit in a 128-slice volume, and their distributed inputs
    carry exact zeros in the slices that fall outside.  Two things follow: the
    window has to be zero-padded rather than shortened (a shorter window would
    be resampled back up, and the input would no longer be reconstruction
    voxels), and the offset search has to be able to reach past the wall - all
    seven pinned at the axial maximum, which is what a search box that is too
    small looks like from the inside.
    """

    @staticmethod
    def _volume(shape=(60, 40, 40), seed=0):
        rng = np.random.default_rng(seed)
        z, y, x = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
        body = (((y - 20) / 9.0) ** 2 + ((x - 20) / 9.0) ** 2) < 1.0
        taper = np.exp(-((z - 34) ** 2) / (2 * 14.0 ** 2))
        return (body * taper + 0.05 * rng.random(shape)).astype(np.float64)

    def test_a_window_past_the_end_is_zero_filled_not_shortened(self):
        from pvc_dlif.data.grid import crop_with_zero_padding

        volume = self._volume(shape=(10, 4, 4))
        out = crop_with_zero_padding(volume, ((6, 14), (0, 4), (0, 4)))
        assert out.shape == (8, 4, 4)
        assert np.array_equal(out[:4], volume[6:10])
        assert np.all(out[4:] == 0)

    def test_a_window_before_the_start_is_zero_filled(self):
        from pvc_dlif.data.grid import crop_with_zero_padding

        volume = self._volume(shape=(10, 4, 4))
        out = crop_with_zero_padding(volume, ((-3, 5), (0, 4), (0, 4)))
        assert out.shape == (8, 4, 4)
        assert np.all(out[:3] == 0)
        assert np.array_equal(out[3:], volume[0:5])

    def test_apply_pads_rather_than_resampling(self):
        """A shortened window would be zoomed back up; the voxels must be exact."""
        from pvc_dlif.data.grid import GridTransform

        volume = self._volume()
        transform = GridTransform(crop=((28, 60), (8, 32), (8, 32)), out_shape=(32, 24, 24))
        shifted = GridTransform(crop=((34, 66), (8, 32), (8, 32)), out_shape=(32, 24, 24))
        out = shifted.apply(volume)
        assert out.shape == (32, 24, 24)
        assert np.allclose(out[:26], volume[34:60, 8:32, 8:32], atol=1e-6)
        assert np.all(out[26:] == 0)
        del transform

    def test_the_offset_search_reaches_past_the_wall(self):
        """The optimum here is unreachable without allowing the overhang."""
        from pvc_dlif.data.grid import crop_with_zero_padding
        from pvc_dlif.data.preprocess import fit_offset_to_reference

        volume = self._volume(seed=4)
        crop = (32, 24, 24)
        # 36 + 32 = 68 > 60: the true window runs eight voxels off the end.
        truth = (36, 8, 8)
        window = crop_with_zero_padding(
            volume, tuple((o, o + c) for o, c in zip(truth, crop)))
        assert np.all(window[-8:] == 0)

        series = np.stack([volume * w for w in np.linspace(0.5, 1.0, 42)])
        reference = np.stack([window * w for w in np.linspace(0.5, 1.0, 42)])

        fit = fit_offset_to_reference(series, reference, scan_id="OVERHANG")
        assert fit.offset == truth
        assert fit.exact
        assert fit.correlation > 0.999

    def test_a_window_that_fits_is_still_found(self):
        """The widening must not disturb the scans that were already exact."""
        from pvc_dlif.data.grid import crop_with_zero_padding
        from pvc_dlif.data.preprocess import fit_offset_to_reference

        volume = self._volume(seed=5)
        crop = (32, 24, 24)
        truth = (14, 9, 7)
        window = crop_with_zero_padding(
            volume, tuple((o, o + c) for o, c in zip(truth, crop)))

        series = np.stack([volume * w for w in np.linspace(0.5, 1.0, 42)])
        reference = np.stack([window * w for w in np.linspace(0.5, 1.0, 42)])

        fit = fit_offset_to_reference(series, reference, scan_id="INSIDE")
        assert fit.offset == truth
        assert fit.exact


class TestGridCalibrationFile:
    def test_transform_is_selected_by_matrix_size(self, tmp_path):
        """Two reconstruction fields of view need two crops; using one for the
        other would cut the wrong region out of the volume."""
        import json

        from pvc_dlif.data.grid import load_grid_transform, shape_key

        payload = {
            "by_shape": {
                "128x92x92": {"transform": {"crop": [[8, 88], [8, 56], [8, 56]],
                                            "out_shape": [64, 48, 48]}},
                "128x120x120": {"transform": {"crop": [[8, 88], [20, 100], [20, 100]],
                                              "out_shape": [64, 48, 48]}},
            },
            "majority_shape": "128x92x92",
        }
        path = tmp_path / "grid_calibration.json"
        path.write_text(json.dumps(payload), encoding="utf-8")

        assert shape_key((128, 92, 92)) == "128x92x92"
        assert load_grid_transform(path, (128, 92, 92)).crop[1] == (8, 56)
        assert load_grid_transform(path, (128, 120, 120)).crop[1] == (20, 100)

    def test_uncalibrated_matrix_size_is_an_error(self, tmp_path):
        import json

        from pvc_dlif.data.grid import load_grid_transform

        path = tmp_path / "grid_calibration.json"
        path.write_text(json.dumps({
            "by_shape": {"128x92x92": {"transform": {"crop": [[0, 8], [0, 8], [0, 8]],
                                                     "out_shape": [4, 4, 4]}}},
        }), encoding="utf-8")
        with pytest.raises(KeyError, match="128x120x120"):
            load_grid_transform(path, (128, 120, 120))


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
class TestPreprocessing:
    """The group's preprocessing is a pure crop at native resolution. Two
    properties have to hold: the window is recovered exactly from a reference,
    and it is fixed on the uncorrected data so PVC cannot change it."""

    CROP = (32, 16, 16)

    @staticmethod
    def _series(centre=(48, 32, 32), shape=(64, 48, 48), n_t=12, seed=0):
        rng = np.random.default_rng(seed)
        z, y, x = shape
        zz, yy, xx = np.ogrid[:z, :y, :x]
        body = (((zz - centre[0]) / 18.0) ** 2 + ((yy - centre[1]) / 9.0) ** 2
                + ((xx - centre[2]) / 9.0) ** 2) <= 1.0
        texture = rng.random(shape).astype(np.float32)
        # PET images are smooth; a white-noise phantom is not a fair test of a
        # method that has to localise a window in a real reconstruction.
        from scipy.ndimage import gaussian_filter

        texture = gaussian_filter(texture, 1.2)
        volume = np.where(body, 1.0 + texture, 0.05 * texture).astype(np.float32)
        return np.stack([volume * (0.2 + t) for t in range(n_t)])

    def _reference_from(self, series, offset):
        z, y, x = offset
        dz, dy, dx = self.CROP
        return series[:, z:z + dz, y:y + dy, x:x + dx].copy()

    # -- finding the animal ------------------------------------------------ #
    def test_finds_the_animal(self):
        from pvc_dlif.data.preprocess import measure_body

        m = measure_body(self._series(centre=(30, 20, 34)), (0.5, 0.5, 0.6), "T")
        assert abs(m.centre_vox[0] - 30) < 3
        assert abs(m.centre_vox[1] - 20) < 3
        assert abs(m.centre_vox[2] - 34) < 3

    # -- recovering the group's window ------------------------------------- #
    def test_recovers_a_known_window_exactly(self):
        from pvc_dlif.data.preprocess import fit_offset_to_reference

        series = self._series()
        truth = (11, 7, 19)
        fit = fit_offset_to_reference(series, self._reference_from(series, truth), scan_id="T")
        assert fit.offset == truth
        assert fit.correlation == pytest.approx(1.0, abs=1e-6)
        assert fit.scale == pytest.approx(1.0, abs=1e-4)
        assert fit.exact

    def test_recovers_the_intensity_factor(self):
        from pvc_dlif.data.preprocess import fit_offset_to_reference

        series = self._series()
        truth = (9, 5, 12)
        fit = fit_offset_to_reference(
            series, self._reference_from(series, truth) * 0.9937, scan_id="T"
        )
        assert fit.offset == truth
        assert fit.scale == pytest.approx(0.9937, rel=1e-3)

    def test_a_reference_from_elsewhere_is_not_reported_as_exact(self):
        from pvc_dlif.data.preprocess import fit_offset_to_reference

        series = self._series(seed=1)
        other = self._series(seed=2)[:, :self.CROP[0], :self.CROP[1], :self.CROP[2]]
        assert not fit_offset_to_reference(series, other, scan_id="T").exact

    # -- the plan ---------------------------------------------------------- #
    def _plan(self, scans=None):
        from pvc_dlif.data.preprocess import derive_windows

        if scans is None:
            series = self._series()
            scans = [("a", series, (0.5, 0.5, 0.6), self._reference_from(series, (11, 7, 19)))]
        return derive_windows(scans, crop_shape=self.CROP)

    def test_window_is_a_pure_crop(self):
        """Span equals output shape on every axis, so nothing is resampled."""
        plan = self._plan()
        transform = plan.transform("a", self.CROP)
        assert tuple(hi - lo for lo, hi in transform.crop) == self.CROP
        assert transform.out_shape == self.CROP

    def test_apply_is_bit_identical_to_the_crop(self):
        series = self._series()
        reference = self._reference_from(series, (11, 7, 19))
        plan = self._plan([("a", series, (0.5, 0.5, 0.6), reference)])
        assert np.allclose(plan.transform("a", self.CROP).apply_4d(series), reference, atol=1e-5)

    def test_voxel_size_is_the_reconstruction_s_own(self):
        plan = self._plan()
        assert plan.out_voxel_mm() == (0.5, 0.5, 0.6)

    def test_smaller_output_is_a_centred_sub_crop(self):
        plan = self._plan()
        small = plan.transform("a", (16, 16, 16))
        big = plan.transform("a", self.CROP)
        assert tuple(hi - lo for lo, hi in small.crop) == (16, 16, 16)
        # Centred inside the stored window, and still a pure crop.
        assert small.crop[0][0] == big.crop[0][0] + (self.CROP[0] - 16) // 2
        assert small.crop[1] == big.crop[1]

    def test_output_larger_than_the_window_is_refused(self):
        """Upsampling would stop the input voxels being reconstruction voxels."""
        plan = self._plan()
        with pytest.raises(ValueError, match="without resampling"):
            plan.transform("a", (64, 16, 16))

    def test_corrected_data_gets_the_uncorrected_window(self):
        """The guarantee the whole comparison rests on."""
        series = self._series()
        plan = self._plan([("a", series, (0.5, 0.5, 0.6), self._reference_from(series, (11, 7, 19)))])
        window = plan.transform("a", self.CROP)

        # A "corrected" series whose centre of mass has moved still gets the
        # original window, because the plan is never recomputed.
        corrected = self._series(centre=(20, 14, 40))
        assert window.apply_4d(corrected).shape == (12, *self.CROP)
        assert plan.transform("a", self.CROP).crop == window.crop

    def test_falls_back_to_body_centring_without_a_reference(self):
        series = self._series(centre=(30, 24, 24))
        plan = self._plan([("b", series, (0.5, 0.5, 0.6), None)])
        z0, z1 = plan.transform("b", self.CROP).crop[0]
        assert z1 - z0 == self.CROP[0]
        assert abs((z0 + z1) / 2 - 30) < 4          # centred on the animal
        assert "b" not in plan.fits                 # nothing was matched

    def test_plan_round_trips(self, tmp_path):
        from pvc_dlif.data.preprocess import load_plan

        plan = self._plan()
        reloaded = load_plan(plan.save(tmp_path / "plan.json"))
        assert reloaded.transform("a", self.CROP).crop == plan.transform("a", self.CROP).crop
        assert reloaded.crop_shape == plan.crop_shape
        assert reloaded.voxel_mm == plan.voxel_mm

    def test_unknown_scan_is_an_error(self):
        with pytest.raises(KeyError, match="nope"):
            self._plan().transform("nope", self.CROP)


class TestDeviceAugmentation:
    """The GPU-side augmentation must match the NumPy path in distribution."""

    def test_noise_is_zero_mean_and_poisson_scaled(self):
        import torch
        from pvc_dlif.dlif.train import augment_on_device
        x = torch.full((2, 1, 42, 8, 8, 8), 4.0)
        g = torch.Generator(); g.manual_seed(1)
        y = augment_on_device(x, g, poisson_noise=True, random_flip=False, add_average=False)
        noise = y - x
        assert abs(float(noise.mean())) < 0.05                       # Poisson(rate) - rate
        assert y.shape == x.shape

    def test_average_channel_is_late_mean_of_the_noisy_image(self):
        import torch
        from pvc_dlif.dlif.train import augment_on_device
        x = torch.rand(3, 1, 42, 6, 6, 6) * 3
        g = torch.Generator(); g.manual_seed(2)
        y = augment_on_device(x, g, poisson_noise=True, random_flip=True, add_average=True)
        assert y.shape[1] == 2
        assert torch.allclose(y[:, 1, 0], y[:, 0, 35:].mean(1), atol=1e-5)

    def test_flip_is_in_plane_only(self):
        import torch
        from pvc_dlif.dlif.train import augment_on_device
        x = torch.arange(2 * 1 * 3 * 2 * 4 * 4, dtype=torch.float32).reshape(2, 1, 3, 2, 4, 4)
        g = torch.Generator(); g.manual_seed(3)
        y = augment_on_device(x, g, poisson_noise=False, random_flip=True, add_average=False)
        for b in range(2):
            same = torch.equal(y[b], x[b])
            flipped = torch.equal(y[b], torch.flip(x[b], dims=(-1, -2)))
            assert same or flipped
            assert torch.equal(y[b].sum(dim=(-1, -2)), x[b].sum(dim=(-1, -2)))   # z and t untouched

    def test_training_loop_runs_with_device_augmentation(self, tmp_path):
        """The whole train_condition path with augment_on_device forced on (CPU)."""
        import pickle
        import numpy as np
        from pvc_dlif.data import pkl_io
        from pvc_dlif.dlif.dataset import make_folds
        from pvc_dlif.dlif.train import TrainSettings, train_condition
        import torch.nn as nn

        ids = [f"S{i}" for i in range(6)]
        data = tmp_path / "data"; (data / "AIF_SUV").mkdir(parents=True)
        for scan in ids:
            pkl_io.save_img(data, scan, np.random.rand(42, 8, 8, 8).astype(np.float64), np.arange(42.0),
                            shape=(8, 8, 8))
            with open(data / "AIF_SUV" / f"AIF_{scan}.pkl", "wb") as f:
                pickle.dump({"AIF_A_int": np.random.rand(42), "AIF_t_int": np.arange(42.0)}, f)

        class Tiny(nn.Module):
            def __init__(self):
                super().__init__(); self.lin = nn.Linear(8 * 8 * 8, 1)
            def forward(self, x):                      # (B, C, T, Z, Y, X) -> (B, T)
                return self.lin(x[:, 0].flatten(2)).squeeze(-1)

        settings = TrainSettings(epochs=2, batch_size=3, device="cpu", amp=False)
        results = train_condition(
            model_factory=Tiny, data_root=data, aif_root=data, scan_ids=ids,
            group_of={s: None for s in ids}, out_dir=tmp_path / "models", settings=settings,
            n_folds=2, n_runs=1, validation_size=0.34, img_shape=(8, 8, 8),
            augmentation={"poisson_noise": True, "random_flip": True, "add_average": False},
            seed=1, resume=False, folds=make_folds(ids, 2, 1), augment_on_device=True,
        )
        assert len(results) == 2 and all(r.epochs_trained == 2 for r in results)


class TestDatasetCopies:
    """The device path hands out a view of the cache; nothing may mutate it."""

    @staticmethod
    def _dataset(tmp_path, augment_on_device):
        import pickle
        from pvc_dlif.data import pkl_io
        from pvc_dlif.dlif.dataset import DlifDataset

        data = tmp_path / "data"
        (data / "AIF_SUV").mkdir(parents=True, exist_ok=True)
        pkl_io.save_img(data, "A1", np.arange(42 * 8 * 8 * 8, dtype=np.float64).reshape(42, 8, 8, 8),
                        np.arange(42.0), shape=(8, 8, 8))
        with open(data / "AIF_SUV" / "AIF_A1.pkl", "wb") as f:
            pickle.dump({"AIF_A_int": np.ones(42), "AIF_t_int": np.arange(42.0)}, f)
        return DlifDataset(data, ["A1"], aif_root=data, img_shape=(8, 8, 8), mode="train",
                           augment=True, augment_on_device=augment_on_device)

    def test_device_path_returns_a_view_and_leaves_the_cache_alone(self, tmp_path):
        dataset = self._dataset(tmp_path, augment_on_device=True)
        first = dataset[0]["INPUT"]                          # also fills the cache
        before = np.array(dataset._cache["A1"][0], copy=True)
        # Two reads must give the same values: no augmentation, no mutation.
        second = dataset[0]["INPUT"]
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(dataset._cache["A1"][0], before)
        assert first.base is not None                      # a view, not a copy

    def test_cpu_path_augments_a_copy_not_the_cache(self, tmp_path):
        dataset = self._dataset(tmp_path, augment_on_device=False)
        dataset[0]                                          # first access fills the cache
        before = np.array(dataset._cache["A1"][0], copy=True)
        dataset[0]
        dataset[0]
        np.testing.assert_array_equal(dataset._cache["A1"][0], before)
