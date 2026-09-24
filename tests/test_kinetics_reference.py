"""Check the kinetic models against the group's reference implementation.

``pvc_dlif.eval.kinetics`` follows the conventions of ``pyPET``
(S. Kuttner, https://github.com/Kuttner/pyPET), the library behind the
kinetic-modelling results in Kuttner et al., EJNMMI Research 16:42 (2026) and
Salomonsen et al., EJNMMI Research 16:65 (2026).  It is a separate
implementation, so "follows the conventions" is a claim that has to be checked
rather than asserted: the tests here fit the same curves with both and require
the parameters to agree.

``pyPET`` is GPL-3 and this project is MIT, so it is neither vendored nor
depended on.  The comparison runs against a clone and against the study data,
both supplied by environment variable, and skips when either is absent::

    git clone https://github.com/Kuttner/pyPET ~/pyPET
    PYPET_PATH=~/pyPET DLIF_DATA=/path/to/data pytest tests/test_kinetics_reference.py

The comparison uses real scans rather than synthetic ones on purpose.  Curves
generated from the forward model and then fitted with that same model leave the
optimiser in a shallow, poorly conditioned minimum where ``K_1`` and ``v_b``
trade off against each other; two optimisers then land in different places for
reasons that have nothing to do with the conventions under test.  Measured
curves constrain the fit, and there the two implementations agree to within a
fraction of a per cent.

One convention of the reference is worth stating because it is a trap: its
``patlak`` interpolates with ``TIME_FRAMES`` in whatever units the time vector
carries, and converts to minutes only afterwards.  A time vector in minutes
therefore produces 2.5-*minute* frames and, on a 40-minute acquisition, too few
samples for the break-point search to run at all -- it raises an
``UnboundLocalError`` rather than reporting anything.  The time vectors below
go in as seconds.
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

from pvc_dlif.eval import kinetics

# Agreement thresholds.  The two implementations accumulate the convolution and
# the cumulative integral differently, so exact equality is not expected; these
# are tight enough that a real divergence in convention -- a different break
# point, a different grid, a constrained k4 -- would exceed them by orders of
# magnitude.  Observed on the study data: 0.08 % on the Patlak slope and 0.01 %
# on the rate constants.
KI_TOLERANCE = 0.01          # 1 % on the reported net influx
RATE_TOLERANCE = 0.02        # 2 % on the individual rate constants

#: Scans the comparison runs on.  A handful is enough -- the question is
#: whether two estimators agree, not how they behave across the cohort -- and
#: these span the range of peak recovery in the dataset.
COMPARISON_SCANS = ("Z3", "AX2", "AD3", "O1", "AB1")

#: The VOIs the models are valid for.  See ``kinetics.KINETIC_VOIS``.
COMPARISON_VOIS = ("Brain", "Myocardium")


def _reference():
    """Import pyPET's compartment models, or skip."""
    root = os.environ.get("PYPET_PATH")
    candidates = [Path(root).expanduser()] if root else []
    candidates += [Path.home() / "pyPET"]
    for path in candidates:
        if (path / "Tissue_compartment_modeling.py").exists():
            sys.path.insert(0, str(path))
            import Tissue_compartment_modeling as reference

            return reference
    pytest.skip("pyPET not found; set PYPET_PATH to a clone of the repository")


def _study_curves():
    """Load (time, input function, tissue curves) for the comparison scans."""
    root = os.environ.get("DLIF_DATA")
    if not root:
        pytest.skip("set DLIF_DATA to the directory holding AIF_SUV and VOI_SUV")
    data = Path(root).expanduser()
    if not (data / "AIF_SUV").is_dir() or not (data / "VOI_SUV").is_dir():
        pytest.skip(f"no AIF_SUV / VOI_SUV under {data}")

    out = []
    for scan in COMPARISON_SCANS:
        aif_path = data / "AIF_SUV" / f"AIF_{scan}.pkl"
        voi_path = data / "VOI_SUV" / f"VOI_{scan}.pkl"
        if not aif_path.exists() or not voi_path.exists():
            continue
        aif = pickle.loads(aif_path.read_bytes())
        voi = pickle.loads(voi_path.read_bytes())
        tissues = {name: np.asarray(voi["VOI_A"][name], float)
                   for name in COMPARISON_VOIS if name in voi["VOI_A"]}
        out.append((scan,
                    np.asarray(aif["AIF_t_int"], float),
                    np.asarray(aif["AIF_A_int"], float),
                    tissues))

    if not out:
        pytest.skip(f"none of {COMPARISON_SCANS} found under {data}")
    return out


def test_patlak_matches_reference():
    reference = _reference()

    largest = 0.0
    for scan, t, c_p, tissues in _study_curves():
        for name, c_t in tissues.items():
            ki_reference, _ = reference.patlak(
                c_p.copy(), t.copy() * 60.0, c_t.copy(), t.copy() * 60.0,
                INTERPOLATE=True, TIME_FRAMES=kinetics.FRAME_SECONDS,
            )
            ours = kinetics.patlak(c_t, c_p, t)

            assert np.isfinite(ours.ki), f"{scan}/{name}: no Patlak slope"
            assert ours.ki == pytest.approx(ki_reference, rel=KI_TOLERANCE), (
                f"{scan}/{name}: Ki {ours.ki:.6f} against {ki_reference:.6f}"
            )
            largest = max(largest, abs(ours.ki - ki_reference) / abs(ki_reference))

    print(f"\nlargest relative difference in Patlak Ki: {largest:.3%}")


def test_two_tissue_matches_reference():
    reference = _reference()

    largest: dict[str, float] = {}
    for scan, t, c_p, tissues in _study_curves():
        for name, c_t in tissues.items():
            parameters, ki_reference, _, _ = reference.twoTCMrev(
                c_p.copy(), t.copy() * 60.0, c_t.copy(), t.copy() * 60.0,
                INTERPOLATE=True, TIME_FRAMES=kinetics.FRAME_SECONDS,
            )
            k1_ref, k2_ref, vb_ref, k3_ref, k4_ref = parameters
            ours = kinetics.fit_two_tissue(c_t, c_p, t, irreversible=False)

            assert ours.converged, f"{scan}/{name}: fit did not converge"
            assert ours.ki == pytest.approx(ki_reference, rel=KI_TOLERANCE), (
                f"{scan}/{name}: Ki {ours.ki:.6f} against {ki_reference:.6f}"
            )
            for label, mine, theirs in (
                ("K1", ours.k1, k1_ref), ("k2", ours.k2, k2_ref),
                ("k3", ours.k3, k3_ref), ("k4", ours.k4, k4_ref),
                ("vb", ours.vb, vb_ref),
            ):
                assert mine == pytest.approx(theirs, rel=RATE_TOLERANCE, abs=1e-4), (
                    f"{scan}/{name}: {label} {mine:.6f} against {theirs:.6f}"
                )
                if abs(theirs) > 1e-6:
                    largest[label] = max(largest.get(label, 0.0),
                                         abs(mine - theirs) / abs(theirs))

    print("\nlargest relative difference per parameter: "
          + ", ".join(f"{k} {v:.3%}" for k, v in sorted(largest.items())))


# --------------------------------------------------------------------------- #
# Structural checks.  These do not need the reference or the study data: they
# assert that the conventions are in force at all, which is what would quietly
# regress if someone restored the old defaults.
# --------------------------------------------------------------------------- #

def _curves(seed: int = 0):
    """An input function and a tissue curve on the study's framing."""
    rng = np.random.default_rng(seed)

    # 1x30 s, 24x5 s, 9x20 s, 8x300 s -- the acquisition framing.
    durations = [30.0] + [5.0] * 24 + [20.0] * 9 + [300.0] * 8
    edges = np.concatenate([[0.0], np.cumsum(durations)])
    t = ((edges[:-1] + edges[1:]) / 2.0) / 60.0

    lag = np.clip(t - 0.5, 0.0, None)
    c_p = (28.0 * lag * np.exp(-12.0 * lag)
           + 2.2 * (1.0 - np.exp(-12.0 * lag)) * np.exp(-0.06 * lag)
           + 1.1 * (1.0 - np.exp(-12.0 * lag)) * np.exp(-0.011 * lag))
    c_p = np.clip(c_p + rng.normal(0.0, 0.01, c_p.shape), 0.0, None)

    c_t = kinetics._two_tissue_model(t, c_p, 0.25, 0.50, 0.065, 0.013, 0.05)
    return t, c_p, c_t


def test_break_point_is_fitted_not_fixed():
    """The Patlak break point is estimated from the curve, not assumed."""
    t, c_p, c_t = _curves()
    result = kinetics.patlak(c_t, c_p, t)

    assert np.isfinite(result.t_star_min)
    # It lands in the early part of the acquisition and is reported, so a later
    # reader can see which part of the curve the slope came from.  The search
    # covers a sixth of the *usable* samples, which start after the bolus
    # arrives rather than at t = 0, so the bound here is deliberately loose.
    assert 0.0 < result.t_star_min < t[-1] / 4.0
    assert result.t_star_min != pytest.approx(5.0, abs=1e-6), (
        "the break point equals the old fixed default, which suggests the "
        "search is not running"
    )


def test_uniform_resampling_changes_the_patlak_slope():
    """Guard the claim that frame-duration weighting is not cosmetic.

    Fitted on the acquisition framing, the eight five-minute tail frames carry
    the same weight as eight five-second frames near the bolus.  If resampling
    made no difference there would be no reason to do it, and this is the test
    that would go quiet first.
    """
    t, c_p, c_t = _curves()
    resampled = kinetics.patlak(c_t, c_p, t, t_star_min=5.0, resample=True).ki
    native = kinetics.patlak(c_t, c_p, t, t_star_min=5.0, resample=False).ki

    assert np.isfinite(resampled) and np.isfinite(native)
    assert abs(resampled - native) / abs(resampled) > 1e-3


def test_two_tissue_is_reversible_by_default():
    """``k_4`` is free unless the caller asks for the constrained fit."""
    t, c_p, c_t = _curves()

    free = kinetics.fit_two_tissue(c_t, c_p, t)
    fixed = kinetics.fit_two_tissue(c_t, c_p, t, irreversible=True)

    assert not free.irreversible
    assert fixed.irreversible
    assert fixed.k4 == 0.0
    assert free.k4 > 0.0, "a curve simulated with washout fitted k4 = 0"


def test_liver_is_not_a_default_kinetic_target():
    """Liver is a blood surrogate in the reference pipeline, not a tissue."""
    t, c_p, c_t = _curves()
    curves = {"Brain": c_t, "Liver": c_t * 1.3}

    rows = kinetics.compare_kinetics(c_p, c_p, curves, t, models=("patlak",))
    assert rows, "nothing was fitted"
    assert not any(str(row["voi"]).lower() == "liver" for row in rows)

    everything = kinetics.compare_kinetics(c_p, c_p, curves, t,
                                           models=("patlak",), vois=None)
    assert any(str(row["voi"]).lower() == "liver" for row in everything)
