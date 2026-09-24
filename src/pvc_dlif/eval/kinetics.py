"""Downstream kinetic modelling: what the input function is actually for.

An input function is a means, not an end.  A change in RMSE only matters if it
propagates to the parameters someone would report, so every condition's
predicted curve is also pushed through the models used with dynamic FDG data
and the resulting parameters are compared against the same models driven by the
arterial ground truth.

Two models:

* **Patlak** -- a graphical method for irreversibly trapped tracer.  For
  ``t > t*``, ``C_T(t) / C_P(t)`` is linear in ``\\int_0^t C_P / C_P(t)``, with
  slope ``K_i``.  It is the standard readout for FDG and it is sensitive to the
  *area* of the input function, which is exactly what spill-out bias distorts.
* **Two-tissue compartment model** -- fits ``K_1, k_2, k_3, k_4`` and a blood
  volume fraction by non-linear least squares.  Sensitive to the height and
  timing of the bolus peak as well as the area, so it picks up errors that
  Patlak averages away.

Tissue curves come from the group's ``VOI_SUV`` files, so the tissue side is
identical across conditions and only the input function changes.

Conventions follow the group's own reference implementation, ``pyPET``
(S. Kuttner, https://github.com/Kuttner/pyPET), as used in Kuttner et al.,
EJNMMI Research 16:42 (2026) and Salomonsen et al., EJNMMI Research 16:65
(2026).  Three of them are not cosmetic and are set out here because they
change the numbers:

1. **Uniform resampling before fitting.**  The acquisition framing runs from
   5 s frames through the bolus to 5 min frames in the tail.  Fitted on that
   grid, least squares counts a 5 s frame and a 5 min frame as one observation
   each, so the tail -- which carries most of the area and most of the Patlak
   fit -- is under-weighted by two orders of magnitude relative to its
   duration.  Resampling both curves onto a uniform grid first makes every
   sample cover the same amount of time, which is the frame-duration weighting
   the design needs, obtained by construction rather than by a weight vector.
2. **The Patlak break point is fitted, not assumed.**  ``t*`` is where the
   reversible compartments have equilibrated, and it differs between scans and
   between tissues.  Fixing it at 5 min forces one guess on every curve; here
   two lines are fitted on either side of a candidate break point and the point
   minimising their summed error is taken, searched over the first sixth of the
   acquisition.
3. **The two-tissue model is reversible by default.**  ``k_4 = 0`` is the usual
   FDG shorthand, but the reference implementation leaves ``k_4`` free for
   brain and myocardium and so does this.  ``irreversible=True`` restores the
   constrained fit.

This module is an independent implementation of those conventions, not a copy
of ``pyPET``; ``tests/test_kinetics_reference.py`` checks the two against each
other on the same curves when ``pyPET`` is importable.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Mapping, Sequence

import numpy as np

from ..logging_utils import get_logger

LOGGER = get_logger(__name__)

__all__ = [
    "PatlakResult", "TwoTissueResult", "patlak", "fit_two_tissue",
    "compare_kinetics", "resample_uniform",
]

#: Uniform sampling interval used before fitting, in seconds.  2.5 s is the
#: reference implementation's default: short enough to resolve the bolus, which
#: the native framing samples at 5 s, and coarse enough that a 40-minute scan
#: stays under a thousand samples.
FRAME_SECONDS = 2.5

#: Values at or below this are treated as zero in the input function.  The
#: Patlak transform divides by ``C_P``; the pre-injection frames sit at the
#: noise floor and, left alone, produce a handful of enormous ratios that
#: dominate the regression.
INPUT_FLOOR = 1e-4


def resample_uniform(
    curves: Sequence[Sequence[float]],
    time_min: Sequence[float],
    frame_seconds: float = FRAME_SECONDS,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Linearly resample several curves onto a shared uniform time grid.

    Returns the resampled curves and the grid.  The grid spans the original
    time range, so no extrapolation is involved; the point is the spacing, not
    the extent.
    """
    t = np.asarray(time_min, dtype=np.float64).ravel()
    if t.size < 2:
        return [np.asarray(c, dtype=np.float64).ravel() for c in curves], t

    step = float(frame_seconds) / 60.0
    n = max(int(round((t[-1] - t[0]) / step)), 2)
    grid = np.linspace(t[0], t[-1], n)

    out: list[np.ndarray] = []
    for curve in curves:
        c = np.asarray(curve, dtype=np.float64).ravel()
        m = min(c.size, t.size)
        out.append(np.interp(grid, t[:m], c[:m]))
    return out, grid


def _cumulative_integral(curve: np.ndarray, time_min: np.ndarray) -> np.ndarray:
    """Cumulative trapezoidal integral, starting from zero at the first sample."""
    integral = np.zeros_like(curve, dtype=np.float64)
    if curve.size > 1:
        increments = 0.5 * (curve[1:] + curve[:-1]) * np.diff(time_min)
        integral[1:] = np.cumsum(increments)
    return integral


@dataclass
class PatlakResult:
    """Patlak graphical analysis output."""

    ki: float                 # net influx rate, 1/min
    intercept: float          # distribution volume-like intercept
    r_squared: float
    n_points: int
    t_star_min: float         # the break point actually used

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def patlak(
    tissue: Sequence[float],
    plasma: Sequence[float],
    time_min: Sequence[float],
    t_star_min: float | None = None,
    resample: bool = True,
    frame_seconds: float = FRAME_SECONDS,
    search_fraction: float = 1.0 / 6.0,
) -> PatlakResult:
    """Patlak slope ``K_i`` from a tissue curve and an input function.

    With ``t_star_min`` left at ``None`` the break point is fitted: two lines
    are regressed on either side of each candidate point and the one with the
    smallest summed residual is kept, searched over the first
    ``search_fraction`` of the acquisition.  Passing a value fixes it instead,
    which is what the sensitivity check in the thesis uses.

    ``resample`` puts both curves on a uniform grid first, so that each sample
    of the regression covers the same span of time.  Turning it off fits on the
    acquisition framing, where the long tail frames are under-weighted; it
    exists so the difference can be quantified rather than asserted.
    """
    c_t = np.asarray(tissue, dtype=np.float64).ravel()
    c_p = np.asarray(plasma, dtype=np.float64).ravel()
    t = np.asarray(time_min, dtype=np.float64).ravel()

    n = min(c_t.size, c_p.size, t.size)
    c_t, c_p, t = c_t[:n].copy(), c_p[:n].copy(), t[:n]

    if resample:
        (c_p, c_t), t = resample_uniform((c_p, c_t), t, frame_seconds)

    # Below the floor the transform is meaningless, and the ratio it would
    # produce is an artefact of dividing by noise rather than a data point.
    c_p = np.where(c_p < INPUT_FLOOR, 0.0, c_p)

    integral = _cumulative_integral(c_p, t)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = integral / c_p
        y = c_t / c_p

    valid = np.isfinite(x) & np.isfinite(y) & (c_p > 0.0)
    if valid.sum() < 4:
        return PatlakResult(float("nan"), float("nan"), float("nan"), int(valid.sum()),
                            float("nan"))

    x, y, t_valid = x[valid], y[valid], t[valid]

    if t_star_min is not None:
        start = int(np.searchsorted(t_valid, float(t_star_min)))
        if x.size - start < 3:
            return PatlakResult(float("nan"), float("nan"), float("nan"),
                                int(x.size - start), float(t_star_min))
        return _patlak_fit(x, y, start, float(t_valid[start]))

    # Fitted break point.  A line before it and a line after it; the split that
    # explains both halves best is the one where the curve stops bending.
    upper = max(int(x.size * float(search_fraction)), 4)
    best_start, best_error = None, np.inf
    for start in range(2, min(upper, x.size - 3)):
        early = _line_error(x[:start], y[:start])
        late = _line_error(x[start:], y[start:])
        if not np.isfinite(early) or not np.isfinite(late):
            continue
        if early + late < best_error:
            best_error, best_start = early + late, start

    if best_start is None:
        return PatlakResult(float("nan"), float("nan"), float("nan"), 0, float("nan"))

    return _patlak_fit(x, y, best_start, float(t_valid[best_start]))


def _line_error(x: np.ndarray, y: np.ndarray) -> float:
    """Root sum of squared residuals of a straight-line fit."""
    if x.size < 2 or np.ptp(x) < 1e-12:
        return float("inf")
    slope, intercept = np.polyfit(x, y, 1)
    return float(np.sqrt(np.sum((y - (slope * x + intercept)) ** 2)))


def _patlak_fit(x: np.ndarray, y: np.ndarray, start: int, t_star: float) -> PatlakResult:
    xs, ys = x[start:], y[start:]
    slope, intercept = np.polyfit(xs, ys, 1)
    fitted = slope * xs + intercept
    ss_res = float(np.sum((ys - fitted) ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    return PatlakResult(
        ki=float(slope),
        intercept=float(intercept),
        r_squared=float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan"),
        n_points=int(xs.size),
        t_star_min=t_star,
    )


def _two_tissue_model(
    t: np.ndarray,
    c_p: np.ndarray,
    k1: float,
    k2: float,
    k3: float,
    k4: float,
    vb: float,
) -> np.ndarray:
    """Two-tissue compartment tissue curve, evaluated by convolution.

    The impulse response is the standard bi-exponential.  On a uniform grid the
    convolution is a discrete one scaled by the step; on a non-uniform grid it
    falls back to trapezoidal accumulation, which is slower but does not
    silently assume a spacing the samples do not have.
    """
    alpha = k2 + k3 + k4
    disc = max(alpha ** 2 - 4.0 * k2 * k4, 0.0)
    root = np.sqrt(disc)
    theta1 = (alpha - root) / 2.0
    theta2 = (alpha + root) / 2.0

    denom = theta2 - theta1
    if abs(denom) < 1e-12:
        coeff1, coeff2 = k1, 0.0
    else:
        coeff1 = k1 * (k3 + k4 - theta1) / denom
        coeff2 = k1 * (theta2 - k3 - k4) / denom

    steps = np.diff(t)
    uniform = steps.size > 0 and float(np.ptp(steps)) < 1e-9

    if uniform:
        dt = float(steps[0])
        lag = t - t[0]
        response = coeff1 * np.exp(-theta1 * lag) + coeff2 * np.exp(-theta2 * lag)
        out = np.convolve(response, c_p)[: t.size] * dt
    else:
        out = np.zeros_like(t, dtype=np.float64)
        for i in range(1, t.size):
            tau = t[: i + 1]
            response = (coeff1 * np.exp(-theta1 * (t[i] - tau))
                        + coeff2 * np.exp(-theta2 * (t[i] - tau)))
            out[i] = np.trapezoid(response * c_p[: i + 1], tau)

    return (1.0 - vb) * out + vb * c_p


@dataclass
class TwoTissueResult:
    """Two-tissue compartment fit output."""

    k1: float
    k2: float
    k3: float
    k4: float
    vb: float
    ki: float                 # K1*k3/(k2+k3), the net influx rate
    rmse: float
    converged: bool
    irreversible: bool = False
    #: True when a fitted parameter sits on one of its bounds.  ``curve_fit``
    #: reports such a fit as successful, and it is -- the optimiser did what it
    #: was asked -- but the parameter is not an estimate: the data did not
    #: determine it and the bound did.  In myocardium this happens in roughly
    #: one fit in six, almost always ``k_4`` running to its ceiling, where the
    #: reversible model collapses towards a one-tissue model.  Rows carrying
    #: this flag are reported, not silently dropped, so the reader can see how
    #: many there were.
    bounds_hit: bool = False
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fit_two_tissue(
    tissue: Sequence[float],
    plasma: Sequence[float],
    time_min: Sequence[float],
    irreversible: bool = False,
    initial: Sequence[float] | None = None,
    resample: bool = True,
    frame_seconds: float = FRAME_SECONDS,
) -> TwoTissueResult:
    """Fit ``K_1, k_2, k_3, (k_4), v_b`` by non-linear least squares.

    Reversible by default: the reference implementation leaves ``k_4`` free for
    brain and myocardium, and a model that cannot represent washout will absorb
    any washout present into the other rate constants.  ``irreversible=True``
    fixes ``k_4 = 0``, the conventional FDG simplification.

    ``K_i`` is reported as ``K_1 k_3 / (k_2 + k_3)`` in both cases, which is the
    net influx the reference implementation reports; for a reversible fit that
    is an approximation, valid to the extent ``k_4`` is small.

    ``v_b`` is bounded at 1.0 rather than at a physiological blood fraction.
    In myocardium it routinely fits to 0.3-0.5, which is not blood volume but
    spill-in from the left ventricle -- a partial volume effect the model has
    no other term for.  Capping it lower would not remove that signal, it would
    push it into ``K_1``.
    """
    from scipy.optimize import curve_fit

    c_t = np.asarray(tissue, dtype=np.float64).ravel()
    c_p = np.asarray(plasma, dtype=np.float64).ravel()
    t = np.asarray(time_min, dtype=np.float64).ravel()
    n = min(c_t.size, c_p.size, t.size)
    c_t, c_p, t = c_t[:n].copy(), c_p[:n].copy(), t[:n]

    if n < 6:
        return TwoTissueResult(*([float("nan")] * 6), float("nan"), False,
                               irreversible, False, "too few samples")

    if resample:
        (c_p, c_t), t = resample_uniform((c_p, c_t), t, frame_seconds)

    c_p = np.where(c_p < INPUT_FLOOR, 0.0, c_p)

    p0 = list(initial) if initial else [0.1, 0.1, 0.1, 0.1, 0.0]
    if irreversible:
        def model(tt, k1, k2, k3, vb):
            return _two_tissue_model(tt, c_p, k1, k2, k3, 0.0, vb)

        p0 = [p0[0], p0[1], p0[2], p0[4]]
        bounds = ([1e-6, 1e-6, 0.0, 0.0], [5.0, 5.0, 5.0, 1.0])
    else:
        def model(tt, k1, k2, k3, k4, vb):
            return _two_tissue_model(tt, c_p, k1, k2, k3, k4, vb)

        p0 = [p0[0], p0[1], p0[2], p0[3], p0[4]]
        bounds = ([1e-6, 1e-6, 0.0, 0.0, 0.0], [5.0, 5.0, 5.0, 5.0, 1.0])

    try:
        popt, _ = curve_fit(model, t, c_t, p0=p0, bounds=bounds, maxfev=20000)
        converged, message = True, ""
    except Exception as exc:  # noqa: BLE001 - a failed fit is a result, not a crash
        LOGGER.debug("two-tissue fit failed: %s", exc)
        return TwoTissueResult(*([float("nan")] * 6), float("nan"), False,
                               irreversible, False, str(exc))

    # A parameter resting on its bound was set by the bound, not by the data.
    upper = np.asarray(bounds[1], dtype=np.float64)
    bounds_hit = bool(np.any(np.asarray(popt, dtype=np.float64) >= upper * (1 - 1e-3)))

    if irreversible:
        k1, k2, k3, vb = popt
        k4 = 0.0
    else:
        k1, k2, k3, k4, vb = popt

    fitted = model(t, *popt)
    rmse = float(np.sqrt(np.mean((fitted - c_t) ** 2)))
    ki = float(k1 * k3 / (k2 + k3)) if (k2 + k3) > 1e-12 else float("nan")

    return TwoTissueResult(
        k1=float(k1), k2=float(k2), k3=float(k3), k4=float(k4), vb=float(vb),
        ki=ki, rmse=rmse, converged=converged, irreversible=bool(irreversible),
        bounds_hit=bounds_hit, message=message,
    )


#: VOIs the kinetic models are fitted to.  The liver is delineated in the
#: group's data but is not a kinetic target there: it enters the analysis as a
#: late-time blood surrogate, calibrated against a blood sample and combined
#: with the partial-volume-corrected left ventricle to form the image-derived
#: input function (Kuttner et al., EJNMMI Res 16:42, 2026).  Fitting a trapping
#: model to it would also be wrong on its own terms -- hepatic
#: glucose-6-phosphatase dephosphorylates FDG-6-phosphate, so hepatic FDG is
#: not irreversibly trapped, and both Patlak and a k4-free compartment model
#: assume it is.
KINETIC_VOIS: tuple[str, ...] = ("Brain", "Myocardium")


def compare_kinetics(
    predicted_input: Sequence[float],
    truth_input: Sequence[float],
    tissue_curves: Mapping[str, Sequence[float]],
    time_min: Sequence[float],
    models: Sequence[str] = ("patlak", "two_tissue"),
    t_star_min: float | None = None,
    vois: Sequence[str] | None = KINETIC_VOIS,
    irreversible: bool = False,
) -> list[dict[str, Any]]:
    """Fit each tissue VOI twice -- with the predicted and the true input.

    Returns one row per ``(voi, model, parameter)`` with both values and their
    relative difference, which is the quantity a reader wants: how much does the
    input-function error move the parameter someone would report?

    ``vois`` restricts which VOIs are fitted; the default is the two the models
    are valid for.  Passing ``None`` fits everything supplied, which is how the
    liver appears in the sensitivity appendix rather than in the results.
    """
    rows: list[dict[str, Any]] = []
    wanted = None if vois is None else {v.lower() for v in vois}

    for voi, tissue in tissue_curves.items():
        if wanted is not None and str(voi).lower() not in wanted:
            continue

        if "patlak" in models:
            with_pred = patlak(tissue, predicted_input, time_min, t_star_min)
            with_truth = patlak(tissue, truth_input, time_min, t_star_min)
            rows.append(
                {
                    "voi": voi, "model": "patlak", "parameter": "ki",
                    "value_predicted": with_pred.ki, "value_truth": with_truth.ki,
                    "relative_error": _relative(with_pred.ki, with_truth.ki),
                    "r_squared_predicted": with_pred.r_squared,
                    "r_squared_truth": with_truth.r_squared,
                    "t_star_predicted": with_pred.t_star_min,
                    "t_star_truth": with_truth.t_star_min,
                }
            )

        if "two_tissue" in models:
            with_pred = fit_two_tissue(tissue, predicted_input, time_min,
                                       irreversible=irreversible)
            with_truth = fit_two_tissue(tissue, truth_input, time_min,
                                        irreversible=irreversible)
            parameters = ("k1", "k2", "k3", "vb", "ki")
            if not irreversible:
                parameters = ("k1", "k2", "k3", "k4", "vb", "ki")
            for parameter in parameters:
                rows.append(
                    {
                        "voi": voi, "model": "two_tissue", "parameter": parameter,
                        "value_predicted": getattr(with_pred, parameter),
                        "value_truth": getattr(with_truth, parameter),
                        "relative_error": _relative(
                            getattr(with_pred, parameter), getattr(with_truth, parameter)
                        ),
                        "converged_predicted": with_pred.converged,
                        "converged_truth": with_truth.converged,
                        "bounds_hit_predicted": with_pred.bounds_hit,
                        "bounds_hit_truth": with_truth.bounds_hit,
                        "irreversible": bool(irreversible),
                    }
                )

    return rows


def _relative(value: float, reference: float) -> float:
    if not np.isfinite(value) or not np.isfinite(reference) or abs(reference) < 1e-12:
        return float("nan")
    return float((value - reference) / reference)
