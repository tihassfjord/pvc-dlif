"""Downstream kinetic modelling: what the input function is actually for.

An input function is a means, not an end.  A change in RMSE only matters if it
propagates to the parameters someone would report, so every condition's
predicted curve is also pushed through the two models used with dynamic FDG
data and the resulting parameters are compared against the same models driven
by the arterial ground truth.

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
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Mapping, Sequence

import numpy as np

from ..logging_utils import get_logger

LOGGER = get_logger(__name__)

__all__ = ["PatlakResult", "TwoTissueResult", "patlak", "fit_two_tissue", "compare_kinetics"]


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
    t_star_min: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def patlak(
    tissue: Sequence[float],
    plasma: Sequence[float],
    time_min: Sequence[float],
    t_star_min: float = 5.0,
) -> PatlakResult:
    """Patlak slope ``K_i`` from a tissue curve and an input function.

    Only samples after ``t_star_min`` are used, where the reversible
    compartments are assumed to have equilibrated.
    """
    c_t = np.asarray(tissue, dtype=np.float64).ravel()
    c_p = np.asarray(plasma, dtype=np.float64).ravel()
    t = np.asarray(time_min, dtype=np.float64).ravel()

    n = min(c_t.size, c_p.size, t.size)
    c_t, c_p, t = c_t[:n], c_p[:n], t[:n]

    integral = _cumulative_integral(c_p, t)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = integral / c_p
        y = c_t / c_p

    usable = (t >= t_star_min) & np.isfinite(x) & np.isfinite(y) & (c_p > 1e-9)
    if usable.sum() < 3:
        return PatlakResult(float("nan"), float("nan"), float("nan"), int(usable.sum()), t_star_min)

    slope, intercept = np.polyfit(x[usable], y[usable], 1)
    fitted = slope * x[usable] + intercept
    ss_res = float(np.sum((y[usable] - fitted) ** 2))
    ss_tot = float(np.sum((y[usable] - y[usable].mean()) ** 2))

    return PatlakResult(
        ki=float(slope),
        intercept=float(intercept),
        r_squared=float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan"),
        n_points=int(usable.sum()),
        t_star_min=float(t_star_min),
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

    The impulse response is the standard bi-exponential; the convolution with
    the plasma curve is done on the (irregular) sample grid by trapezoidal
    accumulation, so the very short early frames are not implicitly resampled.
    """
    alpha = k2 + k3 + k4
    disc = max(alpha ** 2 - 4.0 * k2 * k4, 0.0)
    root = np.sqrt(disc)
    theta1 = (alpha - root) / 2.0
    theta2 = (alpha + root) / 2.0

    denom = theta2 - theta1
    if abs(denom) < 1e-12:
        coeff1 = k1
        coeff2 = 0.0
    else:
        coeff1 = k1 * (k3 + k4 - theta1) / denom
        coeff2 = k1 * (theta2 - k3 - k4) / denom

    out = np.zeros_like(t, dtype=np.float64)
    for i in range(t.size):
        if i == 0:
            continue
        tau = t[: i + 1]
        dt = t[i] - tau
        response = coeff1 * np.exp(-theta1 * dt) + coeff2 * np.exp(-theta2 * dt)
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
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fit_two_tissue(
    tissue: Sequence[float],
    plasma: Sequence[float],
    time_min: Sequence[float],
    irreversible: bool = True,
    initial: Sequence[float] | None = None,
) -> TwoTissueResult:
    """Fit ``K_1, k_2, k_3, (k_4), v_b`` by non-linear least squares.

    ``irreversible=True`` fixes ``k_4 = 0``, the usual assumption for FDG.
    """
    from scipy.optimize import curve_fit

    c_t = np.asarray(tissue, dtype=np.float64).ravel()
    c_p = np.asarray(plasma, dtype=np.float64).ravel()
    t = np.asarray(time_min, dtype=np.float64).ravel()
    n = min(c_t.size, c_p.size, t.size)
    c_t, c_p, t = c_t[:n], c_p[:n], t[:n]

    if n < 6:
        return TwoTissueResult(*([float("nan")] * 6), float("nan"), False, "too few samples")

    p0 = list(initial) if initial else ([0.3, 0.3, 0.05, 0.05] + [0.05])
    if irreversible:
        def model(tt, k1, k2, k3, vb):
            return _two_tissue_model(tt, c_p, k1, k2, k3, 0.0, vb)

        p0 = [p0[0], p0[1], p0[2], p0[4]]
        bounds = ([1e-6, 1e-6, 0.0, 0.0], [5.0, 5.0, 5.0, 0.5])
    else:
        def model(tt, k1, k2, k3, k4, vb):
            return _two_tissue_model(tt, c_p, k1, k2, k3, k4, vb)

        bounds = ([1e-6, 1e-6, 0.0, 0.0, 0.0], [5.0, 5.0, 5.0, 5.0, 0.5])

    try:
        popt, _ = curve_fit(model, t, c_t, p0=p0, bounds=bounds, maxfev=20000)
        converged, message = True, ""
    except Exception as exc:  # noqa: BLE001 - a failed fit is a result, not a crash
        LOGGER.debug("two-tissue fit failed: %s", exc)
        return TwoTissueResult(*([float("nan")] * 6), float("nan"), False, str(exc))

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
        ki=ki, rmse=rmse, converged=converged, message=message,
    )


def compare_kinetics(
    predicted_input: Sequence[float],
    truth_input: Sequence[float],
    tissue_curves: Mapping[str, Sequence[float]],
    time_min: Sequence[float],
    models: Sequence[str] = ("patlak", "two_tissue"),
    t_star_min: float = 5.0,
) -> list[dict[str, Any]]:
    """Fit each tissue VOI twice -- with the predicted and the true input.

    Returns one row per ``(voi, model, parameter)`` with both values and their
    relative difference, which is the quantity a reader wants: how much does the
    input-function error move the parameter someone would report?
    """
    rows: list[dict[str, Any]] = []

    for voi, tissue in tissue_curves.items():
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
                }
            )

        if "two_tissue" in models:
            with_pred = fit_two_tissue(tissue, predicted_input, time_min)
            with_truth = fit_two_tissue(tissue, truth_input, time_min)
            for parameter in ("k1", "k2", "k3", "vb", "ki"):
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
                    }
                )

    return rows


def _relative(value: float, reference: float) -> float:
    if not np.isfinite(value) or not np.isfinite(reference) or abs(reference) < 1e-12:
        return float("nan")
    return float((value - reference) / reference)
