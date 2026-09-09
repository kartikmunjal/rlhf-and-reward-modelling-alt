"""Preregistered linear versus saturating-exponential curve comparison."""

from __future__ import annotations

import numpy as np


def _linear_fit(x, y):
    coefficients = np.polyfit(x, y, 1)
    return lambda values: np.polyval(coefficients, values), [float(v) for v in coefficients]


def _saturating_fit(x, y):
    from scipy.optimize import curve_fit

    def function(values, asymptote, gap, rate):
        return asymptote - gap * np.exp(-rate * values)

    scale = max(float(np.max(x)), 1.0)
    params, _ = curve_fit(
        function, x, y,
        p0=[min(1.0, float(np.max(y)) + 0.05), max(0.01, float(np.ptp(y))), 1.0 / scale],
        bounds=([-1.0, 0.0, 1e-12], [2.0, 3.0, np.inf]),
        maxfev=50000,
    )
    return lambda values: function(np.asarray(values), *params), [float(v) for v in params]


def _loo_error(x, y, fitter) -> float:
    errors = []
    for heldout in range(len(x)):
        keep = np.arange(len(x)) != heldout
        predict, _ = fitter(x[keep], y[keep])
        errors.append((float(predict(np.asarray([x[heldout]]))[0]) - y[heldout]) ** 2)
    return float(np.mean(errors))


def select_curve(x, y, *, tie_tolerance: float = 1e-6) -> dict:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 1 or y.shape != x.shape or len(x) < 4 or len(set(x)) != len(x):
        raise ValueError("Curve comparison needs at least four unique x observations")
    candidates = {}
    for name, fitter in (("linear", _linear_fit), ("saturating_exponential", _saturating_fit)):
        try:
            prediction, parameters = fitter(x, y)
            candidates[name] = {
                "parameters": parameters,
                "loo_mse": _loo_error(x, y, fitter),
                "fitted": [float(v) for v in prediction(x)],
            }
        except (RuntimeError, ValueError, FloatingPointError) as error:
            candidates[name] = {"status": "fit_failed", "error": str(error), "loo_mse": float("inf")}
    difference = candidates["saturating_exponential"]["loo_mse"] - candidates["linear"]["loo_mse"]
    selected = "linear" if difference >= -tie_tolerance else "saturating_exponential"
    return {"selected": selected, "tie_tolerance": tie_tolerance, "candidates": candidates}
