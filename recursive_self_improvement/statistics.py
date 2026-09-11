"""Deterministic paired inference for the recursive-improvement study."""

from __future__ import annotations

import math
import numpy as np


def paired_bootstrap(left, right, *, replicates: int, seed: int) -> dict:
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if left.shape != right.shape or left.ndim != 1 or not len(left):
        raise ValueError("Paired vectors must be non-empty and identically shaped")
    diff = left - right
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates)
    for index in range(replicates):
        draw = rng.integers(0, len(diff), len(diff))
        samples[index] = np.mean(diff[draw])
    return {
        "estimate": float(np.mean(diff)),
        "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
        "n_trials": int(len(diff)),
        "bootstrap_replicates": replicates,
    }


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> dict:
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("Invalid binomial counts")
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    half = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return {"rate": rate, "valid": successes, "total": total, "wilson_ci95": [center - half, center + half]}


def holm_adjust(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: item[1])
    output, running, total = {}, 0.0, len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * value))
        output[name] = running
    return output


def paired_sign_flip_test(left, right, *, replicates: int, seed: int) -> dict:
    """Two-sided paired randomization test of a zero mean difference."""
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if left.shape != right.shape or left.ndim != 1 or not len(left):
        raise ValueError("Paired vectors must be non-empty and identically shaped")
    differences = left - right
    observed = abs(float(np.mean(differences)))
    rng = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(replicates):
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=len(differences))
        exceedances += abs(float(np.mean(differences * signs))) >= observed
    return {
        "p_value_two_sided": float((exceedances + 1) / (replicates + 1)),
        "observed_mean_difference": float(np.mean(differences)),
        "n_trials": int(len(differences)),
        "randomization_replicates": int(replicates),
    }
