"""Deterministic clustered bootstrap and Wilson intervals."""

from __future__ import annotations

import math
from typing import Callable

import numpy as np


def bootstrap_ci(values, *, replicates: int, seed: int, statistic: Callable = np.mean) -> dict:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values):
        raise ValueError("Bootstrap values must be a non-empty vector")
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates)
    for index in range(replicates):
        sample = values[rng.integers(0, len(values), len(values))]
        draws[index] = statistic(sample)
    return {
        "estimate": float(statistic(values)),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "n_trials": int(len(values)),
        "bootstrap_replicates": replicates,
    }


def paired_bootstrap(left, right, *, replicates: int, seed: int) -> dict:
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if left.shape != right.shape:
        raise ValueError("Paired vectors must have identical shapes")
    return bootstrap_ci(left - right, replicates=replicates, seed=seed)


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> dict:
    if total <= 0 or successes < 0 or successes > total:
        raise ValueError("Invalid binomial counts")
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return {"rate": p, "valid": successes, "total": total, "wilson_ci95": [center - half, center + half]}


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted, running = {}, 0.0
    count = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * value))
        adjusted[name] = running
    return adjusted


def paired_sign_permutation(left, right, *, permutations: int, seed: int) -> float:
    """Two-sided paired randomization p-value for the mean difference."""
    differences = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    if differences.ndim != 1 or not len(differences):
        raise ValueError("Paired vectors must be non-empty")
    observed = abs(float(np.mean(differences)))
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(permutations):
        signs = rng.choice((-1.0, 1.0), size=len(differences))
        exceed += abs(float(np.mean(differences * signs))) >= observed
    return float((exceed + 1) / (permutations + 1))
