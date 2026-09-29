"""Locked task-clustered and hierarchical paired inference."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def _interval(samples) -> list[float]:
    return np.quantile(samples, [0.025, 0.975]).tolist()


def bootstrap_p_two_sided(samples) -> float:
    """Pre-result bootstrap sign probability used for Holm adjustment."""
    samples = np.asarray(samples, dtype=float)
    denominator = len(samples) + 1
    lower = (np.count_nonzero(samples <= 0) + 1) / denominator
    upper = (np.count_nonzero(samples >= 0) + 1) / denominator
    return float(min(1.0, 2 * min(lower, upper)))


def task_bootstrap_mean(values, replicates: int, seed: int) -> dict:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = np.array([values[rng.integers(0, len(values), len(values))].mean() for _ in range(replicates)])
    return {"estimate": float(values.mean()), "ci95": _interval(samples), "n_trials": 1, "n_tasks_per_trial": int(len(values)), "bootstrap_replicates": replicates}


def hierarchical_bootstrap_mean(values, replicates: int, seed: int) -> dict:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("values must have shape [seed, task]")
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates)
    for index in range(replicates):
        seeds = rng.integers(0, values.shape[0], values.shape[0])
        samples[index] = np.mean([values[s, rng.integers(0, values.shape[1], values.shape[1])].mean() for s in seeds])
    return {"estimate": float(values.mean()), "ci95": _interval(samples), "n_trials": int(values.shape[0]), "n_tasks_per_trial": int(values.shape[1]), "bootstrap_replicates": replicates}


def paired_task_bootstrap(left, right, replicates: int, seed: int) -> dict:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("paired task arrays must be equal one-dimensional shapes")
    diff = left - right
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates)
    for index in range(replicates):
        choice = rng.integers(0, len(diff), len(diff))
        samples[index] = diff[choice].mean()
    return {
        "estimate": float(diff.mean()), "ci95": _interval(samples),
        "n_tasks": int(len(diff)), "bootstrap_replicates": replicates,
        "p_two_sided": bootstrap_p_two_sided(samples),
    }


def hierarchical_paired_bootstrap(left, right, replicates: int, seed: int) -> dict:
    """Resample paired seeds, then paired tasks within sampled seeds."""
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("arrays must have shape [seed, task] and match")
    diff = left - right
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates)
    for index in range(replicates):
        seeds = rng.integers(0, diff.shape[0], diff.shape[0])
        seed_means = []
        for selected_seed in seeds:
            tasks = rng.integers(0, diff.shape[1], diff.shape[1])
            seed_means.append(diff[selected_seed, tasks].mean())
        samples[index] = np.mean(seed_means)
    return {
        "estimate": float(diff.mean()), "ci95": _interval(samples),
        "n_trials": int(diff.shape[0]), "n_tasks_per_trial": int(diff.shape[1]),
        "bootstrap_replicates": replicates, "p_two_sided": bootstrap_p_two_sided(samples),
    }


def spearman_task_bootstrap(x, y, replicates: int, seed: int) -> dict:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("Spearman inputs must be paired vectors")
    estimate = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(replicates):
        choice = rng.integers(0, len(x), len(x))
        value = spearmanr(x[choice], y[choice]).statistic
        if np.isfinite(value):
            samples.append(value)
    return {"estimate": estimate, "ci95": _interval(samples), "n_tasks": int(len(x)), "bootstrap_replicates": replicates, "valid_bootstrap_replicates": len(samples)}


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    count = len(p_values)
    for rank, original_index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * p_values[original_index]))
        adjusted[original_index] = running
    return adjusted.tolist()
