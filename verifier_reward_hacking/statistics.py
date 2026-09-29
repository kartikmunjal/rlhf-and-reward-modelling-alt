"""Locked task-clustered and hierarchical paired inference."""

from __future__ import annotations

import numpy as np


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
        "estimate": float(diff.mean()), "ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
        "n_tasks": int(len(diff)), "bootstrap_replicates": replicates,
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
        "estimate": float(diff.mean()), "ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
        "n_trials": int(diff.shape[0]), "n_tasks_per_trial": int(diff.shape[1]),
        "bootstrap_replicates": replicates,
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    count = len(p_values)
    for rank, original_index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * p_values[original_index]))
        adjusted[original_index] = running
    return adjusted.tolist()
