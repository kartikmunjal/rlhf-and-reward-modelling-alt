"""Research metrics with deterministic bootstrap confidence intervals."""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from .taxonomy import TARGET_LABELS


def select_thresholds(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    grid: np.ndarray | None = None,
) -> np.ndarray:
    """Select each threshold on calibration data only, maximizing F1.

    Ties resolve to the higher threshold, favoring fewer false positives.
    """

    candidates = np.asarray(grid if grid is not None else np.arange(0.05, 0.951, 0.01))
    thresholds = np.empty(y_true.shape[1], dtype=float)
    for column in range(y_true.shape[1]):
        scores = []
        for threshold in candidates:
            prediction = probabilities[:, column] >= threshold
            _, _, f1, _ = precision_recall_fscore_support(
                y_true[:, column], prediction, average="binary", zero_division=0
            )
            scores.append(f1)
        best = np.flatnonzero(np.asarray(scores) == np.max(scores))[-1]
        thresholds[column] = candidates[best]
    return thresholds


def _bootstrap_ci(
    statistic: Callable[[np.ndarray], float],
    n_rows: int,
    n_bootstrap: int,
    seed: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    estimates = np.asarray(
        [statistic(rng.integers(0, n_rows, size=n_rows)) for _ in range(n_bootstrap)]
    )
    return [float(x) for x in np.quantile(estimates, [0.025, 0.975])]


def classification_report(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray,
    *,
    n_bootstrap: int = 2000,
    seed: int = 2025,
    n_trials: int = 1,
) -> dict:
    predictions = probabilities >= thresholds
    report: dict = {
        "n_examples": int(len(y_true)),
        "n_trials": int(n_trials),
        "n_bootstrap": int(n_bootstrap),
        "thresholds": dict(zip(TARGET_LABELS, thresholds.tolist())),
        "per_category": {},
    }
    for column, name in enumerate(TARGET_LABELS):
        truth = y_true[:, column]
        pred = predictions[:, column]
        precision, recall, f1, _ = precision_recall_fscore_support(
            truth, pred, average="binary", zero_division=0
        )
        metrics = {"precision": precision, "recall": recall, "f1": f1}
        item = {"support": int(truth.sum())}
        for offset, (metric_name, value) in enumerate(metrics.items()):
            def statistic(indices: np.ndarray, metric_name: str = metric_name) -> float:
                p, r, f, _ = precision_recall_fscore_support(
                    truth[indices], pred[indices], average="binary", zero_division=0
                )
                return {"precision": p, "recall": r, "f1": f}[metric_name]

            item[metric_name] = {
                "value": float(value),
                "ci95": _bootstrap_ci(statistic, len(truth), n_bootstrap, seed + column * 10 + offset),
            }
        report["per_category"][name] = item
    return report


def fairness_report(
    base_y_true: np.ndarray,
    base_probabilities: np.ndarray,
    benign_probabilities: np.ndarray,
    benign_slices: np.ndarray,
    thresholds: np.ndarray,
    *,
    n_bootstrap: int = 2000,
    seed: int = 2025,
    n_trials: int = 1,
) -> dict:
    """Report example-level any-label FPR and adjacent-benign FPR gap."""

    base_negative = ~base_y_true.astype(bool).any(axis=1)
    base_flags = (base_probabilities >= thresholds).any(axis=1)[base_negative]
    benign_flags = (benign_probabilities >= thresholds).any(axis=1)
    if len(base_flags) == 0:
        raise ValueError("FPR requires at least one all-negative base example")
    if len(benign_flags) == 0:
        raise ValueError("FPR requires at least one adjacent-benign example")

    def summarize(flags: np.ndarray, offset: int) -> dict:
        value = float(flags.mean()) if len(flags) else float("nan")
        ci = _bootstrap_ci(lambda idx: float(flags[idx].mean()), len(flags), n_bootstrap, seed + offset)
        return {"value": value, "ci95": ci, "n_examples": int(len(flags))}

    base = summarize(base_flags, 0)
    adjacent = summarize(benign_flags, 1)
    gap_value = adjacent["value"] - base["value"]
    rng = np.random.default_rng(seed + 2)
    gaps = []
    for _ in range(n_bootstrap):
        a = benign_flags[rng.integers(0, len(benign_flags), len(benign_flags))].mean()
        b = base_flags[rng.integers(0, len(base_flags), len(base_flags))].mean()
        gaps.append(a - b)

    per_slice = {}
    for offset, slice_name in enumerate(sorted(set(benign_slices)), start=10):
        per_slice[str(slice_name)] = summarize(benign_flags[benign_slices == slice_name], offset)
    return {
        "n_trials": int(n_trials),
        "n_bootstrap": int(n_bootstrap),
        "overall_test_fpr": base,
        "adjacent_benign_fpr": adjacent,
        "adjacent_benign_fpr_gap": {
            "value": float(gap_value),
            "ci95": [float(x) for x in np.quantile(gaps, [0.025, 0.975])],
        },
        "adjacent_benign_by_slice": per_slice,
    }
