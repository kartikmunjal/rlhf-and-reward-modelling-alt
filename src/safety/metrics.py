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


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    """Wilson score interval for a binomial proportion."""

    if total <= 0:
        raise ValueError("Wilson interval requires a positive total")
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    radius = (
        z
        * np.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2))
        / denominator
    )
    return [float(max(0.0, center - radius)), float(min(1.0, center + radius))]


def _newcombe_difference_interval(
    first_successes: int,
    first_total: int,
    second_successes: int,
    second_total: int,
) -> list[float]:
    """Newcombe score interval for two independent proportions, first-second."""

    first = first_successes / first_total
    second = second_successes / second_total
    first_low, first_high = _wilson_interval(first_successes, first_total)
    second_low, second_high = _wilson_interval(second_successes, second_total)
    difference = first - second
    lower = difference - np.sqrt((first - first_low) ** 2 + (second_high - second) ** 2)
    upper = difference + np.sqrt((first_high - first) ** 2 + (second - second_low) ** 2)
    return [float(max(-1.0, lower)), float(min(1.0, upper))]


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
    per_row_truth = y_true
    per_row_prediction = predictions
    macro_point = float(
        np.mean(
            [
                precision_recall_fscore_support(
                    per_row_truth[:, column],
                    per_row_prediction[:, column],
                    average="binary",
                    zero_division=0,
                )[2]
                for column in range(per_row_truth.shape[1])
            ]
        )
    )
    report["macro_f1"] = {
        "value": macro_point,
        "ci95": _bootstrap_ci(
            lambda indices: float(
                np.mean(
                    [
                        precision_recall_fscore_support(
                            per_row_truth[indices, column],
                            per_row_prediction[indices, column],
                            average="binary",
                            zero_division=0,
                        )[2]
                        for column in range(per_row_truth.shape[1])
                    ]
                )
            ),
            len(y_true),
            n_bootstrap,
            seed + 100,
        ),
    }
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

    def summarize(flags: np.ndarray) -> dict:
        successes = int(flags.sum())
        total = int(len(flags))
        return {
            "value": float(successes / total),
            "ci95": _wilson_interval(successes, total),
            "n_examples": total,
            "n_flagged": successes,
        }

    base = summarize(base_flags)
    adjacent = summarize(benign_flags)
    gap_value = adjacent["value"] - base["value"]
    gap_interval = _newcombe_difference_interval(
        adjacent["n_flagged"],
        adjacent["n_examples"],
        base["n_flagged"],
        base["n_examples"],
    )

    per_slice = {}
    for slice_name in sorted(set(benign_slices)):
        per_slice[str(slice_name)] = summarize(benign_flags[benign_slices == slice_name])
    return {
        "n_trials": int(n_trials),
        "interval_method": "Wilson score for FPR; Newcombe score for independent FPR gap",
        "overall_test_fpr": base,
        "adjacent_benign_fpr": adjacent,
        "adjacent_benign_fpr_gap": {
            "value": float(gap_value),
            "ci95": gap_interval,
        },
        "adjacent_benign_by_slice": per_slice,
    }
