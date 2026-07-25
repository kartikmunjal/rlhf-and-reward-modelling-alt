"""Calibration, identity fairness, and external diagnostics for v2."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_fscore_support,
)

from .metrics import _bootstrap_ci, _wilson_interval
from .taxonomy import TARGET_LABELS


def expected_calibration_error(
    truth: np.ndarray, probability: np.ndarray, n_bins: int = 15
) -> tuple[float, list[dict]]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    ece = 0.0
    for index in range(n_bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (probability >= lower) & (
            probability <= upper if index == n_bins - 1 else probability < upper
        )
        count = int(mask.sum())
        if count:
            confidence = float(probability[mask].mean())
            observed = float(truth[mask].mean())
            ece += count / len(truth) * abs(confidence - observed)
        else:
            confidence = observed = None
        bins.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "n": count,
                "mean_probability": confidence,
                "observed_rate": observed,
            }
        )
    return float(ece), bins


def calibration_report(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    n_bins: int = 15,
    n_bootstrap: int = 2000,
    seed: int = 2025,
    n_trials: int = 12,
) -> dict:
    report = {
        "n_examples": int(len(y_true)),
        "n_trials": int(n_trials),
        "n_bootstrap": int(n_bootstrap),
        "n_bins": int(n_bins),
        "per_category": {},
    }
    for column, name in enumerate(TARGET_LABELS):
        truth = y_true[:, column].astype(int)
        probability = probabilities[:, column]
        if len(np.unique(truth)) < 2:
            raise ValueError(f"Calibration requires both classes for {name}")
        ap = float(average_precision_score(truth, probability))
        brier = float(brier_score_loss(truth, probability))
        ece, bins = expected_calibration_error(truth, probability, n_bins)
        functions = {
            "pr_auc": lambda idx: average_precision_score(truth[idx], probability[idx])
            if len(np.unique(truth[idx])) == 2
            else np.nan,
            "brier": lambda idx: brier_score_loss(truth[idx], probability[idx]),
            "ece": lambda idx: expected_calibration_error(
                truth[idx], probability[idx], n_bins
            )[0],
        }
        values = {"pr_auc": ap, "brier": brier, "ece": ece}
        item = {"reliability_bins": bins}
        for offset, metric_name in enumerate(("pr_auc", "brier", "ece")):
            rng = np.random.default_rng(seed + column * 20 + offset)
            estimates = []
            while len(estimates) < n_bootstrap:
                indices = rng.integers(0, len(truth), len(truth))
                value = functions[metric_name](indices)
                if np.isfinite(value):
                    estimates.append(float(value))
            item[metric_name] = {
                "value": values[metric_name],
                "ci95": [float(x) for x in np.quantile(estimates, [0.025, 0.975])],
            }
        report["per_category"][name] = item
    macro_prob = probabilities
    macro_truth = y_true
    macro_f1 = float(
        np.mean(
            [
                f1_score(macro_truth[:, col], macro_prob[:, col] >= 0.5, zero_division=0)
                for col in range(macro_truth.shape[1])
            ]
        )
    )
    report["macro_f1_at_0_5"] = {
        "value": macro_f1,
        "ci95": _bootstrap_ci(
            lambda idx: float(
                np.mean(
                    [
                        f1_score(
                            macro_truth[idx, col],
                            macro_prob[idx, col] >= 0.5,
                            zero_division=0,
                        )
                        for col in range(macro_truth.shape[1])
                    ]
                )
            ),
            len(y_true),
            n_bootstrap,
            seed + 99,
        ),
    }
    return report


def threshold_sensitivity(
    y_true: np.ndarray, probabilities: np.ndarray, thresholds: Iterable[float]
) -> dict:
    output = {}
    for column, name in enumerate(TARGET_LABELS):
        rows = []
        for threshold in thresholds:
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true[:, column],
                probabilities[:, column] >= threshold,
                average="binary",
                zero_division=0,
            )
            rows.append(
                {
                    "threshold": float(threshold),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                }
            )
        output[name] = rows
    return output


def binary_diagnostic_report(
    truth: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    slices: np.ndarray,
    *,
    n_bootstrap: int = 2000,
    seed: int = 2025,
    n_trials: int = 12,
) -> dict:
    truth = truth.astype(int)
    prediction = probabilities >= threshold

    def summarize(mask: np.ndarray, offset: int) -> dict:
        local_truth, local_prediction = truth[mask], prediction[mask]
        precision, recall, f1, _ = precision_recall_fscore_support(
            local_truth, local_prediction, average="binary", zero_division=0
        )
        correct = int((local_truth == local_prediction).sum())
        item = {
            "n_examples": int(mask.sum()),
            "n_positive": int(local_truth.sum()),
            "accuracy": {
                "value": float(correct / len(local_truth)),
                "ci95": _wilson_interval(correct, len(local_truth)),
            },
        }
        for metric_offset, (metric_name, point) in enumerate(
            (("precision", precision), ("recall", recall), ("f1", f1))
        ):
            def statistic(indices, metric_name=metric_name):
                p, r, f, _ = precision_recall_fscore_support(
                    local_truth[indices],
                    local_prediction[indices],
                    average="binary",
                    zero_division=0,
                )
                return {"precision": p, "recall": r, "f1": f}[metric_name]

            item[metric_name] = {
                "value": float(point),
                "ci95": _bootstrap_ci(
                    statistic, len(local_truth), n_bootstrap, seed + offset * 10 + metric_offset
                ),
            }
        return item

    report = {
        "n_trials": int(n_trials),
        "threshold": float(threshold),
        "overall": summarize(np.ones(len(truth), dtype=bool), 0),
        "by_slice": {},
    }
    for offset, name in enumerate(sorted(set(slices)), start=1):
        report["by_slice"][str(name)] = summarize(slices == name, offset)
    return report


def identity_fairness_report(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray,
    memberships: dict[str, np.ndarray],
    *,
    n_bootstrap: int = 2000,
    seed: int = 2025,
    n_trials: int = 12,
) -> dict:
    true_any = y_true.astype(bool).any(axis=1)
    predicted_any = (probabilities >= thresholds).any(axis=1)
    groups = {}
    for name, membership in memberships.items():
        member = np.asarray(membership, dtype=bool)
        negatives = member & ~true_any
        positives = member & true_any
        item = {"n_members": int(member.sum())}
        for rate_name, mask, events in (
            ("fpr", negatives, predicted_any),
            ("fnr", positives, ~predicted_any),
        ):
            total = int(mask.sum())
            if total == 0:
                item[rate_name] = None
            else:
                successes = int(events[mask].sum())
                item[rate_name] = {
                    "value": float(successes / total),
                    "ci95": _wilson_interval(successes, total),
                    "n": total,
                    "events": successes,
                }
        groups[name] = item

    def gap(rate_name: str) -> dict | None:
        available = {
            name: item[rate_name]["value"]
            for name, item in groups.items()
            if item[rate_name] is not None
        }
        if len(available) < 2:
            return None
        maximum = max(available, key=available.get)
        minimum = min(available, key=available.get)
        high_interval = groups[maximum][rate_name]["ci95"]
        low_interval = groups[minimum][rate_name]["ci95"]
        return {
            "value": float(available[maximum] - available[minimum]),
            "ci95": [
                float(max(-1.0, high_interval[0] - low_interval[1])),
                float(min(1.0, high_interval[1] - low_interval[0])),
            ],
            "highest_group": maximum,
            "lowest_group": minimum,
            "interval_method": "conservative difference of Wilson bounds",
        }

    return {
        "n_examples": int(len(y_true)),
        "n_trials": int(n_trials),
        "groups": groups,
        "fpr_gap_worst_minus_best": gap("fpr"),
        "fnr_gap_worst_minus_best": gap("fnr"),
    }
