"""Offline metrics for the preregistered SummEval judge study."""

from __future__ import annotations

import math
import re
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata


TOKEN = re.compile(r"\w+", re.UNICODE)


def spearman(x, y) -> float:
    left, right = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if left.shape != right.shape or left.ndim != 1 or len(left) < 2:
        raise ValueError("Spearman inputs must be equal-length vectors")
    if np.ptp(left) == 0 or np.ptp(right) == 0:
        return float("nan")
    return float(np.corrcoef(rankdata(left), rankdata(right))[0, 1])


def partial_spearman(x, y, controls) -> float:
    """Spearman correlation after linearly residualizing ranked controls."""
    left, right = rankdata(np.asarray(x, dtype=float)), rankdata(np.asarray(y, dtype=float))
    control = np.asarray(controls, dtype=float)
    if control.ndim == 1:
        control = control[:, None]
    if len(left) != len(right) or control.shape[0] != len(left) or len(left) < 3:
        raise ValueError("Partial Spearman inputs must have matching rows")
    design = np.column_stack([np.ones(len(left)), np.apply_along_axis(rankdata, 0, control)])
    left_residual = left - design @ np.linalg.lstsq(design, left, rcond=None)[0]
    right_residual = right - design @ np.linalg.lstsq(design, right, rcond=None)[0]
    return spearman(left_residual, right_residual)


def rouge_l_fmeasure(candidate: str, reference: str) -> float:
    cand, ref = TOKEN.findall(candidate.lower()), TOKEN.findall(reference.lower())
    if not cand or not ref:
        return 0.0
    previous = [0] * (len(ref) + 1)
    for token in cand:
        current = [0]
        for index, target in enumerate(ref, start=1):
            current.append(previous[index - 1] + 1 if token == target else max(previous[index], current[-1]))
        previous = current
    lcs = previous[-1]
    precision, recall = lcs / len(cand), lcs / len(ref)
    return 2 * precision * recall / (precision + recall) if lcs else 0.0


def article_cluster_bootstrap(article_ids, values, statistic, *, replicates: int, seed: int,
                              cluster_unit: str = "source_article") -> dict:
    article_ids = np.asarray(article_ids)
    values = tuple(np.asarray(value) for value in values)
    unique = np.unique(article_ids)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(replicates):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(article_ids == article) for article in sampled])
        draws.append(float(statistic(*(value[indices] for value in values))))
    finite = np.asarray([value for value in draws if np.isfinite(value)])
    if not len(finite):
        raise ValueError("Bootstrap statistic was never finite")
    estimate = float(statistic(*values))
    result = {"estimate": estimate, "ci95": [float(x) for x in np.quantile(finite, [0.025, 0.975])],
              "cluster_unit": cluster_unit, "n_clusters": len(unique), "n_observations": len(article_ids),
              "replicates": replicates, "finite_replicates": len(finite)}
    if cluster_unit == "source_article":
        result["n_articles"] = len(unique)
    return result


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if not 0 <= successes <= total or total <= 0:
        raise ValueError("Invalid binomial counts")
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    low = 0.0 if successes == 0 else max(0.0, center - radius)
    high = 1.0 if successes == total else min(1.0, center + radius)
    return [low, high]


def fit_bradley_terry(items: list[str], comparisons: list[tuple[str, str, float]], l2: float) -> dict[str, float]:
    ordered = sorted(set(items))
    index = {item: position for position, item in enumerate(ordered)}

    def objective(raw):
        scores = raw - raw.mean()
        loss = l2 * float(scores @ scores)
        for left, right, left_outcome in comparisons:
            delta = scores[index[left]] - scores[index[right]]
            probability = 1 / (1 + np.exp(-np.clip(delta, -30, 30)))
            loss -= left_outcome * np.log(probability + 1e-12) + (1 - left_outcome) * np.log(1 - probability + 1e-12)
        return loss

    result = minimize(objective, np.zeros(len(ordered)), method="BFGS")
    if not result.success and result.jac is not None and np.linalg.norm(result.jac) > 1e-4:
        raise RuntimeError(f"Bradley-Terry fit failed: {result.message}")
    centered = result.x - result.x.mean()
    return {item: float(centered[index[item]]) for item in ordered}


def position_bias_counts(rows: list[dict]) -> dict[str, int]:
    grouped = defaultdict(dict)
    for row in rows:
        grouped[(row["pair_id"], row["axis"])][row["order"]] = row["underlying_winner"]
    counts = {"complete_pairs": 0, "preference_flips": 0, "tie_instability": 0, "stable": 0}
    for orders in grouped.values():
        if set(orders) != {"ab", "ba"}:
            continue
        counts["complete_pairs"] += 1
        first, second = orders["ab"], orders["ba"]
        if "tie" in (first, second) and first != second:
            counts["tie_instability"] += 1
        elif first != second:
            counts["preference_flips"] += 1
        else:
            counts["stable"] += 1
    return counts
