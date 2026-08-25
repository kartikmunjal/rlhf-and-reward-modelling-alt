import numpy as np
import pytest

from llm_judge_summeval.statistics import (
    article_cluster_bootstrap,
    fit_bradley_terry,
    position_bias_counts,
    rouge_l_fmeasure,
    spearman,
    wilson_interval,
)


def test_spearman_and_rouge_l_known_cases():
    assert spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert spearman([1, 2, 3], [30, 20, 10]) == pytest.approx(-1.0)
    assert rouge_l_fmeasure("a b c", "a b c") == pytest.approx(1.0)
    assert rouge_l_fmeasure("a b", "a x b") == pytest.approx(0.8)


def test_cluster_bootstrap_resamples_articles_not_rows():
    result = article_cluster_bootstrap(
        ["a", "a", "b", "b"], ([1, 2, 3, 4],), lambda values: np.mean(values), replicates=100, seed=7
    )
    assert result["estimate"] == pytest.approx(2.5)
    assert result["n_articles"] == 2
    assert result["replicates"] == 100


def test_wilson_has_nonzero_boundary_width():
    low, high = wilson_interval(0, 100)
    assert low == 0
    assert 0 < high < 0.1


def test_bradley_terry_orders_consistent_winner():
    scores = fit_bradley_terry(["a", "b", "c"], [("a", "b", 1), ("a", "c", 1), ("b", "c", 1)], 0.001)
    assert scores["a"] > scores["b"] > scores["c"]
    assert sum(scores.values()) == pytest.approx(0.0, abs=1e-8)


def test_position_bias_separates_flips_from_tie_instability():
    rows = [
        {"pair_id": "1", "axis": "relevance", "order": "ab", "underlying_winner": "x"},
        {"pair_id": "1", "axis": "relevance", "order": "ba", "underlying_winner": "y"},
        {"pair_id": "2", "axis": "relevance", "order": "ab", "underlying_winner": "tie"},
        {"pair_id": "2", "axis": "relevance", "order": "ba", "underlying_winner": "x"},
    ]
    assert position_bias_counts(rows) == {
        "complete_pairs": 2, "preference_flips": 1, "tie_instability": 1, "stable": 0
    }
