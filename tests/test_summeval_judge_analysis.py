import pytest

from llm_judge_summeval.analysis import _human_winner, _underlying_winner


def test_pairwise_display_winner_maps_to_underlying_summary():
    row = {"parsed": {"relevance": {"winner": "B"}}, "metadata": {"display_a_id": "x", "display_b_id": "y"}}
    assert _underlying_winner(row, "relevance") == "y"
    row["parsed"]["relevance"]["winner"] = "tie"
    assert _underlying_winner(row, "relevance") == "tie"


def test_human_winner_uses_expert_mean_and_handles_ties():
    labels = {"x": {"expert_mean": {"relevance": 4}}, "y": {"expert_mean": {"relevance": 3}}}
    assert _human_winner("x", "y", labels, "relevance") == "x"
    labels["y"]["expert_mean"]["relevance"] = 4
    assert _human_winner("x", "y", labels, "relevance") == "tie"
