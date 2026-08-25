import json
from pathlib import Path

import pytest

from llm_judge_summeval.data import (
    deterministic_split,
    load_heldout_inputs,
    load_heldout_labels_for_analysis,
    select_balanced_pairs,
)


def test_article_split_is_deterministic_disjoint_and_exact():
    article_ids = [f"article-{index}" for index in range(100)]
    first = deterministic_split(article_ids, "locked-salt", 20)
    second = deterministic_split(reversed(article_ids), "locked-salt", 20)
    assert first == second
    assert len(first[0]) == 20 and len(first[1]) == 80
    assert set(first[0]).isdisjoint(first[1])


def test_pair_selection_is_unique_deterministic_and_balanced():
    ids = [f"summary-{index}" for index in range(16)]
    pairs = select_balanced_pairs(ids, "article", 5, "salt")
    assert pairs == select_balanced_pairs(list(reversed(ids)), "article", 5, "salt")
    assert len(pairs) == len(set(pairs)) == 5
    participation = {item: sum(item in pair for pair in pairs) for item in ids}
    assert max(participation.values()) <= 1


def test_heldout_loader_rejects_label_leakage_and_analysis_requires_unlock(tmp_path: Path):
    (tmp_path / "heldout_inputs.jsonl").write_text(
        json.dumps({"summary_id": "x", "expert_mean": {"relevance": 5}}) + "\n", encoding="utf-8"
    )
    (tmp_path / "heldout_labels.sealed.jsonl").write_text(json.dumps({"summary_id": "x"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="leakage"):
        load_heldout_inputs(tmp_path)
    with pytest.raises(PermissionError, match="sealed"):
        load_heldout_labels_for_analysis(tmp_path, prompt_manifest_verified=False)
    assert load_heldout_labels_for_analysis(tmp_path, prompt_manifest_verified=True) == [{"summary_id": "x"}]

