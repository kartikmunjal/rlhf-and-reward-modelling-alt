import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from summarization_finetune.analysis import controlled_treatment, paired_mean
from summarization_finetune.data import normalized_text, ranked, text_sha256
from summarization_finetune.generation import candidate_seed
from summarization_finetune.judging import balanced_pair_order, build_preferences, verify_frozen_judge
from scripts.publish_summarization_finetune_readme import coverage, effect


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "summarization_finetune/study_config.json").read_text())


def test_preregistration_and_frozen_judge_hashes_are_intact():
    manifest = json.loads((ROOT / "summarization_finetune/preregistration_manifest.json").read_text())
    for row in manifest["files"].values():
        assert hashlib.sha256((ROOT / row["path"]).read_bytes()).hexdigest() == row["sha256"]
    verify_frozen_judge(CONFIG, ROOT)


def test_normalization_hash_and_rank_are_deterministic():
    assert normalized_text(" A\n b  ") == "a b"
    assert text_sha256("A b") == text_sha256(" a\nB ")
    rows = [{"id": "b"}, {"id": "a"}, {"id": "c"}]
    assert ranked(rows, "salt") == ranked(list(reversed(rows)), "salt")


def test_candidate_seed_is_stable_and_candidate_specific():
    assert candidate_seed(7, "article", 0) == candidate_seed(7, "article", 0)
    assert candidate_seed(7, "article", 0) != candidate_seed(7, "article", 1)


def test_pair_display_order_is_deterministic_and_not_fixed_to_index():
    left, right = {"candidate_id": "left"}, {"candidate_id": "right"}
    observed = [balanced_pair_order(7, f"pair-{index}", left, right)[0]["candidate_id"] for index in range(30)]
    assert observed == [balanced_pair_order(7, f"pair-{index}", left, right)[0]["candidate_id"] for index in range(30)]
    assert set(observed) == {"left", "right"}


def test_preference_builder_requires_primary_axis_agreement(tmp_path: Path):
    candidates = []
    for index in range(4):
        candidates.append({"article_id": "x", "candidate_index": index, "candidate_id": f"x:c{index}",
                           "source": "source", "reference": "ref", "summary": f"summary {index}"})
    candidate_path = tmp_path / "candidates.jsonl"
    candidate_path.write_text("".join(json.dumps(row) + "\n" for row in candidates))
    def result(request_id, left, right, rel, con):
        parsed = {axis: {"winner": "tie", "rationale": "r"} for axis in ("coherence", "consistency", "fluency", "relevance")}
        parsed["relevance"]["winner"], parsed["consistency"]["winner"] = rel, con
        return {"request_id": request_id, "status": "success", "item_id": request_id, "parsed": parsed,
                "metadata": {"article_id": "x", "pair_id": request_id, "display_a_id": left, "display_b_id": right}}
    ledger = tmp_path / "ledger.jsonl"
    rows = [result("p1", "x:c0", "x:c1", "A", "A"), result("p2", "x:c0", "x:c2", "A", "B"),
            result("p3", "x:c0", "x:c3", "tie", "A")]
    ledger.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "preferences.jsonl"
    summary = build_preferences(candidates_path=candidate_path, ledger_path=ledger, output_path=output)
    assert summary["preferences"] == 1
    preference = json.loads(output.read_text().splitlines()[0])
    assert preference["chosen_id"] == "x:c0" and preference["rejected_id"] == "x:c1"


def test_paired_bootstrap_and_length_control_recover_known_effect():
    config = {"statistics": {"bootstrap_replicates": 200, "bootstrap_seed": 5}}
    ids = np.array([f"a{i}" for i in range(30)])
    worse = np.linspace(1, 2, 30); better = worse + 0.5
    result = paired_mean(ids, better, worse, config)
    assert result["estimate"] == pytest.approx(0.5)
    assert result["ci95"][0] > 0
    sft_length = np.arange(30) + 20
    length_shift = np.linspace(-5, 5, 30)
    controlled = controlled_treatment(ids, worse + 0.5 + 0.1 * length_shift, worse,
                                      sft_length + length_shift, sft_length, config)
    assert controlled["estimate"] == pytest.approx(0.5, abs=1e-8)


def test_result_publisher_formats_generated_trials_and_intervals():
    row = {"estimate": 0.25, "ci95": [0.1, 0.4], "n_trials": 20, "bootstrap_replicates": 2000}
    assert effect(row) == "0.250 (95% CI 0.100–0.400; N_trials=20; 2,000 bootstraps)"
    validity = {"valid": 90, "total": 100, "wilson_ci95": [0.826, 0.945]}
    assert coverage(validity) == "90/100 (Wilson 95% CI 0.826–0.945)"


def test_published_result_provenance_matches_local_locked_artifacts():
    metrics = json.loads((ROOT / "results/summarization_finetune_v1/metrics.json").read_text())
    expected = {
        "config": ROOT / "summarization_finetune/study_config.json",
        "preregistration_manifest": ROOT / "summarization_finetune/preregistration_manifest.json",
        "data_manifest": ROOT / "data/processed/summarization_finetune_v1/data_manifest.json",
    }
    for name, path in expected.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == metrics["provenance"][name]["sha256"]
