import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from inference_serving.analysis import paired_trial_family
from inference_serving.data import partition_final_articles, stable_rank, verify_tokenizer_identity
from inference_serving.distillation import distillation_loss
from inference_serving.statistics import bootstrap_ci, holm_adjust, paired_bootstrap, wilson


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "inference_serving/study_config.json").read_text())


def test_preregistration_hashes_and_grpo_exclusion_are_frozen():
    manifest = json.loads((ROOT / "inference_serving/preregistration_manifest.json").read_text())
    assert CONFIG["scope"]["grpo_excluded"] is True
    assert CONFIG["scope"]["targets"] == ["base", "sft", "dpo"]
    for row in manifest["files"].values():
        assert hashlib.sha256((ROOT / row["path"]).read_bytes()).hexdigest() == row["sha256"]


def test_final_partition_is_deterministic_disjoint_and_complete():
    rows = [{"article_id": f"a-{index}", "source": "s", "reference": "r"} for index in range(200)]
    pilot, heldout = partition_final_articles(rows, CONFIG)
    assert len(pilot) == 32 and len(heldout) == 168
    assert {row["article_id"] for row in pilot}.isdisjoint(row["article_id"] for row in heldout)
    assert partition_final_articles(list(reversed(rows)), CONFIG) == (pilot, heldout)
    assert stable_rank(1, "x", "a") == stable_rank(1, "x", "a")


class FakeTokenizer:
    def __init__(self, vocab, special): self.vocab, self.special = vocab, special
    def get_vocab(self): return self.vocab
    @property
    def special_tokens_map(self): return self.special


def test_tokenizer_identity_fails_closed():
    same = FakeTokenizer({"a": 0}, {"eos_token": "a"})
    assert verify_tokenizer_identity({"a": same, "b": same})["identical"]
    with pytest.raises(ValueError, match="vocabulary mismatch"):
        verify_tokenizer_identity({"a": same, "b": FakeTokenizer({"b": 0}, {"eos_token": "a"})})


def test_distillation_loss_rewards_matching_teacher():
    torch = pytest.importorskip("torch")
    labels = torch.tensor([[0, 1, 2]])
    teacher = torch.tensor([[[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]]])
    matched = teacher.clone()
    wrong = -teacher
    matched_total, _, matched_kl = distillation_loss(matched, teacher, labels, temperature=2.0, kl_weight=0.8)
    wrong_total, _, wrong_kl = distillation_loss(wrong, teacher, labels, temperature=2.0, kl_weight=0.8)
    assert matched_kl < wrong_kl and matched_total < wrong_total


def test_bootstrap_wilson_and_holm_are_deterministic():
    first = bootstrap_ci([1, 2, 3], replicates=200, seed=7)
    assert first == bootstrap_ci([1, 2, 3], replicates=200, seed=7)
    paired = paired_bootstrap([2, 3, 4], [1, 2, 3], replicates=200, seed=9)
    assert paired["estimate"] == 1 and paired["ci95"][0] == 1
    assert wilson(90, 100)["wilson_ci95"][0] < 0.9
    adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.02})
    assert all(0 <= value <= 1 for value in adjusted.values())


def test_paired_trial_family_uses_matched_trial_ids():
    indexed = {}
    for trial in range(5):
        for system, rate in (("vllm", 20 + trial), ("hf", 10 + trial)):
            key = (system, "dpo", "fp16", False, 32, trial)
            indexed[key] = {metric: rate for metric in (
                "output_tokens_per_second", "request_throughput", "ttft_ms_p50", "ttft_ms_p95",
                "itl_ms_p50", "itl_ms_p95", "peak_gpu_memory_bytes")}
    result = paired_trial_family(indexed, ("vllm", "dpo", "fp16", False), ("hf", "dpo", "fp16", False), 32, CONFIG)
    assert result["n_trials"] == 5
    assert result["metrics"]["output_tokens_per_second"]["estimate"] == 10
