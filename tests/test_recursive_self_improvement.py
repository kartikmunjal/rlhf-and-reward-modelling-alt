import json
from pathlib import Path

import numpy as np
import pytest

from recursive_self_improvement.curves import select_curve
from recursive_self_improvement.config import canonical_text_sha256, load_effective_config
from recursive_self_improvement.data import assert_disjoint, partition_allocations, partition_pairs, prompt_id
from recursive_self_improvement.mixture import choose_by_normalized_log_likelihood, use_self_label
from recursive_self_improvement.statistics import paired_bootstrap, paired_sign_flip_test


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "recursive_self_improvement" / "study_config.json").read_text())


def test_preregistration_is_hash_frozen():
    import hashlib
    manifest = json.loads((ROOT / "recursive_self_improvement" / "preregistration_manifest.json").read_text())
    for row in manifest["files"].values():
        assert hashlib.sha256((ROOT / row["path"]).read_bytes()).hexdigest() == row["sha256"]


def test_approved_amendment_is_hash_verified_and_applied():
    effective = load_effective_config(ROOT)
    assert effective["data"]["sft_train_pairs"] == 8000
    assert effective["data"]["independent_eval_source_split"] == "test"
    assert effective["data"]["independent_eval_prompts"] == 250
    assert effective["peft"]["rank"] == 16
    assert effective["smoke_gate"]["optimizer_steps"] == 2
    assert len(effective["applied_amendments"]) == 8
    assert effective["applied_amendments"][-1] == "recursive_self_improvement_v1_stage3_prompt_allocation_008"


def test_canonical_protocol_hash_is_eol_portable(tmp_path):
    lf = tmp_path / "lf.json"; crlf = tmp_path / "crlf.json"
    lf.write_bytes(b'{"a": 1}\n'); crlf.write_bytes(b'{"a": 1}\r\n')
    assert canonical_text_sha256(lf) == canonical_text_sha256(crlf)


def test_partitions_are_deterministic_and_disjoint():
    config = json.loads(json.dumps(CONFIG))
    config["data"].update(sft_train_pairs=3, reward_train_pairs=3, reward_validation_pairs=2, improvement_prompts=2, independent_eval_prompts=2)
    rows = [{"prompt": f"Human: prompt {i}\nAssistant:", "chosen": f"yes {i}", "rejected": f"no {i}"} for i in range(15)]
    first = partition_pairs(rows, config)
    second = partition_pairs(reversed(rows), config)
    assert first == second
    assert_disjoint(first)
    assert len({row["prompt_id"] for values in first.values() for row in values}) == 12


def test_conflicting_duplicate_prompt_fails_closed():
    config = json.loads(json.dumps(CONFIG))
    config["data"].update(sft_train_pairs=1, reward_train_pairs=0, reward_validation_pairs=0, improvement_prompts=0, independent_eval_prompts=0)
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        partition_pairs([
            {"prompt": " Same  prompt ", "chosen": "a", "rejected": "b"},
            {"prompt": "same prompt", "chosen": "c", "rejected": "d"},
        ], config)


def test_general_allocator_is_order_invariant():
    rows = [{"prompt": f"p{i}", "chosen": "a", "rejected": "b"} for i in range(12)]
    first = partition_allocations(rows, seed=7, allocations=[("a", 4), ("b", 3)])
    second = partition_allocations(reversed(rows), seed=7, allocations=[("a", 4), ("b", 3)])
    assert first == second
    assert_disjoint(first)


def test_general_allocator_resolves_conflicts_by_pair_hash():
    rows = [
        {"prompt": "same", "chosen": "z", "rejected": "x"},
        {"prompt": " Same ", "chosen": "a", "rejected": "b"},
    ]
    forward = partition_allocations(rows, seed=7, allocations=[("a", 1)])
    reverse = partition_allocations(reversed(rows), seed=7, allocations=[("a", 1)])
    assert forward == reverse


def test_curve_selector_prefers_saturation_for_clear_saturating_data():
    x = np.arange(1, 9, dtype=float)
    y = 0.75 - 0.30 * np.exp(-0.7 * x)
    assert select_curve(x, y)["selected"] == "saturating_exponential"


def test_mixture_assignment_and_tie_break_are_deterministic():
    values = [use_self_label(7, 25, 2, f"p{i}") for i in range(1000)]
    assert values == [use_self_label(7, 25, 2, f"p{i}") for i in range(1000)]
    assert 200 < sum(values) < 300
    assert choose_by_normalized_log_likelihood("a", "b", -1.0, -2.0)[:2] == ("a", "b")


def test_paired_bootstrap_retains_trial_count():
    result = paired_bootstrap([1, 1, 1], [0, 0, 0], replicates=100, seed=2)
    assert result["estimate"] == 1 and result["ci95"] == [1, 1] and result["n_trials"] == 3


def test_paired_sign_flip_test_is_deterministic_and_retains_trial_count():
    first = paired_sign_flip_test([1, 1, 1, 1], [0, 0, 0, 0], replicates=1000, seed=7)
    second = paired_sign_flip_test([1, 1, 1, 1], [0, 0, 0, 0], replicates=1000, seed=7)
    assert first == second and first["n_trials"] == 4 and first["observed_mean_difference"] == 1


def test_stage3_ssh_non_capture_path_executes(monkeypatch):
    from scripts import orchestrate_recursive_stage3 as orchestrator

    calls = []

    class Completed:
        stdout = "unused"

    monkeypatch.setattr(orchestrator, "run", lambda command, **kwargs: calls.append((command, kwargs)) or Completed())
    assert orchestrator.ssh_ps("Write-Output ok", capture=False) == ""
    assert len(calls) == 1
    assert calls[0][0][:4] == ["ssh", "norgate", "powershell", "-NoProfile"]


def test_stage3_jsonl_reader_preserves_unicode_line_separator(tmp_path):
    from scripts.label_recursive_stage3_round import read

    path = tmp_path / "generated.jsonl"
    path.write_text('{"text":"before\u0085after"}\n', encoding="utf-8")
    assert read(path) == [{"text": "before\u0085after"}]
