import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "verifier_reward_hacking"


def test_preregistration_is_frozen_and_hashes_match():
    manifest = json.loads((MODULE / "preregistration_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "frozen_before_data_or_results"
    for relative, metadata in manifest["files"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == metadata["sha256"]


def test_locked_design_counts_and_conditions():
    config = json.loads((MODULE / "study_config.json").read_text(encoding="utf-8"))
    splits = config["tasks"]["splits"]
    assert sum(splits.values()) == config["tasks"]["total"] == 200
    assert config["conditions"] == {
        "C0": "frozen_base",
        "C1": "grpo_v0",
        "C2": "grpo_v2",
        "C3": "grpo_v3",
        "C4": "grpo_v4",
    }
    assert config["paired_seeds"] == [2025, 2026, 2027]
    assert config["training"]["optimizer_steps"] == 200
    assert config["evaluation"]["bootstrap_replicates"] == 2000


def test_hidden_tests_cannot_be_mounted_into_sandbox():
    config = json.loads((MODULE / "study_config.json").read_text(encoding="utf-8"))
    sandbox = config["sandbox"]
    assert sandbox["network"] == "none"
    assert sandbox["read_only_root"] is True
    assert sandbox["hidden_tests_mounted"] is False
