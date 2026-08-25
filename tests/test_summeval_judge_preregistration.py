import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "llm_judge_summeval"


def test_stage0_manifest_matches_locked_sources():
    manifest = json.loads((MODULE / "stage0_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "frozen_before_data_or_results"
    for name, metadata in manifest["files"].items():
        assert hashlib.sha256((MODULE / name).read_bytes()).hexdigest() == metadata["sha256"]


def test_confirmatory_design_is_locked_and_document_disjoint():
    config = json.loads((MODULE / "study_config.json").read_text(encoding="utf-8"))
    assert config["status"] == "preregistered_not_run"
    assert config["dataset"]["split_unit"] == "source_article"
    assert config["dataset"]["dev_articles"] + config["dataset"]["heldout_articles"] == 100
    assert config["axes"]["primary"] == ["relevance", "consistency"]
    assert config["statistics"]["correlation"] == "spearman"
    assert config["success_rule"]["minimum_point_estimate_rho"] == 0.4
    assert config["success_rule"]["judge_minus_rouge_ci_must_exclude_zero_positive"] is True


def test_models_are_pinned_snapshots_and_secondary_cannot_rescue_primary():
    config = json.loads((MODULE / "study_config.json").read_text(encoding="utf-8"))
    assert config["judges"]["primary"]["model"] == "claude-haiku-4-5-20251001"
    assert config["judges"]["secondary"]["model"] == "gpt-5-mini-2025-08-07"
    assert config["judges"]["secondary"]["scope"] == "pointwise_cross_provider_only"
