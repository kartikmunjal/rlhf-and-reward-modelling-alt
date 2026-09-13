#!/usr/bin/env python3
"""Fail-closed completion verifier for recursive_self_improvement_v1."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recursive_self_improvement.config import load_effective_config


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    config = load_effective_config(ROOT)
    result = ROOT / "results/recursive_self_improvement_v1"
    metrics = load(result / "metrics.json")
    data_manifest_path = ROOT / "data/processed/recursive_self_improvement_v1/data_manifest.json"
    data_manifest = load(data_manifest_path)
    integrity = load(result / "stage3_integrity_audit.json")
    judge = load(result / "judge_audit.json")
    evaluation_manifest = load(result / "stage3_evaluation/evaluation_manifest.json")
    label_manifest = load(result / "stage3_training_label_audit/audit_manifest.json")
    evaluations = result / "stage3_evaluations.jsonl"
    label_audit = result / "stage3_training_label_audit.jsonl"

    expected_conditions = len(config["stage3"]["label_mixture_percent_self"]) * config["stage3"]["rounds_per_condition"]
    expected_eval_rows = expected_conditions * config["data"]["independent_eval_prompts"]
    expected_label_rows = expected_conditions * config["stage1"]["rollout_prompts_per_round"]
    expected_judged_pairs = (len(config["scope"]["primary_family"]) - 1) * config["data"]["independent_eval_prompts"]

    partition_ids: dict[str, set[str]] = {}
    for name, entry in data_manifest["files"].items():
        path = ROOT / entry["path"]
        require(path.exists(), f"Missing frozen data partition: {name}")
        require(sha256(path) == entry["sha256"], f"Frozen data hash mismatch: {name}")
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        require(len(rows) == entry["rows"], f"Frozen data row-count mismatch: {name}")
        partition_ids[name] = {row["prompt_id"] for row in rows}
        require(len(partition_ids[name]) == len(rows), f"Duplicate prompt in frozen partition: {name}")
    names = sorted(partition_ids)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            require(not (partition_ids[left] & partition_ids[right]), f"Prompt leakage between {left} and {right}")

    require(integrity["status"] == "pass", "Stage-3 matrix integrity audit did not pass")
    require(integrity["conditions_verified"] == expected_conditions, "Stage-3 matrix is incomplete")
    require(not integrity["evaluation_ensemble_used_for_training"], "Evaluation ensemble leaked into training")
    require(evaluation_manifest["status"] == "complete", "Stage-3 evaluation is incomplete")
    require(evaluation_manifest["rows"] == expected_eval_rows, "Wrong Stage-3 evaluation row count")
    require(evaluation_manifest["sha256"] == sha256(evaluations), "Stage-3 evaluation hash mismatch")
    require(label_manifest["status"] == "complete" and label_manifest["evaluation_only"], "Training-label audit is incomplete or not evaluation-only")
    require(label_manifest["rows"] == expected_label_rows, "Wrong training-label audit row count")
    require(label_manifest["sha256"] == sha256(label_audit), "Training-label audit hash mismatch")
    require(judge["successful_calls"] == expected_judged_pairs * 2, "Wrong independent-judge call count")
    require(judge["judged_prompt_checkpoint_pairs"] == expected_judged_pairs, "Wrong independent-judge pair denominator")
    require(metrics["reward_ensemble_validation"]["status"] == "pass", "Reward ensemble gate failed")
    require(len(metrics["stage2_checkpoints_in_fit"]) == len(config["scope"]["primary_family"]), "Stage-2 checkpoint family incomplete")
    for percent in config["stage3"]["label_mixture_percent_self"]:
        rounds = metrics["stage3_round_metrics"][str(percent)]
        require(len(rounds) == config["stage3"]["rounds_per_condition"], f"Incomplete metrics for {percent}% self")
        require(all(row["external_evaluator_win_rate"] is not None for row in rounds.values()), f"Missing evaluator metrics for {percent}% self")
        require(len(metrics["stage3_training_label_agreement"][str(percent)]) == config["stage3"]["rounds_per_condition"], f"Missing label agreement for {percent}% self")
    for percent in config["stage3"]["label_mixture_percent_self"]:
        if percent:
            comparison = metrics["stage3_vs_zero_percent"][str(percent)]
            require(comparison["n_trials"] == config["data"]["independent_eval_prompts"], f"Wrong paired N for {percent}% self")
            require("holm_adjusted_p_value" in comparison, f"Missing Holm result for {percent}% self")
    figures = [
        result / "figures/stage1_round_curve.svg",
        result / "figures/stage2_capability_compute.svg",
        result / "figures/stage3_self_reliance.svg",
    ]
    require(all(path.exists() and path.stat().st_size > 0 for path in figures), "A generated figure is missing")
    for readme in (ROOT / "README.md", ROOT / "recursive_self_improvement/README.md"):
        text = readme.read_text(encoding="utf-8")
        require(text.count("<!-- recursive-self-improvement-results:start -->") == 1, f"Missing or duplicate generated block in {readme}")
        require(text.count("<!-- recursive-self-improvement-results:end -->") == 1, f"Missing or duplicate generated block in {readme}")
    tracked_artifacts = [
        result / "metrics.json", result / "report.md", result / "compute.jsonl",
        data_manifest_path,
        result / "reward_ensemble_audit.json", result / "judge_audit.json",
        result / "stage3_integrity_audit.json", *figures,
        result / "stage3_evaluation/evaluation_manifest.json",
        result / "stage3_training_label_audit/audit_manifest.json",
    ]
    payload = {
        "study_id": config["study_id"],
        "status": "pass",
        "requirements_verified": {
            "stage3_conditions": expected_conditions,
            "stage3_evaluation_rows": expected_eval_rows,
            "stage3_training_labels_audited": expected_label_rows,
            "stage2_checkpoints": len(config["scope"]["primary_family"]),
            "independent_judge_calls": judge["successful_calls"],
            "generated_figures": len(figures),
            "disjoint_data_partitions": len(partition_ids),
        },
        "published_artifact_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in tracked_artifacts},
        "raw_example_level_artifacts_excluded_from_git": [
            str(evaluations.relative_to(ROOT)), str(label_audit.relative_to(ROOT)),
            "results/recursive_self_improvement_v1/claude_pairwise.jsonl",
        ],
    }
    destination = result / "completion_audit.json"
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
