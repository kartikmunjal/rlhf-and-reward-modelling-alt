#!/usr/bin/env python3
"""Fail-closed integrity audit for the frozen Stage-3 experiment matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recursive_self_improvement.config import load_effective_config
from recursive_self_improvement.mixture import use_self_label


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/recursive_self_improvement_v1/stage3_integrity_audit.json"),
    )
    args = parser.parse_args()
    config = load_effective_config(ROOT)
    expected_rows = config["stage1"]["rollout_prompts_per_round"]
    allocation = config["stage3"]["prompt_allocation"]
    improvement = read_jsonl(ROOT / allocation["source"])
    expected_by_round = {
        round_index: [row["prompt_id"] for row in improvement[start:stop]]
        for round_index, (start, stop) in enumerate(allocation["round_blocks"], 1)
    }
    require(
        len({prompt_id for ids in expected_by_round.values() for prompt_id in ids})
        == expected_rows * config["stage3"]["rounds_per_condition"],
        "Stage-3 prompt blocks are not disjoint",
    )
    conditions = []
    seen_within_condition: dict[int, set[str]] = {}
    for percent in config["stage3"]["label_mixture_percent_self"]:
        seen_within_condition[percent] = set()
        for round_index in range(1, config["stage3"]["rounds_per_condition"] + 1):
            result_dir = (
                ROOT
                / "results/recursive_self_improvement_v1/stage3"
                / f"self_{percent}/round_{round_index}"
            )
            candidate_path = result_dir / "candidates_with_self_scores.jsonl"
            preference_path = result_dir / "preferences.jsonl"
            manifest_path = (
                ROOT
                / "checkpoints/recursive_self_improvement_v1"
                / f"stage3_self_{percent}_round_{round_index}/run_manifest.json"
            )
            candidates = read_jsonl(candidate_path)
            preferences = read_jsonl(preference_path)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            require(len(candidates) == expected_rows, f"Wrong candidate count: {candidate_path}")
            require(len(preferences) == expected_rows, f"Wrong preference count: {preference_path}")
            candidate_ids = [row["prompt_id"] for row in candidates]
            preference_ids = [row["prompt_id"] for row in preferences]
            require(candidate_ids == expected_by_round[round_index], f"Candidate prompt allocation mismatch: {candidate_path}")
            require(preference_ids == expected_by_round[round_index], f"Preference prompt allocation mismatch: {preference_path}")
            require(not (seen_within_condition[percent] & set(candidate_ids)), f"Prompt reused in condition {percent}")
            seen_within_condition[percent].update(candidate_ids)
            for row in preferences:
                expected_source = "self" if use_self_label(
                    config["seed"], percent, round_index, row["prompt_id"]
                ) else "external"
                require(row["label_source"] == expected_source, f"Mixture assignment mismatch: {row['prompt_id']}")
            require(manifest.get("status") == "complete", f"Incomplete checkpoint: {manifest_path}")
            require(manifest.get("optimizer_steps") == config["stage1"]["dpo_steps_per_round"], f"Wrong optimizer steps: {manifest_path}")
            require(manifest.get("preference_rows") in (expected_rows, expected_rows * 2), f"Wrong rolling buffer size: {manifest_path}")
            require(manifest.get("preference_sha256") == sha256(preference_path) or round_index > 1, f"Round-1 preference hash mismatch: {manifest_path}")
            expected_source = (
                "checkpoints\\recursive_self_improvement_v1\\iterative_dpo_round_3"
                if round_index == 1
                else f"checkpoints\\recursive_self_improvement_v1\\stage3_self_{percent}_round_{round_index - 1}"
            )
            require(manifest.get("source_adapter") == expected_source, f"Checkpoint chain mismatch: {manifest_path}")
            conditions.append(
                {
                    "percent_self": percent,
                    "round": round_index,
                    "candidate_rows": len(candidates),
                    "preference_rows_current_round": len(preferences),
                    "self_labels": sum(row["label_source"] == "self" for row in preferences),
                    "external_labels": sum(row["label_source"] == "external" for row in preferences),
                    "optimizer_steps": manifest["optimizer_steps"],
                    "checkpoint_manifest_sha256": sha256(manifest_path),
                }
            )
    expected_conditions = len(config["stage3"]["label_mixture_percent_self"]) * config["stage3"]["rounds_per_condition"]
    require(len(conditions) == expected_conditions, "Incomplete Stage-3 condition matrix")
    payload = {
        "study_id": config["study_id"],
        "status": "pass",
        "evaluation_ensemble_used_for_training": False,
        "conditions_verified": len(conditions),
        "expected_conditions": expected_conditions,
        "prompts_per_round": expected_rows,
        "unique_prompts_per_condition": expected_rows * config["stage3"]["rounds_per_condition"],
        "same_prompt_blocks_across_conditions": True,
        "conditions": conditions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
