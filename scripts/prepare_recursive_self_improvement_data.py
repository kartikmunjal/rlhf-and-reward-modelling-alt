#!/usr/bin/env python3
"""Materialize hash-ranked, prompt-disjoint HH-RLHF study partitions."""

import argparse, hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from recursive_self_improvement.config import load_effective_config
from recursive_self_improvement.data import NORMALIZATION_VERSION, assert_disjoint, partition_allocations, prompt_id, sha256, stable_rank, write_jsonl
from src.data.preprocessing import extract_prompt_and_response


def normalize_row(row):
    chosen_prompt, chosen = extract_prompt_and_response(row["chosen"])
    rejected_prompt, rejected = extract_prompt_and_response(row["rejected"])
    if prompt_id(chosen_prompt) != prompt_id(rejected_prompt):
        raise ValueError("Chosen and rejected transcripts have different prompts")
    return {"prompt": chosen_prompt, "chosen": chosen, "rejected": rejected}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", required=True, help="Immutable HH-RLHF dataset commit")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/recursive_self_improvement_v1"))
    args = parser.parse_args()
    from datasets import load_dataset
    config = load_effective_config(ROOT); counts = config["data"]
    dataset = load_dataset(counts["dataset"], data_dir=counts["subset"], revision=args.revision)
    partitions = partition_allocations(map(normalize_row, dataset["train"]), seed=config["seed"], allocations=[
        ("sft_train", counts["sft_train_pairs"]), ("reward_train", counts["reward_train_pairs"]),
        ("reward_validation", counts["reward_validation_pairs"]), ("improvement", counts["improvement_prompts"]),
    ])
    train_ids = {row["prompt_id"] for values in partitions.values() for row in values}
    test_unique = {}
    for row in map(normalize_row, dataset["test"]):
        row["prompt_id"] = prompt_id(row["prompt"])
        if row["prompt_id"] not in train_ids: test_unique.setdefault(row["prompt_id"], row)
    ordered = sorted(test_unique.values(), key=lambda row: stable_rank(config["seed"], row["prompt"]))
    needed = counts["independent_eval_prompts"]
    if len(ordered) < needed: raise ValueError(f"Need {needed} disjoint test rows, found {len(ordered)}")
    partitions["independent_eval"] = ordered[:needed]; assert_disjoint(partitions)
    args.output_dir.mkdir(parents=True, exist_ok=True); files = {}
    for name, rows in partitions.items():
        path = args.output_dir / f"{name}.jsonl"; write_jsonl(path, rows)
        files[name] = {"path": str(path), "rows": len(rows), "sha256": sha256(path)}
    manifest = {
        "schema_version": 1, "study_id": config["study_id"], "dataset": counts["dataset"], "subset": counts["subset"],
        "revision": args.revision, "dataset_fingerprints": {name: split._fingerprint for name, split in dataset.items()},
        "normalization_version": NORMALIZATION_VERSION, "applied_amendments": config["applied_amendments"], "files": files,
        "selected_prompt_ids_sha256": hashlib.sha256("\n".join(sorted(row["prompt_id"] for values in partitions.values() for row in values)).encode()).hexdigest(),
    }
    destination = args.output_dir / "data_manifest.json"
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {destination}")


if __name__ == "__main__": main()
