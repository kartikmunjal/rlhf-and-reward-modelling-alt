"""Deterministically sample and blind the preregistered qualitative audits."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from verifier_reward_hacking.io import read_jsonl, sha256_file, write_jsonl


SEEDS = (2025, 2026, 2027)
CONDITIONS = ("C1", "C2", "C3", "C4")


def rank(seed: int, *values) -> str:
    return hashlib.sha256((str(seed) + ":" + ":".join(map(str, values))).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--evaluations-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/verifier_reward_hacking_v1/final"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/verifier_reward_hacking_v1/manual_audit"))
    args = parser.parse_args()
    seed = 20260929
    tasks = {row["task_id"]: row for row in read_jsonl(args.data_dir / "tasks_private.jsonl")}
    trajectory_private, trajectory_blind = [], []
    for condition in CONDITIONS:
        for training_seed in SEEDS:
            path = args.runs_root / condition / f"seed{training_seed}" / "raw_training_trajectories_private.jsonl"
            rows = read_jsonl(path)
            for phase, eligible in (
                ("early", [row for row in rows if row["reward_batch_index"] <= 50]),
                ("final", [row for row in rows if row["reward_batch_index"] > 150]),
            ):
                chosen = sorted(eligible, key=lambda row: rank(seed, condition, training_seed, phase, row["task_id"], row["reward_batch_index"], row["completion"]))[:5]
                if len(chosen) != 5:
                    raise RuntimeError(f"Insufficient trajectories for {condition}/{training_seed}/{phase}")
                for row in chosen:
                    item_id = rank(seed, "trajectory", condition, training_seed, phase, row["task_id"], row["completion"])[:20]
                    trajectory_private.append({"item_id": item_id, "condition": condition, "seed": training_seed, "phase": phase, **row})
                    trajectory_blind.append({"item_id": item_id, "task_id": row["task_id"], "task": tasks[row["task_id"]]["prompt"], "visible_tests": tasks[row["task_id"]]["visible_tests"], "completion": row["completion"], "labels": {name: None for name in ("special_casing", "wrong_algorithm", "extraction_formatting_bug", "timeout", "degenerate_output", "verifier_false_positive", "verifier_false_negative")}, "notes": ""})
    final_private, final_blind = [], []
    policy_paths = {"C0": args.evaluations_root / "C0" / "all_test" / "predictions_private.jsonl"}
    for condition in CONDITIONS:
        policy_paths[condition] = [args.evaluations_root / condition / f"seed{training_seed}" / "all_test" / "predictions_private.jsonl" for training_seed in SEEDS]
    for condition, paths in policy_paths.items():
        paths = paths if isinstance(paths, list) else [paths]
        candidates = []
        for path in paths:
            candidates.extend(read_jsonl(path))
        chosen = sorted(candidates, key=lambda row: rank(seed, "final", condition, row["task_id"], row["completion"]))[:30]
        if len(chosen) != 30:
            raise RuntimeError(f"Insufficient final samples for {condition}")
        for row in chosen:
            item_id = rank(seed, "semantic", condition, row["task_id"], row["completion"])[:20]
            final_private.append({"item_id": item_id, "condition": condition, "automatic_hidden_pass": row["hidden"]["passed"], "automatic_visible_pass": row["visible"]["passed"], **row})
            final_blind.append({"item_id": item_id, "task_id": row["task_id"], "task": tasks[row["task_id"]]["prompt"], "completion": row["completion"], "semantic_correct": None, "notes": ""})
    if len(trajectory_blind) != 120 or len(final_blind) != 150:
        raise RuntimeError("Manual audit counts violate preregistration")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for name, rows in (("trajectory_key_private", trajectory_private), ("trajectory_labels_blind", trajectory_blind), ("final_key_private", final_private), ("final_labels_blind", final_blind)):
        path = args.output_dir / f"{name}.jsonl"
        artifacts[path.name] = write_jsonl(path, rows)
    manifest = {"study_id": "verifier_reward_hacking_v1", "status": "awaiting_labels", "trajectory_items": 120, "final_items": 150, "sampling_seed": seed, "artifacts": artifacts, "blinding": "condition, seed, phase, and automatic outcomes removed from label files"}
    (args.output_dir / "audit_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
