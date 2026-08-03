#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.ppo_grpo_v2 import evaluate_v2, load_config, train_grpo_v2, train_ppo_v2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("ppo", "grpo", "baseline"), required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--config", type=Path, default=Path("configs/ppo_grpo_v2.json"))
    parser.add_argument("--output-root", type=Path, default=Path("results/ppo_grpo_v2"))
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.smoke:
        config["compute_budget"]["optimizer_steps"] = 4
        config["compute_budget"]["rollout_groups"] = 2
        config["compute_budget"]["expected_generated_completions"] = 8
        config["runtime_assertions"] = {
            "optimizer_steps_must_equal": 4,
            "rollout_groups_must_equal": 2,
            "generated_completions_must_equal": 8,
        }
    if args.method == "baseline":
        directory = args.output_root / "baseline_sft"
        directory.mkdir(parents=True, exist_ok=True)
        rows = evaluate_v2(config, None)
        (directory / "predictions.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(json.dumps({"method": "baseline", "evaluation_rows": len(rows)}))
        return
    if args.seed not in config["paired_seeds"]:
        raise ValueError("Training requires a preregistered seed")
    directory = args.output_root / (f"{args.method}_seed{args.seed}_smoke" if args.smoke else f"{args.method}_seed{args.seed}")
    trainer = train_ppo_v2 if args.method == "ppo" else train_grpo_v2
    manifest = trainer(config, args.seed, directory)
    if args.evaluate and not args.smoke:
        rows = evaluate_v2(config, directory)
        (directory / "predictions.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        manifest["numeric_exact"] = sum(row["numeric_exact"] for row in rows) / len(rows)
        manifest["tagged_exact"] = sum(row["tagged_exact"] for row in rows) / len(rows)
        (directory / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
