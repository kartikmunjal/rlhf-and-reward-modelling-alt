#!/usr/bin/env python3
"""Run one preregistered PPO/GRPO seed or its excluded smoke test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.ppo_grpo_v1 import evaluate_adapter, load_config, train_grpo, train_ppo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("ppo", "grpo"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/ppo_grpo_v1.json"))
    parser.add_argument("--output-root", type=Path, default=Path("results/ppo_grpo_v1"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.seed not in config["paired_seeds"] and not args.smoke:
        raise ValueError("Full runs must use a preregistered paired seed")
    label = f"{args.method}_seed{args.seed}" + ("_smoke" if args.smoke else "")
    output_dir = args.output_root / label
    train = train_ppo if args.method == "ppo" else train_grpo
    manifest = train(config, args.seed, output_dir, args.smoke)
    if args.evaluate and not args.smoke:
        predictions = evaluate_adapter(config, output_dir)
        (output_dir / "predictions.json").write_text(json.dumps(predictions, indent=2), encoding="utf-8")
        manifest["exact_match"] = sum(row["correct"] for row in predictions) / len(predictions)
        (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
