#!/usr/bin/env python3
"""Train and compare PPO vs GRPO under matched checkpoints and prompt budgets."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.evaluate import load_eval_prompts
from src.evaluation.metrics import compute_kl_divergence, compute_win_rate
from src.models.reward_model import GPT2RewardModel
from src.training.grpo import GRPOTrainingConfig, train_grpo
from src.training.ppo import PPOTrainingConfig, train_ppo


def _compare_table(rows: list[dict]) -> str:
    headers = ["Method", "Win vs SFT", "KL from Ref", "Train Time (s)", "GPU Memory (MB)"]
    body = []
    for row in rows:
        body.append(
            [
                row["method"],
                f"{row['win_rate_vs_sft'] * 100:.1f}%",
                f"{row['kl_from_ref']:.4f}",
                f"{row['training_time_seconds']:.2f}",
                "n/a" if row["peak_memory_mb"] < 0 else f"{row['peak_memory_mb']:.1f}",
            ]
        )

    widths = [max(len(str(item[i])) for item in [headers] + body) for i in range(len(headers))]
    fmt = "| " + " | ".join(f"{{:<{w}}}" for w in widths) + " |"
    sep = "|-" + "-|-".join("-" * w for w in widths) + "-|"
    return "\n".join([fmt.format(*headers), sep, *[fmt.format(*row) for row in body]])


def main() -> None:
    p = argparse.ArgumentParser(description="Compare PPO vs GRPO under matched prompt budgets")
    p.add_argument("--sft_checkpoint", default="checkpoints/sft")
    p.add_argument("--reward_checkpoint", default="checkpoints/reward_model")
    p.add_argument("--ppo_output_dir", default="checkpoints/ppo_compare")
    p.add_argument("--grpo_output_dir", default="checkpoints/grpo_compare")
    p.add_argument("--num_samples", type=int, default=2000)
    p.add_argument("--num_eval", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_generations", type=int, default=4)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--epsilon", type=float, default=0.2)
    p.add_argument("--output_dir", default="results/ppo_grpo")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ppo_stats = train_ppo(
        PPOTrainingConfig(
            sft_checkpoint=args.sft_checkpoint,
            reward_checkpoint=args.reward_checkpoint,
            output_dir=args.ppo_output_dir,
            batch_size=args.batch_size,
            mini_batch_size=max(1, args.batch_size // 2),
            num_train_samples=args.num_samples,
        )
    )
    grpo_stats = train_grpo(
        GRPOTrainingConfig(
            sft_checkpoint=args.sft_checkpoint,
            reward_checkpoint=args.reward_checkpoint,
            output_dir=args.grpo_output_dir,
            per_device_train_batch_size=args.batch_size,
            num_train_samples=args.num_samples,
            num_generations=args.num_generations,
            beta=args.beta,
            epsilon=args.epsilon,
        )
    )

    tokenizer = AutoTokenizer.from_pretrained(args.sft_checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    reward_model = GPT2RewardModel.from_pretrained(args.reward_checkpoint).to(device)
    sft_model = GPT2LMHeadModel.from_pretrained(args.sft_checkpoint).to(device)
    ppo_model = AutoModelForCausalLM.from_pretrained(args.ppo_output_dir).to(device)
    grpo_model = AutoModelForCausalLM.from_pretrained(args.grpo_output_dir).to(device)
    prompts = load_eval_prompts(args.sft_checkpoint, args.num_eval)

    ppo_win = compute_win_rate(ppo_model, sft_model, reward_model, tokenizer, prompts, device)
    grpo_win = compute_win_rate(grpo_model, sft_model, reward_model, tokenizer, prompts, device)
    ppo_kl = compute_kl_divergence(ppo_model, sft_model, tokenizer, prompts, device, n_samples=min(100, args.num_eval))
    grpo_kl = compute_kl_divergence(grpo_model, sft_model, tokenizer, prompts, device, n_samples=min(100, args.num_eval))

    rows = [
        {
            "method": "PPO",
            "win_rate_vs_sft": ppo_win["win_rate_a"],
            "kl_from_ref": ppo_kl,
            **ppo_stats,
        },
        {
            "method": "GRPO",
            "win_rate_vs_sft": grpo_win["win_rate_a"],
            "kl_from_ref": grpo_kl,
            **grpo_stats,
        },
    ]

    table = _compare_table(rows)
    summary = {
        "config": vars(args),
        "results": rows,
        "table": table,
    }

    with open(Path(args.output_dir) / "ppo_grpo_comparison.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(Path(args.output_dir) / "ppo_grpo_comparison.md", "w") as f:
        f.write("# PPO vs GRPO\n\n")
        f.write(table + "\n")

    print(table)


if __name__ == "__main__":
    main()
