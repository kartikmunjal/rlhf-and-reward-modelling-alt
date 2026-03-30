#!/usr/bin/env python3
"""
Evaluate and compare trained policies (SFT, PPO, DPO).

Computes three metrics for each policy:
  1. Mean reward  (from the reward model)
  2. Win rate     (vs SFT baseline, using reward model as judge)
  3. KL divergence from the SFT reference policy

Usage
-----
python scripts/evaluate.py \\
    --sft_checkpoint checkpoints/sft \\
    --reward_checkpoint checkpoints/reward_model \\
    --ppo_checkpoint checkpoints/ppo \\
    --dpo_checkpoint checkpoints/dpo \\
    --num_eval 500

# Outputs a markdown table + per-model JSON files to results/
"""

import argparse
import json
import os
import sys

import torch
from transformers import AutoTokenizer, GPT2LMHeadModel
from datasets import load_dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.reward_model import GPT2RewardModel
from src.data.preprocessing import extract_prompt_and_response
from src.evaluation.metrics import (
    compute_win_rate,
    compute_reward_stats,
    compute_kl_divergence,
    generate_comparison_table,
)


def load_eval_prompts(sft_checkpoint: str, n: int) -> list[str]:
    """Load evaluation prompts from the hh-rlhf test split."""
    tokenizer = AutoTokenizer.from_pretrained(sft_checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    raw = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base", split="test")
    raw = raw.select(range(min(n, len(raw))))
    prompts = []
    for ex in raw:
        prompt, _ = extract_prompt_and_response(ex["chosen"])
        prompts.append(prompt)
    return prompts


def main():
    p = argparse.ArgumentParser(description="Evaluate SFT / PPO / DPO policies")
    p.add_argument("--sft_checkpoint", default="checkpoints/sft")
    p.add_argument("--reward_checkpoint", default="checkpoints/reward_model")
    p.add_argument("--ppo_checkpoint", default=None, help="Path to PPO checkpoint (optional)")
    p.add_argument("--dpo_checkpoint", default=None, help="Path to DPO checkpoint (optional)")
    p.add_argument("--num_eval", type=int, default=500)
    p.add_argument("--output_dir", default="results")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=128)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading tokenizer from {args.sft_checkpoint} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.sft_checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"Loading reward model from {args.reward_checkpoint} ...")
    reward_model = GPT2RewardModel.from_pretrained(args.reward_checkpoint).to(device)

    print(f"Loading SFT model ...")
    sft_model = GPT2LMHeadModel.from_pretrained(args.sft_checkpoint)

    print(f"Loading {args.num_eval} eval prompts ...")
    prompts = load_eval_prompts(args.sft_checkpoint, args.num_eval)

    # Build policy registry
    policies = {"SFT": sft_model}
    if args.ppo_checkpoint and os.path.exists(args.ppo_checkpoint):
        from transformers import AutoModelForCausalLM
        policies["PPO"] = AutoModelForCausalLM.from_pretrained(args.ppo_checkpoint)
    if args.dpo_checkpoint and os.path.exists(args.dpo_checkpoint):
        policies["DPO"] = GPT2LMHeadModel.from_pretrained(args.dpo_checkpoint)

    all_results = {}

    for name, model in policies.items():
        print(f"\nEvaluating {name} ...")
        stats = compute_reward_stats(
            model, reward_model, tokenizer, prompts, device,
            max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
        )
        kl = compute_kl_divergence(
            model, sft_model, tokenizer, prompts, device,
            max_new_tokens=64, n_samples=min(200, args.num_eval),
        )
        all_results[name] = {
            "mean_reward": stats["mean"],
            "std_reward": stats["std"],
            "kl_from_ref": kl,
        }

        if name != "SFT":
            win_metrics = compute_win_rate(
                model, sft_model, reward_model, tokenizer,
                prompts[:200], device,
                max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
            )
            all_results[name]["win_rate_vs_sft"] = win_metrics["win_rate_a"]
            all_results[name]["mean_reward"] = win_metrics["mean_reward_a"]
        else:
            all_results[name]["win_rate_vs_sft"] = float("nan")

        json_path = os.path.join(args.output_dir, f"{name.lower()}_metrics.json")
        with open(json_path, "w") as f:
            json.dump(all_results[name], f, indent=2)
        print(f"  → {json_path}")

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    table = generate_comparison_table(all_results)
    print(table)

    summary_path = os.path.join(args.output_dir, "summary.md")
    with open(summary_path, "w") as f:
        f.write("# Evaluation Results\n\n")
        f.write(table + "\n")
    print(f"\nMarkdown summary saved to {summary_path}")


if __name__ == "__main__":
    main()
