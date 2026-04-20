#!/usr/bin/env python3
"""
Extension 2 Addendum: ensemble uncertainty-penalty lambda sweep.

Research question
-----------------
Does a small uncertainty penalty (lambda=0.1 or 0.3) capture most of the
benefit of ensemble reward modeling, or is lambda=0.5 required to materially
reduce verbose-bias reward hacking?

This script evaluates a fixed set of ensemble-PPO checkpoints across lambda
values on the same prompt set and reports:
  1. Mean penalized ensemble reward
  2. KL from SFT reference
  3. Verbose-bias rate

Usage
-----
  # Evaluate existing checkpoints
  python scripts/run_ensemble_lambda_sweep.py

  # Train missing checkpoints, then evaluate
  python scripts/run_ensemble_lambda_sweep.py --train_missing
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def _format_lambda_tag(lam: float) -> str:
    return str(lam).replace(".", "p")


def load_eval_prompts(sft_checkpoint: str, n: int) -> list[str]:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    from src.data.preprocessing import extract_prompt_and_response

    tokenizer = AutoTokenizer.from_pretrained(sft_checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    raw = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base", split="test")
    raw = raw.select(range(min(n, len(raw))))
    prompts = []
    for ex in raw:
        prompt, _ = extract_prompt_and_response(ex["chosen"])
        prompts.append(prompt)
    return prompts


def compute_penalized_reward_stats(
    policy,
    ensemble: RewardEnsemble,
    tokenizer,
    prompts: list[str],
    device,
    penalty: float,
    max_new_tokens: int = 128,
    batch_size: int = 8,
) -> dict:
    import torch

    policy.eval()
    ensemble.eval()
    policy.to(device)
    ensemble.to(device)

    rewards = []
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            ).to(device)
            out = policy.generate(**enc, **gen_kwargs)
            texts = tokenizer.batch_decode(out, skip_special_tokens=True)
            enc_r = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)
            batch_rewards = ensemble.penalized_reward(
                enc_r["input_ids"],
                enc_r["attention_mask"],
                penalty_override=penalty,
            )
            rewards.extend([float(r) for r in batch_rewards])

    t = torch.tensor(rewards)
    return {
        "mean_reward": t.mean().item(),
        "std_reward": t.std().item() if len(rewards) > 1 else 0.0,
    }


def maybe_train_checkpoint(args, lam: float, output_dir: str) -> None:
    if Path(output_dir).exists():
        return
    if not args.train_missing:
        return

    cmd = [
        sys.executable,
        "scripts/train_ppo_ensemble.py",
        "--sft_checkpoint", args.sft_checkpoint,
        "--ensemble_dir", args.ensemble_dir,
        "--uncertainty_penalty", str(lam),
        "--output_dir", output_dir,
    ]
    print(f"Training missing checkpoint for lambda={lam} -> {output_dir}")
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate ensemble-PPO lambda sweep")
    p.add_argument("--sft_checkpoint", default="checkpoints/sft")
    p.add_argument("--ensemble_dir", default="checkpoints/reward_ensemble")
    p.add_argument("--policy_dir_template", default="checkpoints/ppo_ensemble_lam{lam_tag}")
    p.add_argument("--lambdas", nargs="+", type=float, default=[0.1, 0.3, 0.5])
    p.add_argument("--num_eval_prompts", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--output", default="results/ensemble_lambda_sweep.json")
    p.add_argument("--train_missing", action="store_true",
                   help="Train missing lambda checkpoints before evaluation")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    from transformers import AutoTokenizer, GPT2LMHeadModel

    from src.evaluation.metrics import compute_kl_divergence
    from src.models.reward_ensemble import RewardEnsemble
    from scripts.run_ppo_reward_comparison import compute_verbose_bias_rate

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not Path(args.sft_checkpoint).exists():
        raise SystemExit(f"SFT checkpoint not found: {args.sft_checkpoint}")
    if not Path(args.ensemble_dir).exists():
        raise SystemExit(f"Reward ensemble directory not found: {args.ensemble_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.sft_checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    manifest = Path(args.ensemble_dir) / "manifest.txt"
    if manifest.exists():
        ckpts = [line.strip() for line in manifest.read_text().splitlines() if line.strip()]
    else:
        ckpts = sorted(
            str(Path(args.ensemble_dir) / d)
            for d in os.listdir(args.ensemble_dir)
            if d.startswith("model_")
        )
    ensemble = RewardEnsemble.from_checkpoints(ckpts, uncertainty_penalty=0.5)
    sft_model = GPT2LMHeadModel.from_pretrained(args.sft_checkpoint).to(device)
    prompts = load_eval_prompts(args.sft_checkpoint, args.num_eval_prompts)

    results = []
    for lam in args.lambdas:
        lam_tag = _format_lambda_tag(lam)
        policy_dir = args.policy_dir_template.format(lam=lam, lam_tag=lam_tag)
        maybe_train_checkpoint(args, lam, policy_dir)

        row = {
            "lambda": lam,
            "policy_dir": policy_dir,
            "available": Path(policy_dir).exists(),
        }
        if not row["available"]:
            row["status"] = "missing_checkpoint"
            results.append(row)
            continue

        policy = GPT2LMHeadModel.from_pretrained(policy_dir).to(device)
        reward_stats = compute_penalized_reward_stats(
            policy,
            ensemble,
            tokenizer,
            prompts,
            device,
            penalty=lam,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
        )
        vb_rate, mean_len = compute_verbose_bias_rate(
            policy,
            tokenizer,
            prompts,
            device,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
        )
        tokenizer.padding_side = "right"
        kl = compute_kl_divergence(
            policy,
            sft_model,
            tokenizer,
            prompts,
            device,
            n_samples=min(200, len(prompts)),
        )
        tokenizer.padding_side = "left"

        row.update(
            {
                "status": "ok",
                "mean_reward": reward_stats["mean_reward"],
                "std_reward": reward_stats["std_reward"],
                "kl_from_ref": kl,
                "verbose_bias_rate": vb_rate,
                "mean_length_tokens": mean_len,
            }
        )
        results.append(row)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(
            {
                "config": {
                    "sft_checkpoint": args.sft_checkpoint,
                    "ensemble_dir": args.ensemble_dir,
                    "lambdas": args.lambdas,
                    "num_eval_prompts": args.num_eval_prompts,
                },
                "results": results,
            },
            f,
            indent=2,
        )

    print("\nEnsemble lambda sweep")
    print(f"{'lambda':<8} {'reward':>8} {'KL':>8} {'verbose%':>10} {'status':>18}")
    print("-" * 60)
    for row in results:
        if row["status"] != "ok":
            print(f"{row['lambda']:<8.1f} {'-':>8} {'-':>8} {'-':>10} {row['status']:>18}")
            continue
        print(
            f"{row['lambda']:<8.1f} "
            f"{row['mean_reward']:>8.3f} "
            f"{row['kl_from_ref']:>8.3f} "
            f"{row['verbose_bias_rate']:>9.0%} "
            f"{row['status']:>18}"
        )
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
