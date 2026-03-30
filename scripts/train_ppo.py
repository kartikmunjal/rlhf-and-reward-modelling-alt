#!/usr/bin/env python3
"""
Train a PPO policy — Stage 3a of the RLHF pipeline.

Requires completed Stage 1 (SFT) and Stage 2 (Reward Model) checkpoints.

Usage
-----
python scripts/train_ppo.py \\
    --sft_checkpoint checkpoints/sft \\
    --reward_checkpoint checkpoints/reward_model \\
    --output_dir checkpoints/ppo \\
    --num_samples 5000

# Quick test (less memory, fewer samples)
python scripts/train_ppo.py \\
    --sft_checkpoint checkpoints/sft \\
    --reward_checkpoint checkpoints/reward_model \\
    --batch_size 8 \\
    --num_samples 500
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.training.ppo import PPOTrainingConfig, train_ppo


def parse_args() -> PPOTrainingConfig:
    p = argparse.ArgumentParser(description="Stage 3a: PPO Policy Optimization")
    p.add_argument("--sft_checkpoint", default="checkpoints/sft")
    p.add_argument("--reward_checkpoint", default="checkpoints/reward_model")
    p.add_argument("--output_dir", default="checkpoints/ppo")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--mini_batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1.41e-5)
    p.add_argument("--kl_coef", type=float, default=0.2,
                   help="Initial KL penalty coefficient β")
    p.add_argument("--target_kl", type=float, default=6.0,
                   help="Target KL for adaptive controller")
    p.add_argument("--num_samples", type=int, default=5_000)
    p.add_argument("--max_new_tokens", type=int, default=128)
    args = p.parse_args()

    return PPOTrainingConfig(
        sft_checkpoint=args.sft_checkpoint,
        reward_checkpoint=args.reward_checkpoint,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        learning_rate=args.lr,
        init_kl_coef=args.kl_coef,
        target_kl=args.target_kl,
        num_train_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    cfg = parse_args()
    print("=" * 60)
    print("Stage 3a: PPO Policy Optimization")
    print(f"  sft_checkpoint    : {cfg.sft_checkpoint}")
    print(f"  reward_checkpoint : {cfg.reward_checkpoint}")
    print(f"  output_dir        : {cfg.output_dir}")
    print(f"  batch_size        : {cfg.batch_size}")
    print(f"  init_kl_coef      : {cfg.init_kl_coef}")
    print(f"  target_kl         : {cfg.target_kl}")
    print("=" * 60)
    train_ppo(cfg)
