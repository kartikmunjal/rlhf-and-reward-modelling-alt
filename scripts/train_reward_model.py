#!/usr/bin/env python3
"""
Train the reward model — Stage 2 of the RLHF pipeline.

The reward model is initialised from the SFT checkpoint and fine-tuned on
(chosen, rejected) preference pairs using the Bradley-Terry loss.

Usage
-----
# Requires a trained SFT checkpoint at checkpoints/sft
python scripts/train_reward_model.py

# Custom paths
python scripts/train_reward_model.py \\
    --sft_checkpoint checkpoints/sft \\
    --output_dir checkpoints/reward_model \\
    --num_samples 10000 \\
    --epochs 2

# Smoke test
python scripts/train_reward_model.py --num_samples 200 --epochs 1 --no_fp16
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.training.reward import RewardConfig, train_reward_model


def parse_args() -> RewardConfig:
    p = argparse.ArgumentParser(description="Stage 2: Reward Model Training")
    p.add_argument("--sft_checkpoint", default="checkpoints/sft")
    p.add_argument("--output_dir", default="checkpoints/reward_model")
    p.add_argument("--num_samples", type=int, default=10_000)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--no_fp16", action="store_true")
    args = p.parse_args()

    return RewardConfig(
        sft_checkpoint=args.sft_checkpoint,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
        num_train_samples=None if args.num_samples == -1 else args.num_samples,
        fp16=not args.no_fp16,
    )


if __name__ == "__main__":
    cfg = parse_args()
    print("=" * 60)
    print("Stage 2: Reward Model Training")
    print(f"  sft_checkpoint : {cfg.sft_checkpoint}")
    print(f"  output_dir     : {cfg.output_dir}")
    print(f"  train_samples  : {cfg.num_train_samples or 'all'}")
    print(f"  epochs         : {cfg.num_epochs}")
    print(f"  lr             : {cfg.learning_rate}")
    print("=" * 60)
    train_reward_model(cfg)
