#!/usr/bin/env python3
"""Train a GRPO policy from the SFT checkpoint and reward model."""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.training.grpo import GRPOTrainingConfig, train_grpo


def parse_args() -> GRPOTrainingConfig:
    p = argparse.ArgumentParser(description="Stage 3c: GRPO Policy Optimization")
    p.add_argument("--sft_checkpoint", default="checkpoints/sft")
    p.add_argument("--reward_checkpoint", default="checkpoints/reward_model")
    p.add_argument("--output_dir", default="checkpoints/grpo")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--num_samples", type=int, default=5_000)
    p.add_argument("--max_completion_length", type=int, default=128)
    p.add_argument("--num_generations", type=int, default=4)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--epsilon", type=float, default=0.2)
    args = p.parse_args()

    return GRPOTrainingConfig(
        sft_checkpoint=args.sft_checkpoint,
        reward_checkpoint=args.reward_checkpoint,
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_samples=args.num_samples,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        beta=args.beta,
        epsilon=args.epsilon,
    )


if __name__ == "__main__":
    cfg = parse_args()
    print("=" * 60)
    print("Stage 3c: GRPO Policy Optimization")
    print(f"  sft_checkpoint    : {cfg.sft_checkpoint}")
    print(f"  reward_checkpoint : {cfg.reward_checkpoint}")
    print(f"  output_dir        : {cfg.output_dir}")
    print(f"  num_generations   : {cfg.num_generations}")
    print(f"  beta              : {cfg.beta}")
    print(f"  epsilon           : {cfg.epsilon}")
    print("=" * 60)
    train_grpo(cfg)
