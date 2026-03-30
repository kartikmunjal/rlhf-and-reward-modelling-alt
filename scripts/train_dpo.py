#!/usr/bin/env python3
"""
Train a DPO policy — Stage 3b of the RLHF pipeline.

DPO only requires Stage 1 (SFT) — no reward model is needed at training time.

Usage
-----
python scripts/train_dpo.py \\
    --sft_checkpoint checkpoints/sft \\
    --output_dir checkpoints/dpo \\
    --beta 0.1

# Sweep over beta values to understand sensitivity
for beta in 0.05 0.1 0.2 0.5; do
    python scripts/train_dpo.py --beta $beta --output_dir checkpoints/dpo_beta$beta
done

# Smoke test
python scripts/train_dpo.py --num_samples 200 --epochs 1 --no_fp16
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.training.dpo import DPOTrainingConfig, train_dpo


def parse_args() -> DPOTrainingConfig:
    p = argparse.ArgumentParser(description="Stage 3b: DPO Policy Optimization")
    p.add_argument("--sft_checkpoint", default="checkpoints/sft")
    p.add_argument("--output_dir", default="checkpoints/dpo")
    p.add_argument("--beta", type=float, default=0.1,
                   help="KL regularisation strength. Higher β → less deviation from SFT.")
    p.add_argument("--num_samples", type=int, default=10_000)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-7)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--no_fp16", action="store_true")
    p.add_argument("--report_to", default="none")
    args = p.parse_args()

    return DPOTrainingConfig(
        sft_checkpoint=args.sft_checkpoint,
        output_dir=args.output_dir,
        beta=args.beta,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_length=args.max_length,
        num_train_samples=None if args.num_samples == -1 else args.num_samples,
        fp16=not args.no_fp16,
        report_to=args.report_to,
    )


if __name__ == "__main__":
    cfg = parse_args()
    print("=" * 60)
    print("Stage 3b: DPO Policy Optimization")
    print(f"  sft_checkpoint : {cfg.sft_checkpoint}")
    print(f"  output_dir     : {cfg.output_dir}")
    print(f"  beta (β)       : {cfg.beta}")
    print(f"  train_samples  : {cfg.num_train_samples or 'all'}")
    print(f"  epochs         : {cfg.num_train_epochs}")
    print(f"  lr             : {cfg.learning_rate}")
    print("=" * 60)
    train_dpo(cfg)
