#!/usr/bin/env python3
"""
Train the SFT (supervised fine-tuning) model — Stage 1 of the RLHF pipeline.

Usage
-----
# Quick smoke test (CPU, no GPU needed)
python scripts/train_sft.py --num_samples 200 --epochs 1 --batch_size 2 --no_fp16

# Standard training on a single GPU
python scripts/train_sft.py --num_samples 10000 --epochs 3

# Full dataset
python scripts/train_sft.py --num_samples -1 --epochs 3 --output_dir checkpoints/sft_full
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.training.sft import SFTConfig, train_sft


def parse_args() -> SFTConfig:
    p = argparse.ArgumentParser(description="Stage 1: Supervised Fine-Tuning")
    p.add_argument("--model", default="gpt2-medium", help="HuggingFace model name")
    p.add_argument("--output_dir", default="checkpoints/sft")
    p.add_argument("--num_samples", type=int, default=10_000,
                   help="Training examples (-1 = full dataset ~42k)")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--no_fp16", action="store_true", help="Disable mixed precision")
    p.add_argument("--report_to", default="none", help="none | wandb | tensorboard")
    args = p.parse_args()

    return SFTConfig(
        model_name=args.model,
        output_dir=args.output_dir,
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
    print("Stage 1: Supervised Fine-Tuning")
    print(f"  model       : {cfg.model_name}")
    print(f"  output_dir  : {cfg.output_dir}")
    print(f"  train_samples: {cfg.num_train_samples or 'all'}")
    print(f"  epochs      : {cfg.num_train_epochs}")
    print(f"  eff_batch   : {cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps}")
    print(f"  fp16        : {cfg.fp16}")
    print("=" * 60)
    train_sft(cfg)
