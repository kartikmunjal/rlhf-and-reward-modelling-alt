#!/usr/bin/env python3
"""
Train PRM and ORM on GSM8K for the PRM vs ORM ablation.

Usage
-----
# Train both (default)
python scripts/train_prm.py

# Train only PRM
python scripts/train_prm.py --only prm

# Then compare:
python scripts/compare_prm_orm.py \\
    --prm_checkpoint checkpoints/prm \\
    --orm_checkpoint checkpoints/orm \\
    --num_eval 500
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.training.prm import PRMConfig, ORMConfig, train_prm, train_orm


def main():
    p = argparse.ArgumentParser(description="Train PRM and/or ORM on GSM8K")
    p.add_argument("--sft_checkpoint", default="checkpoints/sft")
    p.add_argument("--prm_output_dir", default="checkpoints/prm")
    p.add_argument("--orm_output_dir", default="checkpoints/orm")
    p.add_argument("--aggregation", default="mean", choices=["mean", "min", "sum"],
                   help="PRM step-score aggregation method")
    p.add_argument("--num_samples", type=int, default=5_000)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--only", choices=["prm", "orm", "both"], default="both")
    p.add_argument("--no_fp16", action="store_true")
    args = p.parse_args()

    if args.only in ("prm", "both"):
        prm_cfg = PRMConfig(
            sft_checkpoint=args.sft_checkpoint,
            output_dir=args.prm_output_dir,
            aggregation_mode=args.aggregation,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            num_train_samples=None if args.num_samples == -1 else args.num_samples,
            fp16=not args.no_fp16,
        )
        print("=" * 60)
        print(f"Training PRM (aggregation={prm_cfg.aggregation_mode})")
        print(f"  output: {prm_cfg.output_dir}")
        print("=" * 60)
        train_prm(prm_cfg)

    if args.only in ("orm", "both"):
        orm_cfg = ORMConfig(
            sft_checkpoint=args.sft_checkpoint,
            output_dir=args.orm_output_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            num_train_samples=None if args.num_samples == -1 else args.num_samples,
            fp16=not args.no_fp16,
        )
        print("=" * 60)
        print("Training ORM")
        print(f"  output: {orm_cfg.output_dir}")
        print("=" * 60)
        train_orm(orm_cfg)


if __name__ == "__main__":
    main()
