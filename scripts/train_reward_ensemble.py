#!/usr/bin/env python3
"""
Train an ensemble of K reward models for uncertainty-penalised PPO.

Usage
-----
# Train 3 models (default)
python scripts/train_reward_ensemble.py \\
    --sft_checkpoint checkpoints/sft \\
    --output_dir checkpoints/reward_ensemble \\
    --k 3

# Then run PPO with the ensemble:
python scripts/train_ppo_ensemble.py \\
    --sft_checkpoint checkpoints/sft \\
    --ensemble_dir checkpoints/reward_ensemble \\
    --uncertainty_penalty 0.5
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.training.reward_ensemble import EnsembleTrainingConfig, train_reward_ensemble


def main():
    p = argparse.ArgumentParser(description="Train reward model ensemble for uncertainty-penalised PPO")
    p.add_argument("--sft_checkpoint", default="checkpoints/sft")
    p.add_argument("--output_dir", default="checkpoints/reward_ensemble")
    p.add_argument("--k", type=int, default=3,
                   help="Number of ensemble members (3 is a good balance of cost vs uncertainty)")
    p.add_argument("--num_samples", type=int, default=10_000)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--base_seed", type=int, default=0,
                   help="Model i uses seed base_seed + i")
    p.add_argument("--no_fp16", action="store_true")
    args = p.parse_args()

    cfg = EnsembleTrainingConfig(
        sft_checkpoint=args.sft_checkpoint,
        output_dir=args.output_dir,
        k=args.k,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_samples=None if args.num_samples == -1 else args.num_samples,
        base_seed=args.base_seed,
        fp16=not args.no_fp16,
    )

    print("=" * 60)
    print(f"Reward Ensemble Training  (K={cfg.k})")
    print(f"  SFT checkpoint : {cfg.sft_checkpoint}")
    print(f"  Output dir     : {cfg.output_dir}")
    print(f"  Seeds          : {cfg.base_seed} … {cfg.base_seed + cfg.k - 1}")
    print(f"  Train samples  : {cfg.num_train_samples or 'all'}")
    print("=" * 60)

    checkpoints = train_reward_ensemble(cfg)
    print(f"\nAll {cfg.k} models trained:")
    for ckpt in checkpoints:
        print(f"  {ckpt}")


if __name__ == "__main__":
    main()
