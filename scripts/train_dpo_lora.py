"""
CLI script: LoRA DPO training with comparison against full DPO.

Usage
-----
    # Train LoRA DPO (default r=16)
    python scripts/train_dpo_lora.py

    # Compare against full DPO using reward model as judge
    python scripts/train_dpo_lora.py --compare_full \
        --full_dpo_checkpoint checkpoints/dpo \
        --reward_checkpoint checkpoints/reward

    # Quick smoke test
    python scripts/train_dpo_lora.py --num_samples 500 --epochs 1 --rank 16
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def parse_args():
    p = argparse.ArgumentParser(description="LoRA DPO training — parameter-efficient preference optimization")
    p.add_argument("--sft_checkpoint", default="checkpoints/sft",
                   help="Path to SFT checkpoint (used as reference policy + LoRA init)")
    p.add_argument("--rank", type=int, default=16, help="LoRA rank (8 or 16)")
    p.add_argument("--beta", type=float, default=0.1, help="DPO beta (KL coefficient)")
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate for adapter params")
    p.add_argument("--num_samples", type=int, default=10_000)
    p.add_argument("--num_eval_samples", type=int, default=1_000)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--output_dir", default=None,
                   help="Output directory (default: checkpoints/dpo_lora_r{rank})")
    p.add_argument("--merge_after", action="store_true", default=True,
                   help="Merge LoRA adapters into base model after training")
    p.add_argument("--compare_full", action="store_true",
                   help="Compare LoRA DPO vs full DPO using RM win rate")
    p.add_argument("--full_dpo_checkpoint", default="checkpoints/dpo",
                   help="Full DPO checkpoint path for comparison")
    p.add_argument("--reward_checkpoint", default="checkpoints/reward",
                   help="Reward model checkpoint for evaluation")
    p.add_argument("--num_eval", type=int, default=500)
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--report_to", default="none", choices=["none", "wandb"])
    return p.parse_args()


def main():
    args = parse_args()

    from src.training.dpo_lora import LoRADPOConfig, train_dpo_lora

    output_dir = args.output_dir or f"checkpoints/dpo_lora_r{args.rank}"

    cfg = LoRADPOConfig(
        sft_checkpoint=args.sft_checkpoint,
        output_dir=output_dir,
        lora_r=args.rank,
        lora_alpha=args.rank * 2,
        beta=args.beta,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        num_train_samples=args.num_samples,
        num_eval_samples=args.num_eval_samples,
        fp16=args.fp16,
        report_to=args.report_to,
        merge_after_training=args.merge_after,
        merged_output_dir=f"{output_dir}_merged",
    )

    print(f"\nTraining LoRA DPO  r={args.rank}  β={args.beta}")
    param_stats = train_dpo_lora(cfg)

    results = {"lora_dpo": {"param_stats": param_stats}}

    if args.compare_full:
        print(f"\nComparing LoRA DPO (r={args.rank}) vs full DPO using RM win rate...")
        from src.training.dpo_lora import compare_lora_vs_full_dpo

        lora_ckpt = cfg.merged_output_dir if args.merge_after else output_dir
        comparison = compare_lora_vs_full_dpo(
            lora_dpo_dir=lora_ckpt,
            full_dpo_dir=args.full_dpo_checkpoint,
            reward_model_dir=args.reward_checkpoint,
            num_eval=args.num_eval,
        )
        results["comparison"] = comparison

        print(f"\n{'='*55}")
        print("  ABLATION: LoRA DPO vs Full DPO")
        print(f"{'='*55}")
        print(f"{'Method':<20} {'Win Rate':>12} {'Mean Reward':>14}")
        print("-" * 50)
        for method, metrics in comparison.items():
            print(
                f"{method:<20} "
                f"{metrics['win_rate']:>12.4f} "
                f"{metrics['mean_reward']:>14.4f}"
            )

    # Save stats
    os.makedirs("results", exist_ok=True)
    with open("results/lora_dpo_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to results/lora_dpo_results.json")


if __name__ == "__main__":
    main()
