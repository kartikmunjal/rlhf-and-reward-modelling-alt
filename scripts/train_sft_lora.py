"""
CLI script: LoRA SFT training with ablation comparison.

Runs LoRA SFT at rank 8 and rank 16, then compares both against full SFT
trainable-parameter counts and eval loss.

Usage
-----
    # Train LoRA r=16 (default)
    python scripts/train_sft_lora.py

    # Train both r=8 and r=16 with comparison table
    python scripts/train_sft_lora.py --ranks 8 16 --compare_full

    # Quick smoke test
    python scripts/train_sft_lora.py --num_samples 500 --epochs 1 --ranks 16
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def parse_args():
    p = argparse.ArgumentParser(description="LoRA SFT training — parameter-efficient fine-tuning ablation")
    p.add_argument("--model", default="gpt2-medium", help="Base model name (gpt2 or gpt2-medium)")
    p.add_argument("--ranks", type=int, nargs="+", default=[16],
                   help="LoRA ranks to train. E.g. --ranks 8 16")
    p.add_argument("--num_samples", type=int, default=10_000)
    p.add_argument("--num_eval_samples", type=int, default=1_000)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--output_prefix", default="checkpoints/sft_lora",
                   help="Checkpoints saved as {prefix}_r{rank}")
    p.add_argument("--compare_full", action="store_true",
                   help="Print side-by-side table vs full SFT parameter count")
    p.add_argument("--merge_after", action="store_true",
                   help="Merge LoRA adapters into base model and save full checkpoint")
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--report_to", default="none", choices=["none", "wandb"])
    return p.parse_args()


def main():
    args = parse_args()

    from src.training.sft_lora import LoRASFTConfig, train_sft_lora, merge_and_save

    all_results = {}

    for rank in args.ranks:
        print(f"\n{'='*60}")
        print(f"  Training LoRA SFT  rank r={rank}  on {args.model}")
        print(f"{'='*60}")

        cfg = LoRASFTConfig(
            model_name=args.model,
            output_dir=f"{args.output_prefix}_r{rank}",
            lora_r=rank,
            lora_alpha=rank * 2,          # standard: alpha = 2 * r
            num_train_epochs=args.epochs,
            num_train_samples=args.num_samples,
            num_eval_samples=args.num_eval_samples,
            learning_rate=args.lr,
            max_length=args.max_length,
            fp16=args.fp16,
            report_to=args.report_to,
        )
        param_stats = train_sft_lora(cfg)
        all_results[f"lora_r{rank}"] = param_stats

        if args.merge_after:
            merge_and_save(
                adapter_dir=cfg.output_dir,
                output_dir=f"{cfg.output_dir}_merged",
                model_name=args.model,
            )

    # ── Comparison table ──────────────────────────────────────────────────────
    if args.compare_full:
        # Approximate full SFT trainable params (all params)
        from transformers import GPT2LMHeadModel
        full_model = GPT2LMHeadModel.from_pretrained(args.model)
        full_params = sum(p.numel() for p in full_model.parameters())
        del full_model

        print(f"\n{'='*60}")
        print("  ABLATION: Trainable parameters vs method")
        print(f"{'='*60}")
        print(f"{'Method':<20} {'Trainable':>15} {'Total':>15} {'%':>8}")
        print("-" * 62)
        print(f"{'Full SFT':<20} {full_params:>15,} {full_params:>15,} {'100.00':>8}%")
        for label, stats in all_results.items():
            print(
                f"{label:<20} "
                f"{stats['trainable']:>15,} "
                f"{stats['total']:>15,} "
                f"{stats['fraction_pct']:>8.2f}%"
            )
        print(
            f"\nKey result: LoRA r=16 uses "
            f"{all_results.get('lora_r16', list(all_results.values())[-1])['fraction_pct']:.2f}% "
            f"of full SFT parameters."
        )

    # Save stats JSON for notebook
    os.makedirs("results", exist_ok=True)
    with open("results/lora_sft_param_stats.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nParameter stats saved to results/lora_sft_param_stats.json")


if __name__ == "__main__":
    main()
