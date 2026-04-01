"""
CLI script: Run scaling comparison — GPT-2-small (117M) vs GPT-2-medium (355M).

Trains the full SFT → Reward → DPO pipeline at two model sizes and produces:
  - results/scaling_results.csv  (raw metrics per model × stage)
  - results/scaling_summary.md   (formatted markdown table)

Usage
-----
    # Full run (SFT + RM + DPO for both model sizes)
    python scripts/run_scaling_comparison.py

    # Quick smoke test (5k samples, 1 epoch each stage)
    python scripts/run_scaling_comparison.py \
        --num_samples 1000 --sft_epochs 1 --reward_epochs 1 --dpo_epochs 1

    # Only run gpt2-medium (skip small model)
    python scripts/run_scaling_comparison.py --models gpt2-medium
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def parse_args():
    p = argparse.ArgumentParser(
        description="Scaling comparison: GPT-2-small vs GPT-2-medium on RLHF pipeline"
    )
    p.add_argument("--models", nargs="+", default=["gpt2", "gpt2-medium"],
                   help="Model names to compare")
    p.add_argument("--num_samples", type=int, default=5_000,
                   help="Training samples per stage per model")
    p.add_argument("--num_eval_samples", type=int, default=500)
    p.add_argument("--sft_epochs", type=int, default=2)
    p.add_argument("--reward_epochs", type=int, default=2)
    p.add_argument("--dpo_epochs", type=int, default=1)
    p.add_argument("--output_prefix", default="checkpoints/scaling",
                   help="Root directory for all scaling checkpoints")
    p.add_argument("--results_csv", default="results/scaling_results.csv")
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--report_to", default="none", choices=["none", "wandb"])
    return p.parse_args()


def main():
    args = parse_args()

    from src.training.scaling import ScalingConfig, run_scaling_comparison, format_scaling_table

    cfg = ScalingConfig(
        model_sizes=args.models,
        num_train_samples=args.num_samples,
        num_eval_samples=args.num_eval_samples,
        num_eval=args.num_eval_samples,
        sft_epochs=args.sft_epochs,
        reward_epochs=args.reward_epochs,
        dpo_epochs=args.dpo_epochs,
        fp16=args.fp16,
        output_dir_prefix=args.output_prefix,
        results_csv=args.results_csv,
        report_to=args.report_to,
    )

    print("Starting scaling comparison:")
    for m in args.models:
        from transformers import GPT2Config
        n = cfg.model_params.get(m, 0)
        print(f"  {m:20s}  ~{n/1e6:.0f}M parameters")
    print()

    df = run_scaling_comparison(cfg)

    # Print summary table
    print("\n" + "="*70)
    print("  SCALING RESULTS SUMMARY")
    print("="*70)
    print(format_scaling_table(df))

    # Save markdown summary
    os.makedirs("results", exist_ok=True)
    with open("results/scaling_summary.md", "w") as f:
        f.write("# Scaling Analysis: GPT-2-small (117M) vs GPT-2-medium (355M)\n\n")
        f.write("## Pipeline: SFT → Reward Model → DPO\n\n")
        f.write(format_scaling_table(df))
        f.write("\n\n## Notes\n")
        f.write("- RM accuracy: pairwise preference accuracy on held-out preference pairs\n")
        f.write("- DPO preference accuracy: RM-judged win rate of DPO vs SFT reference\n")
        f.write("- Training time: wall-clock seconds (hardware-dependent)\n")
    print("\nScaling summary saved to results/scaling_summary.md")


if __name__ == "__main__":
    main()
