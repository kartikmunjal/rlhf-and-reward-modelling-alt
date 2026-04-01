"""
CLI script: Iterative DPO training with buffer strategy ablation.

Runs iterative DPO for multiple iterations and compares three buffer strategies
(current / rolling2 / full) to show how data staleness affects alignment quality.

Usage
-----
    # Single run: rolling2 buffer, 3 iterations (default)
    python scripts/train_iterative_dpo.py

    # Compare all three buffer strategies
    python scripts/train_iterative_dpo.py --compare_buffers

    # Quick smoke test (100 rollout prompts, 1 iteration, 50 DPO steps)
    python scripts/train_iterative_dpo.py \
        --num_iterations 1 --rollout_batch_size 100 --dpo_steps 50

    # Full ablation vs PPO and single DPO baselines
    python scripts/train_iterative_dpo.py \
        --compare_buffers \
        --ppo_win_rate 0.712 --single_dpo_win_rate 0.634
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def parse_args():
    p = argparse.ArgumentParser(description="Iterative DPO: self-improving alignment loop")
    p.add_argument("--sft_checkpoint", default="checkpoints/sft")
    p.add_argument("--reward_checkpoint", default="checkpoints/reward")
    p.add_argument("--output_dir", default="checkpoints/iterative_dpo")
    p.add_argument("--num_iterations", type=int, default=3)
    p.add_argument("--rollout_batch_size", type=int, default=256,
                   help="Prompts to roll out per iteration")
    p.add_argument("--dpo_steps", type=int, default=200,
                   help="DPO gradient steps per iteration")
    p.add_argument("--eval_prompts", type=int, default=200)
    p.add_argument("--buffer", default="rolling2",
                   choices=["current", "rolling2", "full"],
                   help="Buffer strategy for preference pair management")
    p.add_argument("--compare_buffers", action="store_true",
                   help="Run all three buffer strategies and produce comparison table")
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=5e-7)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--report_to", default="none", choices=["none", "wandb"])
    # Baseline win rates for comparison table
    p.add_argument("--ppo_win_rate", type=float, default=0.712)
    p.add_argument("--ppo_kl", type=float, default=4.821)
    p.add_argument("--single_dpo_win_rate", type=float, default=0.634)
    p.add_argument("--single_dpo_kl", type=float, default=1.734)
    return p.parse_args()


def run_one(args, buffer_strategy: str) -> list:
    from src.training.iterative_dpo import IterativeDPOConfig, run_iterative_dpo

    cfg = IterativeDPOConfig(
        sft_checkpoint=args.sft_checkpoint,
        reward_checkpoint=args.reward_checkpoint,
        output_dir=os.path.join(args.output_dir, buffer_strategy),
        num_iterations=args.num_iterations,
        rollout_batch_size=args.rollout_batch_size,
        dpo_steps_per_iter=args.dpo_steps,
        eval_prompts=args.eval_prompts,
        buffer_strategy=buffer_strategy,
        beta=args.beta,
        learning_rate=args.lr,
        max_new_tokens=args.max_new_tokens,
        fp16=args.fp16,
        report_to=args.report_to,
    )
    return run_iterative_dpo(cfg)


def main():
    args = parse_args()

    from src.training.iterative_dpo import compare_with_baselines

    strategies = ["current", "rolling2", "full"] if args.compare_buffers else [args.buffer]
    all_results = {}

    for strategy in strategies:
        print(f"\n{'#'*60}")
        print(f"  Buffer strategy: {strategy}")
        print(f"{'#'*60}")
        results = run_one(args, strategy)
        all_results[strategy] = results

    # Print comparison table for each strategy
    if args.compare_buffers:
        print(f"\n{'='*70}")
        print("  BUFFER STRATEGY COMPARISON: Win Rate per Iteration")
        print(f"{'='*70}")
        print(f"{'Strategy':<12} " + "  ".join(f"Iter{i+1} WR" for i in range(args.num_iterations)))
        print("-" * 70)
        for strategy, results in all_results.items():
            wrs = "  ".join(f"{r.win_rate:>8.4f}" for r in results)
            print(f"{strategy:<12} {wrs}")

        # Baseline comparison for the rolling2 run
        ref_results = all_results.get("rolling2", list(all_results.values())[0])
        compare_with_baselines(
            ref_results,
            ppo_win_rate=args.ppo_win_rate,
            ppo_kl=args.ppo_kl,
            single_dpo_win_rate=args.single_dpo_win_rate,
            single_dpo_kl=args.single_dpo_kl,
        )
    else:
        results = all_results[args.buffer]
        compare_with_baselines(
            results,
            ppo_win_rate=args.ppo_win_rate,
            ppo_kl=args.ppo_kl,
            single_dpo_win_rate=args.single_dpo_win_rate,
            single_dpo_kl=args.single_dpo_kl,
        )

    # Save combined results
    os.makedirs("results", exist_ok=True)
    combined = {k: [r.__dict__ for r in v] for k, v in all_results.items()}
    with open("results/iterative_dpo_results.json", "w") as f:
        json.dump(combined, f, indent=2)
    print("\nResults saved to results/iterative_dpo_results.json")


if __name__ == "__main__":
    main()
