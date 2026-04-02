"""
CLI: Generate agentic SFT trajectory dataset.

Usage
-----
    export ANTHROPIC_API_KEY=sk-ant-...

    # Generate all 17 catalogue tasks × 3 temperatures → 51 trajectories
    python scripts/generate_agentic_sft.py

    # Subset: only tool_use and multi_step categories
    python scripts/generate_agentic_sft.py --categories tool_use multi_step

    # Quick smoke test (1 generation per task)
    python scripts/generate_agentic_sft.py --generations_per_task 1

    # Use a different model / output path
    python scripts/generate_agentic_sft.py \
        --model claude-haiku-4-5-20251001 \
        --output data/agentic_sft_custom.jsonl
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def parse_args():
    p = argparse.ArgumentParser(description="Generate agentic SFT trajectory dataset")
    p.add_argument(
        "--output",
        default="data/agentic_sft.jsonl",
        help="Where to write the JSONL (default: data/agentic_sft.jsonl)",
    )
    p.add_argument(
        "--model",
        default="claude-haiku-4-5-20251001",
        help="Claude model for trajectory generation",
    )
    p.add_argument(
        "--max_tokens",
        type=int,
        default=800,
        help="Max tokens per trajectory (default: 800)",
    )
    p.add_argument(
        "--generations_per_task",
        type=int,
        default=3,
        help="Trajectories per task (default: 3, diversity via temperature)",
    )
    p.add_argument(
        "--categories",
        nargs="+",
        choices=["tool_use", "multi_step", "failure_recovery"],
        default=None,
        help="Restrict to specific task categories (default: all)",
    )
    p.add_argument(
        "--requests_per_minute",
        type=int,
        default=40,
        help="Rate limit (default: 40 rpm)",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Print task catalogue and exit without calling the API",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set.")
        sys.exit(1)

    from src.data.agentic_sft import (
        AGENTIC_TASK_CATALOGUE,
        AgenticSFTConfig,
        generate_agentic_sft_dataset,
    )

    tasks = AGENTIC_TASK_CATALOGUE
    if args.categories:
        tasks = [t for t in tasks if t["category"] in args.categories]
        print(f"Filtered to {len(tasks)} tasks in: {args.categories}")

    if args.dry_run:
        print(f"\nDry run — {len(tasks)} tasks:")
        by_cat = {}
        for t in tasks:
            by_cat.setdefault(t["category"], []).append(t)
        for cat, cat_tasks in by_cat.items():
            print(f"\n  {cat} ({len(cat_tasks)} tasks):")
            for t in cat_tasks:
                sr = "search" if t.get("requires_search") else "memory"
                steps = t.get("n_steps", 1)
                print(f"    [{sr}, {steps}-step] {t['prompt'][:70]}...")
                print(f"      ground_truth: {t['ground_truth']!r}")
        total = len(tasks) * args.generations_per_task
        print(f"\nWould generate {total} trajectories ({len(tasks)} tasks × {args.generations_per_task} generations)")
        return

    cfg = AgenticSFTConfig(
        output_path=args.output,
        model=args.model,
        max_tokens=args.max_tokens,
        generations_per_task=args.generations_per_task,
        requests_per_minute=args.requests_per_minute,
        seed_tasks=tasks if args.categories else None,
    )

    print(f"Generating {len(tasks) * args.generations_per_task} trajectories")
    print(f"  model={args.model}  max_tokens={args.max_tokens}  generations_per_task={args.generations_per_task}")
    print(f"  output → {args.output}\n")

    generate_agentic_sft_dataset(cfg)

    # Quick sanity: count and show a sample
    with open(args.output) as f:
        lines = [json.loads(l) for l in f]

    print(f"\nSample trajectory (first item):")
    if lines:
        sample = lines[0]
        print(f"  prompt:     {sample['prompt'][:60]}...")
        print(f"  category:   {sample['category']}")
        print(f"  tool_calls: {sample['n_tool_calls']}")
        traj_preview = sample["trajectory"][:300].replace("\n", " ↵ ")
        print(f"  trajectory: {traj_preview}...")


if __name__ == "__main__":
    main()
