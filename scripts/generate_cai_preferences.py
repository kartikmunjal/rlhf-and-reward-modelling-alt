#!/usr/bin/env python3
"""
Generate AI-labeled preference pairs using Constitutional AI (Stage 2 — RLAIF variant).

This script uses Claude to annotate which of two SFT-generated responses better
follows a set of constitutional principles.  The output is a JSONL file that can
be used as a drop-in replacement for the human-labeled hh-rlhf preference data.

Usage
-----
# Set your API key first:
export ANTHROPIC_API_KEY="sk-ant-..."

# Generate 2000 pairs (default)
python scripts/generate_cai_preferences.py \\
    --sft_checkpoint checkpoints/sft \\
    --output data/cai_preferences.jsonl \\
    --num_pairs 2000

# Quick test (50 pairs)
python scripts/generate_cai_preferences.py \\
    --sft_checkpoint checkpoints/sft \\
    --output data/cai_preferences_test.jsonl \\
    --num_pairs 50

# Then train a reward model on the AI-generated labels:
python scripts/train_reward_model.py \\
    --cai_data data/cai_preferences.jsonl \\
    --output_dir checkpoints/reward_model_cai

# Compare pairwise accuracy: human labels vs AI labels
python scripts/compare_cai_vs_human.py
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.cai import CAIConfig, generate_cai_preferences, CONSTITUTION


def main():
    p = argparse.ArgumentParser(
        description="Generate CAI/RLAIF preference labels using Claude as annotator"
    )
    p.add_argument("--sft_checkpoint", default="checkpoints/sft")
    p.add_argument("--output", default="data/cai_preferences.jsonl")
    p.add_argument("--num_pairs", type=int, default=2_000,
                   help="Number of labeled preference pairs to generate")
    p.add_argument("--annotator_model", default="claude-haiku-4-5-20251001",
                   help="Claude model to use for annotation (haiku is cost-efficient)")
    p.add_argument("--temperature", type=float, default=0.8,
                   help="SFT generation temperature (higher = more diverse pairs)")
    p.add_argument("--min_confidence", default="medium",
                   choices=["high", "medium", "low"],
                   help="Filter out pairs where Claude is below this confidence")
    p.add_argument("--rpm", type=int, default=40,
                   help="API requests per minute (stay within rate limits)")
    args = p.parse_args()

    cfg = CAIConfig(
        sft_checkpoint=args.sft_checkpoint,
        output_path=args.output,
        num_pairs=args.num_pairs,
        annotator_model=args.annotator_model,
        generation_temperature=args.temperature,
        min_confidence=args.min_confidence,
        requests_per_minute=args.rpm,
    )

    print("=" * 60)
    print("Constitutional AI / RLAIF — Preference Generation")
    print(f"  SFT checkpoint  : {cfg.sft_checkpoint}")
    print(f"  Output          : {cfg.output_path}")
    print(f"  Target pairs    : {cfg.num_pairs:,}")
    print(f"  Annotator model : {cfg.annotator_model}")
    print(f"  Min confidence  : {cfg.min_confidence}")
    print()
    print("Constitution principles:")
    for i, p_text in enumerate(CONSTITUTION, 1):
        print(f"  {i}. {p_text[:80]}...")
    print("=" * 60)
    print()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable not set.")
        print("       export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    generate_cai_preferences(cfg)


if __name__ == "__main__":
    main()
