"""
CLI script: Generate synthetic SFT (prompt, response) pairs via Claude.

Requires ANTHROPIC_API_KEY environment variable.

Usage
-----
    # Generate 2000 pairs with critic pass (default)
    export ANTHROPIC_API_KEY=sk-ant-...
    python scripts/generate_synthetic_sft.py --num_samples 2000

    # Quick test: 10 pairs, no critic, fast model
    python scripts/generate_synthetic_sft.py --num_samples 10 --no_critic

    # Larger run with more expensive model and comparison to hh-rlhf
    python scripts/generate_synthetic_sft.py --num_samples 10000 \
        --model claude-haiku-4-5-20251001 --compare_hh_rlhf
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate synthetic SFT data from Claude, grounded in a constitution"
    )
    p.add_argument("--num_samples", type=int, default=2_000,
                   help="Number of (prompt, response) pairs to generate")
    p.add_argument("--output", default="data/synthetic_sft.jsonl",
                   help="Output JSONL file path")
    p.add_argument("--model", default="claude-haiku-4-5-20251001",
                   help="Claude model to use for generation")
    p.add_argument("--max_tokens", type=int, default=400,
                   help="Max response tokens per pair")
    p.add_argument("--no_critic", action="store_true",
                   help="Skip critic (revision) pass — faster but lower quality")
    p.add_argument("--no_extra_prompts", action="store_true",
                   help="Only use seed prompts, do not generate new ones")
    p.add_argument("--rpm", type=int, default=40,
                   help="Maximum API requests per minute")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--compare_hh_rlhf", action="store_true",
                   help="After generation, print a constitution coverage summary")
    return p.parse_args()


def main():
    args = parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set.")
        sys.exit(1)

    from src.data.synthetic_sft import SyntheticSFTConfig, generate_synthetic_sft_dataset

    cfg = SyntheticSFTConfig(
        output_path=args.output,
        num_samples=args.num_samples,
        model=args.model,
        max_tokens=args.max_tokens,
        apply_critic=not args.no_critic,
        requests_per_minute=args.rpm,
        seed=args.seed,
        generate_extra_prompts=not args.no_extra_prompts,
    )

    print(f"Generating {args.num_samples} synthetic SFT pairs")
    print(f"  Model       : {args.model}")
    print(f"  Critic pass : {not args.no_critic}")
    print(f"  Output      : {args.output}")
    print()

    generate_synthetic_sft_dataset(cfg)

    if args.compare_hh_rlhf:
        _print_comparison_summary(args.output)


def _print_comparison_summary(jsonl_path: str) -> None:
    """Print token-length and response-quality summary vs hh-rlhf."""
    import json
    from datasets import load_dataset

    print("\n" + "="*60)
    print("  Synthetic SFT vs hh-rlhf: response length comparison")
    print("="*60)

    # Load synthetic data
    synthetic = []
    with open(jsonl_path) as f:
        for line in f:
            item = json.loads(line)
            synthetic.append(len(item.get("response", "").split()))
    if not synthetic:
        return
    syn_mean = sum(synthetic) / len(synthetic)
    syn_max = max(synthetic)

    # Load a sample from hh-rlhf
    try:
        raw = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base", split="train")
        raw = raw.select(range(min(len(synthetic), len(raw))))

        def chosen_len(ex):
            text = ex["chosen"]
            # Extract just the last assistant response
            parts = text.split("\n\nAssistant:")
            resp = parts[-1].strip() if len(parts) > 1 else text
            return len(resp.split())

        hh_lens = [chosen_len(ex) for ex in raw]
        hh_mean = sum(hh_lens) / len(hh_lens)
        hh_max = max(hh_lens)

        print(f"{'Source':<20} {'Mean words':>12} {'Max words':>12} {'N':>8}")
        print("-" * 56)
        print(f"{'Synthetic (Claude)':<20} {syn_mean:>12.1f} {syn_max:>12} {len(synthetic):>8}")
        print(f"{'hh-rlhf chosen':<20} {hh_mean:>12.1f} {hh_max:>12} {len(hh_lens):>8}")
    except Exception as e:
        print(f"Could not load hh-rlhf for comparison: {e}")
        print(f"Synthetic: mean={syn_mean:.1f} words, max={syn_max}, n={len(synthetic)}")


if __name__ == "__main__":
    main()
