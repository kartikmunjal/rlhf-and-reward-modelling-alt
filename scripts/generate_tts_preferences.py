"""
CLI: Generate TTS preference pairs for reward model training.

Usage
-----
    # Full generation (30 prompts × 5 description pairs = 150 pairs)
    python scripts/generate_tts_preferences.py

    # Smoke test (5 prompts, acoustic scorer, no GPU)
    python scripts/generate_tts_preferences.py --num_prompts 5

    # Use UTMOS scorer (better quality, requires GPU + downloading model)
    python scripts/generate_tts_preferences.py --use_utmos

    # CPU only (slower but no GPU required)
    python scripts/generate_tts_preferences.py --device cpu

    # Dry run: print prompts and description pairs without generating audio
    python scripts/generate_tts_preferences.py --dry_run
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def parse_args():
    p = argparse.ArgumentParser(description="Generate TTS preference pairs")
    p.add_argument("--output", default="data/tts_preferences.jsonl")
    p.add_argument("--model_id", default="parler-tts/parler-tts-mini-v1")
    p.add_argument("--num_prompts", type=int, default=30,
                   help="How many text prompts to use (default: 30, max: 30)")
    p.add_argument("--device", default="cpu",
                   help="Device: 'cpu' or 'cuda' (default: cpu)")
    p.add_argument("--use_utmos", action="store_true",
                   help="Score with UTMOS22 instead of acoustic heuristics (requires GPU)")
    p.add_argument("--dry_run", action="store_true",
                   help="Show plan without calling TTS model")
    return p.parse_args()


def main():
    args = parse_args()

    from src.data.tts_preferences import (
        TTS_PROMPT_CATALOGUE,
        TTS_DESCRIPTION_PAIRS,
        TTS_DESCRIPTIONS,
        TTSPreferenceConfig,
        generate_tts_preference_dataset,
    )

    if args.dry_run:
        prompts = TTS_PROMPT_CATALOGUE[:args.num_prompts]
        total = len(prompts) * len(TTS_DESCRIPTION_PAIRS)
        print(f"TTS preference generation plan:")
        print(f"  model:    {args.model_id}")
        print(f"  device:   {args.device}")
        print(f"  scorer:   {'UTMOS22' if args.use_utmos else 'acoustic heuristics'}")
        print(f"  prompts:  {len(prompts)}")
        print(f"  pairs:    {len(TTS_DESCRIPTION_PAIRS)} description pairs")
        print(f"  total:    {total} preference pairs")
        print(f"  output:   {args.output}")

        print(f"\nDescription pairs (expected preference direction):")
        for a, b in TTS_DESCRIPTION_PAIRS:
            print(f"  {a} > {b}")

        print(f"\nSample prompts:")
        for item in prompts[:5]:
            print(f"  [{item['style']:15s}] {item['text'][:70]}...")

        print(f"\nSample description:")
        key = list(TTS_DESCRIPTIONS.keys())[0]
        print(f"  {key}: {TTS_DESCRIPTIONS[key][:100]}...")
        return

    cfg = TTSPreferenceConfig(
        output_path=args.output,
        model_id=args.model_id,
        prompts_per_pair=args.num_prompts,
        use_utmos=args.use_utmos,
        device=args.device,
    )

    print(f"Generating TTS preference pairs:")
    print(f"  model={args.model_id}  device={args.device}")
    print(f"  scorer={'UTMOS22' if args.use_utmos else 'acoustic heuristics'}")
    print(f"  output → {args.output}\n")

    generate_tts_preference_dataset(cfg)

    # Summary
    with open(args.output) as f:
        records = [json.loads(l) for l in f]

    print(f"\nDataset summary:")
    print(f"  total pairs: {len(records)}")
    if records:
        deltas = [r["score_delta"] for r in records]
        print(f"  avg delta:   {sum(deltas)/len(deltas):.4f}")
        print(f"  max delta:   {max(deltas):.4f}")
        print(f"  min delta:   {min(deltas):.4f}")
        # Breakdown by description pair
        from collections import Counter
        pair_counts = Counter(
            f"{r['chosen_description']} > {r['rejected_description']}"
            for r in records
        )
        print(f"\nPreference directions:")
        for pair, count in sorted(pair_counts.items()):
            print(f"  {pair}: {count}")


if __name__ == "__main__":
    main()
