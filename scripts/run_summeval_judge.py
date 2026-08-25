#!/usr/bin/env python3
"""Run a resumable SummEval judge phase."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_judge_summeval.runner import run_pairwise_heldout, run_pointwise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("dev", "heldout"), required=True)
    parser.add_argument("--provider", choices=("anthropic", "openai"), required=True)
    parser.add_argument("--kind", choices=("pointwise", "pairwise"), default="pointwise")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--config", type=Path, default=Path("llm_judge_summeval/study_config.json"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/summeval_judge_v1"))
    parser.add_argument("--prompts", type=Path, default=Path("llm_judge_summeval/prompts.json"))
    parser.add_argument("--prompt-manifest", type=Path, default=Path("llm_judge_summeval/final_prompt_manifest.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/summeval_judge_v1"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.kind == "pairwise":
        if args.phase != "heldout" or args.provider != "anthropic":
            parser.error("Pairwise is preregistered only for held-out Anthropic")
        result = run_pairwise_heldout(config=config, processed_dir=args.processed_dir, prompts_path=args.prompts,
                                      final_prompt_manifest=args.prompt_manifest,
                                      ledger_path=args.output_dir / "anthropic_pairwise.jsonl", limit=args.limit)
    else:
        result = run_pointwise(phase=args.phase, provider_name=args.provider, config=config,
                               processed_dir=args.processed_dir, prompts_path=args.prompts,
                               final_prompt_manifest=args.prompt_manifest,
                               ledger_path=args.output_dir / f"{args.provider}_{args.phase}_pointwise.jsonl", limit=args.limit)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
