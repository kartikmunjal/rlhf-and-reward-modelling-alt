#!/usr/bin/env python3
"""Generate a conservative pre-run token and API cost estimate."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_judge_summeval.data import load_heldout_inputs
from llm_judge_summeval.prompts import load_prompts, render_pairwise, render_pointwise
from llm_judge_summeval.runner import read_jsonl


def tokens(text: str) -> int:
    return math.ceil(len(text) / 3.5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/summeval_judge_v1"))
    parser.add_argument("--prompts", type=Path, default=Path("llm_judge_summeval/prompts.json"))
    parser.add_argument("--pricing", type=Path, default=Path("llm_judge_summeval/runtime_pricing.json"))
    parser.add_argument("--output", type=Path, default=Path("llm_judge_summeval/cost_estimate.json"))
    args = parser.parse_args()
    prompts, pricing = load_prompts(args.prompts), json.loads(args.pricing.read_text())
    rows = load_heldout_inputs(args.processed_dir)
    by_id = {row["summary_id"]: row for row in rows}
    pointwise_input = sum(tokens("\n".join(render_pointwise(prompts, row["source"], row["summary"]))) for row in rows)
    pairs = read_jsonl(args.processed_dir / "heldout_pairs.jsonl")
    pairwise_input = 0
    for pair in pairs:
        left, right = by_id[pair["summary_a_id"]], by_id[pair["summary_b_id"]]
        pairwise_input += 2 * tokens("\n".join(render_pairwise(prompts, left["source"], left["summary"], right["summary"])))
    # Conservative response budgets derived from the fixed four-axis schemas.
    pointwise_output, pairwise_output = len(rows) * 500, len(pairs) * 2 * 500
    rates = pricing["per_million_tokens"]
    claude_cost = (pointwise_input + pairwise_input) * rates["claude-haiku-4-5-20251001"]["input"] / 1e6 + (pointwise_output + pairwise_output) * rates["claude-haiku-4-5-20251001"]["output"] / 1e6
    openai_cost = pointwise_input * rates["gpt-5-mini-2025-08-07"]["input"] / 1e6 + pointwise_output * rates["gpt-5-mini-2025-08-07"]["output"] / 1e6
    result = {"method": "conservative_character_estimate_not_actual_billing", "heldout": {
        "claude": {"requests": len(rows) + 2 * len(pairs), "estimated_input_tokens": pointwise_input + pairwise_input,
                   "budgeted_output_tokens": pointwise_output + pairwise_output, "estimated_usd": claude_cost},
        "openai": {"requests": len(rows), "estimated_input_tokens": pointwise_input,
                   "budgeted_output_tokens": pointwise_output, "estimated_usd": openai_cost},
        "estimated_total_usd": claude_cost + openai_cost}, "pricing": pricing}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["heldout"], indent=2))


if __name__ == "__main__":
    main()
