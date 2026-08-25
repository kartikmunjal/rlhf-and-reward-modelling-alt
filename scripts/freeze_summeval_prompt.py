#!/usr/bin/env python3
"""Freeze prompts after a valid development run and before held-out execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_judge_summeval.data import load_prompt_development_rows
from llm_judge_summeval.prompts import canonical_sha256, load_prompts, render_pointwise
from llm_judge_summeval.schemas import pointwise_schema


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-ledger", type=Path, default=Path("results/summeval_judge_v1/anthropic_dev_pointwise.jsonl"))
    parser.add_argument("--prompts", type=Path, default=Path("llm_judge_summeval/prompts.json"))
    parser.add_argument("--config", type=Path, default=Path("llm_judge_summeval/study_config.json"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/summeval_judge_v1"))
    parser.add_argument("--output", type=Path, default=Path("llm_judge_summeval/final_prompt_manifest.json"))
    parser.add_argument("--approve", action="store_true", help="Explicitly attest prompt selection is final")
    args = parser.parse_args()
    if not args.approve:
        raise SystemExit("Refusing to freeze without --approve")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    prompts = load_prompts(args.prompts)
    model = config["judges"]["primary"]["model"]
    expected_ids = set()
    dev_rows = load_prompt_development_rows(args.processed_dir)
    for row in dev_rows:
        system, user = render_pointwise(prompts, row["source"], row["summary"])
        expected_ids.add(canonical_sha256({"provider": "anthropic", "model": model, "kind": "pointwise",
                                             "item_id": row["summary_id"], "system": system, "user": user,
                                             "schema": pointwise_schema()}))
    if not args.dev_ledger.is_file():
        raise SystemExit(f"Development ledger not found: {args.dev_ledger}")
    ledger_rows = [json.loads(line) for line in args.dev_ledger.read_text(encoding="utf-8").splitlines()]
    latest = {row["request_id"]: row for row in ledger_rows}
    successes = [latest[request_id] for request_id in expected_ids
                 if request_id in latest and latest[request_id]["status"] == "success"]
    required = int(len(expected_ids) * config["missingness"]["minimum_valid_fraction"] + 0.999999)
    if len(successes) < required:
        raise SystemExit(f"Current prompt/model has {len(successes)} valid dev judgments; {required} required")
    manifest = {"status": "frozen_after_dev_before_heldout", "prompts_sha256": sha(args.prompts),
                "config_sha256": canonical_sha256(config), "dev_ledger_sha256": sha(args.dev_ledger),
                "provider": "anthropic", "model": model, "successful_dev_requests": len(successes),
                "expected_dev_requests": len(expected_ids), "minimum_required": required}
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Wrote", args.output)


if __name__ == "__main__":
    main()
