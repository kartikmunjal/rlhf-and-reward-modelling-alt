#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from summarization_finetune.judging import build_preferences

parser = argparse.ArgumentParser()
parser.add_argument("--candidates", type=Path, default=Path("results/summarization_finetune_v1/candidates.jsonl"))
parser.add_argument("--ledger", type=Path, default=Path("results/summarization_finetune_v1/judge/anthropic_candidate_pairs.jsonl"))
parser.add_argument("--output", type=Path, default=Path("data/processed/summarization_finetune_v1/dpo_preferences.jsonl"))
args = parser.parse_args(); print(json.dumps(build_preferences(candidates_path=args.candidates, ledger_path=args.ledger, output_path=args.output), indent=2))
