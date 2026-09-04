#!/usr/bin/env python3
"""Score quantized and FP16 outputs with the unchanged frozen Claude judge."""

import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from summarization_finetune.judging import score_pointwise

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--ledger", type=Path, default=Path("results/inference_serving_v1/judge/anthropic_pointwise.jsonl"))
parser.add_argument("--judge-config", type=Path, default=Path("summarization_finetune/study_config.json"))
args = parser.parse_args(); config = json.loads(args.judge_config.read_text(encoding="utf-8"))
print(json.dumps(score_pointwise(generations_path=args.input, ledger_path=args.ledger,
                                 provider_name="anthropic", config=config,
                                 root=Path(__file__).resolve().parents[1]), indent=2))
