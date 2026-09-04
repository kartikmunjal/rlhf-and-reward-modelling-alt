#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference_serving.analysis import analyze

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, default=Path("inference_serving/study_config.json"))
parser.add_argument("--trials", type=Path, default=Path("results/inference_serving_v1/raw_trials.jsonl"))
parser.add_argument("--judge-ledger", type=Path, default=Path("results/inference_serving_v1/judge/anthropic_pointwise.jsonl"))
parser.add_argument("--output-dir", type=Path, default=Path("results/inference_serving_v1"))
args = parser.parse_args(); config = json.loads(args.config.read_text(encoding="utf-8"))
print(json.dumps(analyze(config, args.trials, args.judge_ledger, args.output_dir), indent=2))
