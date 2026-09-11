#!/usr/bin/env python3
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from recursive_self_improvement.analysis import analyze, read_jsonl, write_results
from recursive_self_improvement.config import load_effective_config

parser = argparse.ArgumentParser()
parser.add_argument("--evaluations", type=Path, default=Path("results/recursive_self_improvement_v1/evaluations.jsonl"))
parser.add_argument("--compute", type=Path, default=Path("results/recursive_self_improvement_v1/compute.jsonl"))
parser.add_argument("--stage3", type=Path, default=Path("results/recursive_self_improvement_v1/stage3_evaluations.jsonl"))
parser.add_argument("--stage3-training-audit", type=Path, default=Path("results/recursive_self_improvement_v1/stage3_training_label_audit.jsonl"))
parser.add_argument("--output-dir", type=Path, default=Path("results/recursive_self_improvement_v1"))
args = parser.parse_args(); config = load_effective_config(ROOT)
audit = read_jsonl(args.stage3_training_audit) if args.stage3_training_audit.exists() else []
stage3 = read_jsonl(args.stage3) if args.stage3.exists() else []
metrics = analyze(config, read_jsonl(args.evaluations), read_jsonl(args.compute), stage3, audit)
write_results(metrics, args.output_dir); print(args.output_dir / "metrics.json")
