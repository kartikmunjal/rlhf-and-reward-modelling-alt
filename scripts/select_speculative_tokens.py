#!/usr/bin/env python3
"""Freeze the preregistered pilot choice of speculative-token count."""
import argparse, hashlib, json, statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference_serving.data import read_jsonl

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, default=Path("inference_serving/study_config.json"))
parser.add_argument("--trials", type=Path, default=Path("results/inference_serving_v1/raw_trials.jsonl"))
parser.add_argument("--output", type=Path, default=Path("results/inference_serving_v1/speculative_selection.json"))
args = parser.parse_args()
if args.output.exists():
    raise SystemExit(f"Selection is already frozen: {args.output}")
config = json.loads(args.config.read_text(encoding="utf-8"))
candidates = config["stage3"]["pilot_speculative_tokens"]
rows = read_jsonl(args.trials)
medians = {}
for candidate in candidates:
    values = [float(r["output_tokens_per_second"]) for r in rows
              if r.get("phase") == "pilot" and r.get("speculative")
              and int(r.get("speculative_tokens", -1)) == candidate and r.get("target") == "dpo"]
    if len(values) != config["stage3"]["pilot_trials"]:
        raise SystemExit(f"k={candidate}: expected {config['stage3']['pilot_trials']} trials, found {len(values)}")
    medians[str(candidate)] = {"median_output_tokens_per_second": statistics.median(values), "n_trials": len(values)}
best = min(candidates, key=lambda k: (-medians[str(k)]["median_output_tokens_per_second"], k))
payload = {"selected_speculative_tokens": best, "rule": "highest median pilot throughput; ties choose smaller k",
           "candidates": medians, "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
           "trials_sha256": hashlib.sha256(args.trials.read_bytes()).hexdigest()}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
