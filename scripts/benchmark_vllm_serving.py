#!/usr/bin/env python3
import argparse, asyncio, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference_serving.benchmark import append_trial, benchmark_prompts, run_vllm_trial

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, default=Path("inference_serving/study_config.json"))
parser.add_argument("--base-url", default="http://127.0.0.1:8000")
parser.add_argument("--served-model", required=True)
parser.add_argument("--target", choices=("base", "sft", "dpo"), required=True)
parser.add_argument("--precision", choices=("fp16", "gptq"), required=True)
parser.add_argument("--speculative", action="store_true")
parser.add_argument("--articles", type=Path, default=Path("data/processed/inference_serving_v1/heldout_articles.jsonl"))
parser.add_argument("--output", type=Path, default=Path("results/inference_serving_v1/raw_trials.jsonl"))
parser.add_argument("--trials", type=int)
parser.add_argument("--requests", type=int)
parser.add_argument("--concurrency", type=int, nargs="+")
parser.add_argument("--speculative-tokens", type=int)
parser.add_argument("--phase", choices=("pilot", "heldout"), default="heldout")
args = parser.parse_args(); config = json.loads(args.config.read_text(encoding="utf-8"))
trials = args.trials or config["stage1"]["benchmark_trials"]
requests = args.requests or config["stage1"]["requests_per_trial"]
prompts = benchmark_prompts(args.articles, config, requests)
for concurrency in args.concurrency or config["stage1"]["concurrency"]:
    asyncio.run(run_vllm_trial(args.base_url, args.served_model, prompts[:config["stage1"]["warmup_requests"]], concurrency=concurrency, new_tokens=config["generation"]["performance_new_tokens"]))
    for trial_index in range(trials):
        row = asyncio.run(run_vllm_trial(args.base_url, args.served_model, prompts, concurrency=concurrency, new_tokens=config["generation"]["performance_new_tokens"]))
        row.update({"system": "vllm", "target": args.target, "precision": args.precision, "speculative": args.speculative,
                    "concurrency": concurrency, "trial_index": trial_index, "phase": args.phase})
        row["speculative_tokens"] = args.speculative_tokens
        append_trial(args.output, row); print(json.dumps({key: value for key, value in row.items() if key != "request_metrics"}), flush=True)
