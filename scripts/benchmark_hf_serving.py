#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference_serving.benchmark import append_trial, benchmark_prompts, run_hf_trial

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, default=Path("inference_serving/study_config.json"))
parser.add_argument("--model", required=True)
parser.add_argument("--articles", type=Path, default=Path("data/processed/inference_serving_v1/heldout_articles.jsonl"))
parser.add_argument("--output", type=Path, default=Path("results/inference_serving_v1/raw_trials.jsonl"))
parser.add_argument("--trials", type=int)
parser.add_argument("--requests", type=int)
parser.add_argument("--concurrency", type=int, nargs="+")
args = parser.parse_args(); config = json.loads(args.config.read_text(encoding="utf-8"))
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained(args.model)
trials = args.trials or config["stage1"]["benchmark_trials"]
requests = args.requests or config["stage1"]["requests_per_trial"]
prompts = benchmark_prompts(args.articles, config, requests)
for concurrency in args.concurrency or config["stage1"]["concurrency"]:
    run_hf_trial(model, tokenizer, prompts[:config["stage1"]["warmup_requests"]], concurrency=concurrency, new_tokens=config["generation"]["performance_new_tokens"])
    for trial_index in range(trials):
        row = run_hf_trial(model, tokenizer, prompts, concurrency=concurrency, new_tokens=config["generation"]["performance_new_tokens"])
        row.update({"system": "hf", "target": "dpo", "precision": "fp16", "speculative": False,
                    "concurrency": concurrency, "trial_index": trial_index})
        append_trial(args.output, row); print(json.dumps({key: value for key, value in row.items() if key != "request_metrics"}), flush=True)
