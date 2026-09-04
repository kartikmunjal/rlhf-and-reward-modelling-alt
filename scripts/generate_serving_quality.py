#!/usr/bin/env python3
"""Generate held-out summaries through a running OpenAI-compatible vLLM server."""

import argparse, json, sys, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference_serving.data import article_id, prompt_text, read_jsonl, write_jsonl

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, default=Path("inference_serving/study_config.json"))
parser.add_argument("--base-url", default="http://127.0.0.1:8000")
parser.add_argument("--served-model", required=True)
parser.add_argument("--model-label", choices=("dpo_fp16", "dpo_gptq"), required=True)
parser.add_argument("--articles", type=Path, default=Path("data/processed/inference_serving_v1/heldout_articles.jsonl"))
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args(); config = json.loads(args.config.read_text(encoding="utf-8"))
existing = {row["article_id"]: row for row in read_jsonl(args.output)} if args.output.exists() else {}
rows = []
for article in read_jsonl(args.articles):
    item_id = article_id(article); row = existing.get(item_id)
    if row is None:
        payload = json.dumps({"model": args.served_model, "prompt": prompt_text(article["source"], config),
                              "max_tokens": config["generation"]["quality_new_tokens"], "temperature": 0,
                              "ignore_eos": False}).encode()
        request = urllib.request.Request(args.base_url.rstrip("/") + "/v1/completions", data=payload,
                                         headers={"Content-Type": "application/json"})
        response = json.loads(urllib.request.urlopen(request, timeout=600).read())
        row = {"article_id": item_id, "source": article["source"], "reference": article["reference"],
               "model": args.model_label, "summary": response["choices"][0]["text"],
               "completion_tokens": response.get("usage", {}).get("completion_tokens")}
    rows.append(row); write_jsonl(args.output, rows)
print(json.dumps({"model": args.model_label, "articles": len(rows), "output": str(args.output)}, indent=2))
