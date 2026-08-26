#!/usr/bin/env python3
"""Print a compact, read-only status summary for the resumable pipeline."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_judge_summeval.ledger import load_latest
from summarization_finetune.data import read_jsonl

checks = {
    "data_manifest": Path("data/processed/summarization_finetune_v1/data_manifest.json"),
    "sft_manifest": Path("checkpoints/summarization_sft_v1/run_manifest.json"),
    "dpo_manifest": Path("checkpoints/summarization_dpo_v1/run_manifest.json"),
    "final_metrics": Path("results/summarization_finetune_v1/metrics.json"),
    "final_report": Path("results/summarization_finetune_v1/report.md"),
}
for name,path in checks.items(): print(f"{name}: {'complete' if path.exists() else 'pending'}")
for model in ("base","sft","dpo"):
    path=Path(f"results/summarization_finetune_v1/generations/{model}.jsonl")
    print(f"{model}_generations: {len(read_jsonl(path)) if path.exists() else 0}/200")
candidate=Path("results/summarization_finetune_v1/candidates.jsonl")
print(f"candidates: {len(read_jsonl(candidate)) if candidate.exists() else 0}/1024")
for name,total in (("anthropic_candidate_pairs",1536),("anthropic_pointwise",600),("openai_pointwise",600)):
    path=Path(f"results/summarization_finetune_v1/judge/{name}.jsonl")
    latest=load_latest(path); success=sum(row['status']=='success' for row in latest.values())
    print(f"{name}: {success}/{total} successful; {len(latest)-success} terminal failures")
