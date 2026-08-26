#!/usr/bin/env python3
"""Resumable end-to-end driver for the preregistered summarization study."""

from __future__ import annotations

import os, subprocess, sys
from pathlib import Path


ROOT=Path(__file__).resolve().parents[1]
DATA=Path("data/processed/summarization_finetune_v1")
RESULTS=Path("results/summarization_finetune_v1")


def run(*args):
    command=[sys.executable,*map(str,args)]; print("+"," ".join(command),flush=True); subprocess.run(command,check=True)


def load_dotenv(path=Path(".env")):
    if not path.is_file(): return
    for line in path.read_text(encoding="utf-8").splitlines():
        line=line.strip()
        if not line or line.startswith("#") or "=" not in line: continue
        key,value=line.split("=",1); os.environ.setdefault(key.strip(),value.strip())


def main():
    load_dotenv()
    if not DATA.joinpath("data_manifest.json").exists(): run("scripts/prepare_summarization_finetune_data.py")
    final_articles=DATA/"final_eval_articles.jsonl"; preference_articles=DATA/"preference_articles.jsonl"
    generations=RESULTS/"generations"; judge=RESULTS/"judge"; generations.mkdir(parents=True,exist_ok=True); judge.mkdir(parents=True,exist_ok=True)
    run("scripts/generate_summarization_outputs.py","--checkpoint","gpt2-medium","--mode","evaluation","--model-label","base","--articles",final_articles,"--output",generations/"base.jsonl")
    if not Path("checkpoints/summarization_sft_v1/run_manifest.json").exists(): run("scripts/train_summarization_sft.py")
    sft_checkpoint=Path("checkpoints/summarization_sft_v1/merged")
    run("scripts/generate_summarization_outputs.py","--checkpoint",sft_checkpoint,"--mode","evaluation","--model-label","sft","--articles",final_articles,"--output",generations/"sft.jsonl")
    candidates=RESULTS/"candidates.jsonl"
    run("scripts/generate_summarization_outputs.py","--checkpoint",sft_checkpoint,"--mode","candidates","--articles",preference_articles,"--output",candidates)
    if not os.getenv("ANTHROPIC_API_KEY") or not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY and OPENAI_API_KEY are required for judge stages")
    run("scripts/run_summarization_judge.py","--kind","candidate-pairs","--input",candidates,"--ledger",judge/"anthropic_candidate_pairs.jsonl")
    preferences=DATA/"dpo_preferences.jsonl"
    run("scripts/build_summarization_dpo_preferences.py","--candidates",candidates,"--ledger",judge/"anthropic_candidate_pairs.jsonl","--output",preferences)
    if not Path("checkpoints/summarization_dpo_v1/run_manifest.json").exists(): run("scripts/train_summarization_dpo.py")
    run("scripts/generate_summarization_outputs.py","--checkpoint","checkpoints/summarization_dpo_v1/merged","--mode","evaluation","--model-label","dpo","--articles",final_articles,"--output",generations/"dpo.jsonl")
    for provider in ("anthropic","openai"):
        for model in ("base","sft","dpo"):
            run("scripts/run_summarization_judge.py","--kind","pointwise","--provider",provider,"--input",generations/f"{model}.jsonl","--ledger",judge/f"{provider}_pointwise.jsonl")
    run("scripts/analyze_summarization_finetune.py")


if __name__=="__main__": main()
