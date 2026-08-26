"""Deterministic evaluation and seeded DPO-candidate generation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from summarization_finetune.data import read_jsonl, write_jsonl


def prompt_ids(tokenizer, source: str, source_tokens: int):
    prefix = tokenizer.encode("Article:\n", add_special_tokens=False)
    article = tokenizer.encode(source, add_special_tokens=False)[:source_tokens]
    suffix = tokenizer.encode("\n\nSummary:\n", add_special_tokens=False)
    return prefix + article + suffix


def load_model(checkpoint: str, device: str = "cuda"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint); tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    model.config.pad_token_id = tokenizer.pad_token_id; model.eval().to(device)
    return model, tokenizer


def _decode_new(model, tokenizer, ids: list[int], generation: dict, *, seed: int | None = None) -> str:
    import torch
    inputs = torch.tensor([ids], device=model.device)
    if seed is not None:
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    with torch.inference_mode():
        output = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                                **generation)
    return tokenizer.decode(output[0, len(ids):], skip_special_tokens=True).strip()


def generate_evaluation(*, checkpoint: str, model_label: str, articles_path: Path, output_path: Path,
                        config: dict, device: str = "cuda") -> dict:
    model, tokenizer = load_model(checkpoint, device)
    existing = {row["article_id"]: row for row in read_jsonl(output_path)} if output_path.exists() else {}
    generation = {"do_sample": False, "max_new_tokens": config["evaluation_generation"]["max_new_tokens"],
                  "no_repeat_ngram_size": config["evaluation_generation"]["no_repeat_ngram_size"]}
    rows = []
    for article in read_jsonl(articles_path):
        row = existing.get(article["article_id"])
        if row is None:
            ids = prompt_ids(tokenizer, article["source"], config["model"]["source_tokens"])
            row = {**article, "model": model_label,
                   "summary": _decode_new(model, tokenizer, ids, generation)}
        rows.append(row)
        write_jsonl(output_path, rows)
    return {"model": model_label, "articles": len(rows), "output": str(output_path)}


def candidate_seed(study_seed: int, article_id: str, candidate_index: int) -> int:
    digest = hashlib.sha256(f"{study_seed}\0{article_id}\0{candidate_index}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def generate_candidates(*, checkpoint: str, articles_path: Path, output_path: Path,
                        config: dict, device: str = "cuda") -> dict:
    model, tokenizer = load_model(checkpoint, device)
    existing = {(row["article_id"], row["candidate_index"]): row for row in read_jsonl(output_path)} if output_path.exists() else {}
    settings = config["dpo_candidates"]
    generation = {"do_sample": True, "temperature": settings["temperature"], "top_p": settings["top_p"],
                  "top_k": settings["top_k"], "max_new_tokens": settings["max_new_tokens"], "no_repeat_ngram_size": 3}
    rows = []
    for article in read_jsonl(articles_path):
        ids = prompt_ids(tokenizer, article["source"], config["model"]["source_tokens"])
        for candidate_index in range(settings["candidates_per_article"]):
            key = (article["article_id"], candidate_index)
            row = existing.get(key)
            if row is None:
                seed = candidate_seed(config["seed"], *key)
                row = {**article, "candidate_index": candidate_index,
                       "candidate_id": f"{article['article_id']}:c{candidate_index}", "seed": seed,
                       "summary": _decode_new(model, tokenizer, ids, generation, seed=seed)}
            rows.append(row)
        write_jsonl(output_path, rows)
    return {"articles": len({row["article_id"] for row in rows}), "candidates": len(rows), "output": str(output_path)}
