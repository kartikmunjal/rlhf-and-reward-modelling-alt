#!/usr/bin/env python3
"""Audit Stage-3 training labels with the frozen, evaluation-only K=3 RMs.

This script is intentionally post-training.  It never writes preference files or
checkpoints, and therefore cannot leak the evaluation ensemble into Stage-3
optimization.
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recursive_self_improvement.config import load_effective_config
from scripts.train_recursive_stage1 import encode, load_reward, read_jsonl, write_jsonl


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _complete(path: Path) -> bool:
    try:
        return json.loads(path.read_text(encoding="utf-8"))["status"] == "complete"
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return False


def _load_preferences(config: dict) -> list[dict]:
    rows: list[dict] = []
    for percent in config["stage3"]["label_mixture_percent_self"]:
        for round_index in range(1, config["stage3"]["rounds_per_condition"] + 1):
            path = (
                ROOT
                / "results/recursive_self_improvement_v1/stage3"
                / f"self_{percent}/round_{round_index}/preferences.jsonl"
            )
            condition = read_jsonl(path)
            expected = config["stage1"]["rollout_prompts_per_round"]
            if len(condition) != expected:
                raise ValueError(f"Expected {expected} preferences in {path}, found {len(condition)}")
            for row in condition:
                rows.append(
                    {
                        **row,
                        "condition_id": f"self_{percent}_round_{round_index}",
                    }
                )
    return rows


def _score_member(config: dict, rows: list[dict], seed: int, tokenizer, torch, device, output: Path) -> None:
    manifest = output.with_suffix(".manifest.json")
    if _complete(manifest):
        return
    model = load_reward(
        config,
        ROOT / "checkpoints/recursive_self_improvement_v1/sft",
        ROOT / f"checkpoints/recursive_self_improvement_v1/reward_ensemble/seed{seed}",
        device,
    ).eval()
    batch_size = 8
    scored: list[dict] = []
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            chosen = [row["prompt"] + row["chosen"] for row in batch]
            rejected = [row["prompt"] + row["rejected"] for row in batch]
            with torch.autocast("cuda", dtype=torch.float16):
                chosen_scores = model(
                    **encode(tokenizer, chosen, config["model"]["context_length"], device)
                ).rewards.flatten()
                rejected_scores = model(
                    **encode(tokenizer, rejected, config["model"]["context_length"], device)
                ).rewards.flatten()
            for row, margin in zip(batch, (chosen_scores - rejected_scores).tolist()):
                scored.append(
                    {
                        "condition_id": row["condition_id"],
                        "percent_self": row["percent_self"],
                        "round": row["round"],
                        "prompt_id": row["prompt_id"],
                        "label_source": row["label_source"],
                        "seed": seed,
                        "chosen_minus_rejected_reward": float(margin),
                    }
                )
            if len(scored) % 256 == 0:
                print(json.dumps({"reward_seed": seed, "scored": len(scored), "total": len(rows)}), flush=True)
    write_jsonl(output, scored)
    _save_json(
        manifest,
        {
            "status": "complete",
            "rows": len(scored),
            "sha256": _sha256(output),
            "runtime_seconds": time.perf_counter() - started,
            "evaluation_only": True,
        },
    )
    del model
    torch.cuda.empty_cache()


def _assemble(config: dict, rows: list[dict], output_root: Path) -> None:
    member_scores: dict[tuple[str, str], list[float]] = {}
    for path in sorted((output_root / "member_scores").glob("seed*.jsonl")):
        for row in read_jsonl(path):
            key = (row["condition_id"], row["prompt_id"])
            member_scores.setdefault(key, []).append(row["chosen_minus_rejected_reward"])
    output: list[dict] = []
    for row in rows:
        key = (row["condition_id"], row["prompt_id"])
        margins = member_scores.get(key, [])
        if len(margins) != config["reward_ensemble"]["members"]:
            raise ValueError(f"Expected K=3 audit scores for {key}, found {len(margins)}")
        mean_margin = statistics.mean(margins)
        output.append(
            {
                "condition_id": row["condition_id"],
                "percent_self": row["percent_self"],
                "round": row["round"],
                "prompt_id": row["prompt_id"],
                "label_source": row["label_source"],
                "external_evaluator_agrees": 1.0 if mean_margin > 0 else 0.0 if mean_margin < 0 else 0.5,
                "external_margin": mean_margin,
                "ensemble_disagreement": statistics.pstdev(margins),
                "member_margins": margins,
            }
        )
    destination = ROOT / "results/recursive_self_improvement_v1/stage3_training_label_audit.jsonl"
    write_jsonl(destination, output)
    _save_json(
        output_root / "audit_manifest.json",
        {
            "status": "complete",
            "rows": len(output),
            "sha256": _sha256(destination),
            "evaluation_only": True,
            "expected_conditions": len(config["stage3"]["label_mixture_percent_self"])
            * config["stage3"]["rounds_per_condition"],
        },
    )


def main() -> None:
    import torch
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    config = load_effective_config(ROOT)
    rows = _load_preferences(config)
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["base"], revision=config["model"]["base_revision"]
    )
    tokenizer.pad_token = tokenizer.eos_token
    device = torch.device("cuda")
    output_root = ROOT / "results/recursive_self_improvement_v1/stage3_training_label_audit"
    for seed in config["reward_ensemble"]["member_seeds"]:
        _score_member(config, rows, seed, tokenizer, torch, device, output_root / "member_scores" / f"seed{seed}.jsonl")
    _assemble(config, rows, output_root)


if __name__ == "__main__":
    main()
