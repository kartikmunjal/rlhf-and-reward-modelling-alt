"""Execute the separately recorded harder-distribution v2b feasibility pilot."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.evaluation.arithmetic_rl_v2b import generate_problems_v2b, score_completion_v2b
from src.training.ppo_grpo_v2_pilot import corpus_hash, render_prompt, set_seed


@torch.inference_mode()
def run_v2b_pilot(
    config: dict,
    output_dir: Path,
    *,
    problem_generator=generate_problems_v2b,
    scorer=score_completion_v2b,
) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    pilot = config["pilot"]
    sft_dir = Path(config["sft_checkpoint"])
    set_seed(pilot["generation_seed"])
    tokenizer = AutoTokenizer.from_pretrained(sft_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(sft_dir, dtype=torch.float16).cuda().eval()
    problems = problem_generator(pilot["problem_seed"], pilot["problems"], id_offset=2_000_000)
    rows, contrasts = [], []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for problem in problems:
        encoded = tokenizer(
            render_prompt(tokenizer, problem),
            return_tensors="pt",
            truncation=True,
            max_length=pilot["max_prompt_tokens"],
        ).to("cuda")
        generated = model.generate(
            **encoded,
            do_sample=True,
            temperature=pilot["temperature"],
            top_p=pilot["top_p"],
            num_return_sequences=pilot["generations_per_problem"],
            max_new_tokens=pilot["max_completion_tokens"],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        rewards = []
        for sequence in generated:
            completion_ids = sequence[encoded.input_ids.shape[1] :]
            completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
            reward, metadata = scorer(completion, problem, config["reward"])
            truncated = len(completion_ids) >= pilot["max_completion_tokens"] and (
                len(completion_ids) == 0 or completion_ids[-1].item() != tokenizer.eos_token_id
            )
            rows.append(
                {
                    "problem_id": problem.problem_id,
                    "gold": problem.answer,
                    "completion": completion,
                    "reward": reward,
                    "truncated": truncated,
                    **metadata,
                }
            )
            rewards.append(reward)
        contrasts.append(float(np.std(rewards)) > 0.0)
        if len(contrasts) % 8 == 0:
            print(json.dumps({"pilot_problems_complete": len(contrasts)}), flush=True)
    metrics = {
        "numeric_parse_rate": float(np.mean([row["numeric_answer"] is not None for row in rows])),
        "numeric_exact_rate": float(np.mean([row["numeric_exact"] for row in rows])),
        "tagged_exact_rate": float(np.mean([row["tagged_exact"] for row in rows])),
        "truncation_rate": float(np.mean([row["truncated"] for row in rows])),
        "groups_with_reward_contrast": float(np.mean(contrasts)),
        "reward_mean": float(np.mean([row["reward"] for row in rows])),
        "reward_variance": float(np.var([row["reward"] for row in rows], ddof=1)),
        "n_problems": len(problems),
        "n_completions": len(rows),
        "runtime_seconds": time.perf_counter() - started,
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 2**20,
    }
    gates = config["feasibility_gates"]
    gate_results = {
        "numeric_parse_rate": metrics["numeric_parse_rate"] >= gates["numeric_parse_rate_min"],
        "numeric_exact_rate_min": metrics["numeric_exact_rate"] >= gates["numeric_exact_rate_min"],
        "numeric_exact_rate_max": metrics["numeric_exact_rate"] <= gates["numeric_exact_rate_max"],
        "truncation_rate": metrics["truncation_rate"] <= gates["truncation_rate_max"],
        "groups_with_reward_contrast": metrics["groups_with_reward_contrast"]
        >= gates["groups_with_reward_contrast_min"],
    }
    result = {
        "study_id": config["study_id"],
        "status": "pass" if all(gate_results.values()) else "fail",
        "metrics": metrics,
        "gates": gates,
        "gate_results": gate_results,
        "pilot_problem_sha256": corpus_hash(problems),
        "parent_pilot": config["parent_pilot"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pilot_predictions.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (output_dir / "pilot_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
