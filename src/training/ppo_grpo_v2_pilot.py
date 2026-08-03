"""Arithmetic SFT warm start and outcome-blind feasibility pilot for v2."""

from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.utils import clip_grad_norm_
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.evaluation.arithmetic_rl_v2 import (
    ArithmeticProblemV2,
    generate_problems_v2,
    score_completion_v2,
    sft_completion,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def corpus_hash(problems: list[ArithmeticProblemV2]) -> str:
    payload = json.dumps([problem.to_dict() for problem in problems], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def render_prompt(tokenizer, problem: ArithmeticProblemV2) -> str:  # noqa: ANN001
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": problem.prompt}], tokenize=False, add_generation_prompt=True
    )


def encode_sft_example(tokenizer, problem: ArithmeticProblemV2, max_length: int) -> tuple[torch.Tensor, torch.Tensor]:  # noqa: ANN001
    prompt_ids = tokenizer(render_prompt(tokenizer, problem), add_special_tokens=False).input_ids
    completion_ids = tokenizer(sft_completion(problem) + tokenizer.eos_token, add_special_tokens=False).input_ids
    input_ids = (prompt_ids + completion_ids)[:max_length]
    prompt_length = min(len(prompt_ids), len(input_ids))
    labels = [-100] * prompt_length + input_ids[prompt_length:]
    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def train_arithmetic_sft(config: dict, output_dir: Path) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    set_seed(config["sft"]["problem_seed"])
    model_cfg, sft_cfg = config["model"], config["sft"]
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["base_model"], revision=model_cfg["revision"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["base_model"], revision=model_cfg["revision"], dtype=torch.float16
    ).cuda()
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = get_peft_model(
        model,
        LoraConfig(
            r=model_cfg["lora_rank"],
            lora_alpha=model_cfg["lora_alpha"],
            lora_dropout=model_cfg["lora_dropout"],
            target_modules=model_cfg["lora_targets"],
            task_type="CAUSAL_LM",
        ),
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=sft_cfg["learning_rate"], betas=(0.9, 0.95))
    problems = generate_problems_v2(sft_cfg["problem_seed"], sft_cfg["examples"])
    accumulation = int(sft_cfg["gradient_accumulation_steps"])
    total_micro_steps = int(sft_cfg["epochs"]) * len(problems)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    losses = []
    optimizer.zero_grad(set_to_none=True)
    model.train()
    for micro_step in range(total_micro_steps):
        problem = problems[micro_step % len(problems)]
        input_ids, labels = encode_sft_example(tokenizer, problem, sft_cfg["max_sequence_tokens"])
        input_ids = input_ids.unsqueeze(0).cuda()
        labels = labels.unsqueeze(0).cuda()
        loss = model(input_ids=input_ids, labels=labels).loss
        (loss / accumulation).backward()
        losses.append(float(loss.detach().cpu()))
        if (micro_step + 1) % accumulation == 0:
            clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if (micro_step + 1) % 100 == 0:
            print(json.dumps({"micro_step": micro_step + 1, "loss_mean_100": float(np.mean(losses[-100:]))}))
    model.config.use_cache = True
    merged = model.merge_and_unload()
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    manifest = {
        "status": "complete",
        "examples": len(problems),
        "epochs": sft_cfg["epochs"],
        "optimizer_steps": total_micro_steps // accumulation,
        "trainable_parameters": int(sum(parameter.numel() for parameter in trainable)),
        "corpus_sha256": corpus_hash(problems),
        "loss_first_100": float(np.mean(losses[:100])),
        "loss_last_100": float(np.mean(losses[-100:])),
        "training_seconds": time.perf_counter() - started,
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 2**20,
        "base_revision": model_cfg["revision"],
        "torch_version": torch.__version__,
    }
    (output_dir / "sft_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


@torch.inference_mode()
def run_feasibility_pilot(config: dict, sft_dir: Path, output_dir: Path) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    pilot_cfg = config["pilot"]
    set_seed(pilot_cfg["generation_seed"])
    tokenizer = AutoTokenizer.from_pretrained(sft_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(sft_dir, dtype=torch.float16).cuda().eval()
    problems = generate_problems_v2(
        pilot_cfg["problem_seed"], pilot_cfg["problems"], id_offset=1_000_000
    )
    rows = []
    group_contrast = []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for problem in problems:
        encoded = tokenizer(
            render_prompt(tokenizer, problem),
            return_tensors="pt",
            truncation=True,
            max_length=pilot_cfg["max_prompt_tokens"],
        ).to("cuda")
        generated = model.generate(
            **encoded,
            do_sample=True,
            temperature=pilot_cfg["temperature"],
            top_p=pilot_cfg["top_p"],
            num_return_sequences=pilot_cfg["generations_per_problem"],
            max_new_tokens=pilot_cfg["max_completion_tokens"],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        rewards = []
        for sequence in generated:
            completion_ids = sequence[encoded.input_ids.shape[1] :]
            completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
            reward, metadata = score_completion_v2(completion, problem, config["reward"])
            truncated = len(completion_ids) >= pilot_cfg["max_completion_tokens"] and (
                len(completion_ids) == 0 or completion_ids[-1].item() != tokenizer.eos_token_id
            )
            row = {
                "problem_id": problem.problem_id,
                "gold": problem.answer,
                "completion": completion,
                "reward": reward,
                "truncated": truncated,
                **metadata,
            }
            rows.append(row)
            rewards.append(reward)
        group_contrast.append(float(np.std(rewards)) > 0.0)
        if len(group_contrast) % 8 == 0:
            print(json.dumps({"pilot_problems_complete": len(group_contrast)}))
    gates = config["feasibility_gates"]
    metrics = {
        "numeric_parse_rate": float(np.mean([row["numeric_answer"] is not None for row in rows])),
        "numeric_exact_rate": float(np.mean([row["numeric_exact"] for row in rows])),
        "tagged_exact_rate": float(np.mean([row["tagged_exact"] for row in rows])),
        "truncation_rate": float(np.mean([row["truncated"] for row in rows])),
        "groups_with_reward_contrast": float(np.mean(group_contrast)),
        "reward_mean": float(np.mean([row["reward"] for row in rows])),
        "reward_variance": float(np.var([row["reward"] for row in rows], ddof=1)),
        "n_problems": len(problems),
        "n_completions": len(rows),
        "runtime_seconds": time.perf_counter() - started,
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 2**20,
    }
    gate_results = {
        "numeric_parse_rate": metrics["numeric_parse_rate"] >= gates["numeric_parse_rate_min"],
        "numeric_exact_rate": metrics["numeric_exact_rate"] >= gates["numeric_exact_rate_min"],
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
        "sft_manifest": json.loads((sft_dir / "sft_manifest.json").read_text(encoding="utf-8")),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pilot_predictions.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (output_dir / "pilot_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
