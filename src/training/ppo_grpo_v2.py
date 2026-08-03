"""Preregistered PPO-vs-GRPO v2 training and evaluation implementation."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from torch import nn
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from src.evaluation.arithmetic_rl_v2 import parse_final_integer, parse_tagged_answer
from src.evaluation.arithmetic_rl_v2c import generate_problems_v2c, score_completion_v2c
from src.training.ppo_grpo_v2_pilot import render_prompt, set_seed


class _TokenBackbone(nn.Module):
    def forward(self, input_ids, **kwargs):  # noqa: ANN001
        _ = kwargs
        return SimpleNamespace(hidden_states=(input_ids.unsqueeze(-1).to(torch.float32),))


class V2RuleRewardModel(nn.Module):
    base_model_prefix = "backbone"

    def __init__(self, tokenizer, problems, reward_config):  # noqa: ANN001
        super().__init__()
        self.backbone = _TokenBackbone()
        self.tokenizer = tokenizer
        self.problems = {problem.problem_id: problem for problem in problems}
        self.reward_config = reward_config
        self.register_buffer("anchor", torch.zeros(1), persistent=False)

    def score(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ids = hidden_states[..., 0].round().to(torch.long)
        texts = self.tokenizer.batch_decode(ids, skip_special_tokens=False)
        rows = []
        for text in texts:
            prompt_part, completion = text.rsplit("<|im_start|>assistant\n", 1)
            marker = prompt_part.rsplit("Problem ID:", 1)[-1].splitlines()[0].strip()
            problem = self.problems.get(int(marker)) if marker.isdigit() else None
            value = score_completion_v2c(completion, problem, self.reward_config)[0] if problem else -1.0
            rows.append(torch.full((ids.shape[1], 1), value, device=ids.device))
        return torch.stack(rows)


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _common(config: dict):
    checkpoint = config["model"]["sft_checkpoint"]
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model_cfg = config["model"]
    peft = LoraConfig(
        r=model_cfg["lora_rank"],
        lora_alpha=model_cfg["lora_alpha"],
        lora_dropout=model_cfg["lora_dropout"],
        target_modules=model_cfg["lora_targets"],
        task_type="CAUSAL_LM",
    )
    return checkpoint, tokenizer, peft


def _dataset(problems, tokenizer) -> Dataset:  # noqa: ANN001
    rows = [{"prompt": render_prompt(tokenizer, problem), "problem_id": problem.problem_id} for problem in problems]
    dataset = Dataset.from_list(rows)

    def tokenize(row):
        return tokenizer(row["prompt"], truncation=True, max_length=192, padding=False)

    return dataset.map(tokenize)


def _assert_budget(method: str, trajectory: list[dict], config: dict) -> dict:
    if method == "ppo":
        groups = len([row for row in trajectory if "objective/scores" in row])
        optimizer_steps = groups * config["compute_budget"]["updates_per_rollout_group"]
    else:
        groups = len([row for row in trajectory if "reward" in row])
        optimizer_steps = max(int(row.get("step", 0)) for row in trajectory)
    completions = groups * config["compute_budget"]["generations_per_group"]
    expected = config["runtime_assertions"]
    observed = {"optimizer_steps": optimizer_steps, "rollout_groups": groups, "generated_completions": completions}
    required = {
        "optimizer_steps": expected["optimizer_steps_must_equal"],
        "rollout_groups": expected["rollout_groups_must_equal"],
        "generated_completions": expected["generated_completions_must_equal"],
    }
    if observed != required:
        raise RuntimeError(f"Compute-budget assertion failed: observed={observed}, required={required}")
    return observed


def _save_run(output_dir: Path, trainer, started: float, method: str, seed: int, config: dict) -> dict:
    trajectory = list(trainer.state.log_history)
    budget = _assert_budget(method, trajectory, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    trainer.processing_class.save_pretrained(output_dir)
    manifest = {
        "method": method,
        "seed": seed,
        "training_seconds": time.perf_counter() - started,
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 2**20,
        "observed_budget": budget,
        "sft_model_sha256": config["model"]["sft_model_sha256"],
        "torch_version": torch.__version__,
    }
    (output_dir / "trajectory.json").write_text(json.dumps(trajectory, indent=2), encoding="utf-8")
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def train_grpo_v2(config: dict, seed: int, output_dir: Path) -> dict:
    from trl import GRPOConfig, GRPOTrainer

    checkpoint, tokenizer, peft = _common(config)
    budget, optimization = config["compute_budget"], config["optimization"]
    problems = generate_problems_v2c(config["task"]["train_problem_seed"], budget["rollout_groups"])
    problem_map = {problem.problem_id: problem for problem in problems}
    dataset = _dataset(problems, tokenizer)

    def verifier(completions, problem_id, **kwargs):  # noqa: ANN001
        _ = kwargs
        return [
            score_completion_v2c(text, problem_map[int(identifier)], config["reward"])[0]
            for text, identifier in zip(completions, problem_id)
        ]

    args = GRPOConfig(
        output_dir=str(output_dir),
        max_steps=budget["optimizer_steps"],
        per_device_train_batch_size=2,
        learning_rate=optimization["learning_rate"],
        adam_beta1=optimization["adam_beta1"],
        adam_beta2=optimization["adam_beta2"],
        max_grad_norm=optimization["max_grad_norm"],
        num_generations=budget["generations_per_group"],
        num_iterations=1,
        steps_per_generation=budget["updates_per_rollout_group"],
        max_completion_length=budget["max_completion_tokens"],
        beta=optimization["kl_coefficient"],
        epsilon=optimization["clip_epsilon"],
        temperature=config["generation"]["temperature"],
        top_p=config["generation"]["top_p"],
        fp16=True,
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        seed=seed,
    )
    set_seed(seed)
    torch.cuda.reset_peak_memory_stats()
    trainer = GRPOTrainer(
        model=checkpoint,
        reward_funcs=verifier,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft,
    )
    started = time.perf_counter()
    trainer.train()
    return _save_run(output_dir, trainer, started, "grpo", seed, config)


def train_ppo_v2(config: dict, seed: int, output_dir: Path) -> dict:
    from trl.experimental.ppo import PPOConfig, PPOTrainer

    checkpoint, tokenizer, peft = _common(config)
    budget, optimization = config["compute_budget"], config["optimization"]
    problems = generate_problems_v2c(config["task"]["train_problem_seed"], budget["rollout_groups"])
    expanded = [problem for problem in problems for _ in range(budget["generations_per_group"])]
    dataset = _dataset(expanded, tokenizer).remove_columns(["prompt", "problem_id"])
    policy = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.float16)
    value_model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=1, dtype=torch.float16)
    for parameter in value_model.parameters():
        parameter.requires_grad = False
    for parameter in value_model.score.parameters():
        parameter.requires_grad = True
    reward_model = V2RuleRewardModel(tokenizer, problems, config["reward"])
    args = PPOConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=budget["generations_per_group"],
        total_episodes=len(expanded),
        num_mini_batches=1,
        num_ppo_epochs=budget["updates_per_rollout_group"],
        response_length=budget["max_completion_tokens"],
        learning_rate=optimization["learning_rate"],
        adam_beta1=optimization["adam_beta1"],
        adam_beta2=optimization["adam_beta2"],
        max_grad_norm=optimization["max_grad_norm"],
        cliprange=optimization["clip_epsilon"],
        cliprange_value=optimization["ppo_value_clip"],
        vf_coef=optimization["ppo_value_coefficient"],
        kl_coef=optimization["kl_coefficient"],
        temperature=config["generation"]["temperature"],
        fp16=True,
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        seed=seed,
        num_sample_generations=0,
    )
    set_seed(seed)
    torch.cuda.reset_peak_memory_stats()
    trainer = PPOTrainer(
        args=args,
        processing_class=tokenizer,
        model=policy,
        ref_model=None,
        reward_model=reward_model,
        train_dataset=dataset,
        value_model=value_model,
        peft_config=peft,
    )
    started = time.perf_counter()
    trainer.train()
    return _save_run(output_dir, trainer, started, "ppo", seed, config)


@torch.inference_mode()
def evaluate_v2(config: dict, adapter_dir: Path | None) -> list[dict]:
    checkpoint = config["model"]["sft_checkpoint"]
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.float16).cuda()
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    problems = generate_problems_v2c(
        config["task"]["evaluation_problem_seed"], config["task"]["evaluation_examples"], id_offset=3_000_000
    )
    rows = []
    for problem in problems:
        encoded = tokenizer(
            render_prompt(tokenizer, problem), return_tensors="pt", truncation=True, max_length=192
        ).to("cuda")
        generated = model.generate(
            **encoded,
            max_new_tokens=config["compute_budget"]["max_completion_tokens"],
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        completion_ids = generated[0, encoded.input_ids.shape[1] :]
        completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
        reward, metadata = score_completion_v2c(completion, problem, config["reward"])
        rows.append(
            {
                "problem_id": problem.problem_id,
                "gold": problem.answer,
                "completion": completion,
                "reward": reward,
                "numeric_prediction": parse_final_integer(completion),
                "tagged_prediction": parse_tagged_answer(completion),
                **metadata,
            }
        )
    return rows
