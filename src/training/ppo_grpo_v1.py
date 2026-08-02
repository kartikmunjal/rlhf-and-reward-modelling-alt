"""TRL-backed, matched-budget PPO/GRPO study utilities."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig
from torch import nn
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from src.evaluation.arithmetic_rl import ID_RE, answer_map, extract_answer, generate_problems, score_completion


class _TokenBackbone(nn.Module):
    """Expose token IDs as a one-channel hidden state for deterministic scoring."""

    def forward(self, input_ids, **kwargs):  # noqa: ANN001
        _ = kwargs
        hidden = input_ids.unsqueeze(-1).to(torch.float32)
        return SimpleNamespace(hidden_states=(hidden,))


class VerifiableRewardModel(nn.Module):
    """TRL PPO reward-model contract backed by the frozen arithmetic verifier."""

    base_model_prefix = "backbone"

    def __init__(self, tokenizer, answers: dict[int, int], reward_config: dict):
        super().__init__()
        self.backbone = _TokenBackbone()
        self.tokenizer = tokenizer
        self.answers = answers
        self.reward_config = reward_config
        self.register_buffer("anchor", torch.zeros(1), persistent=False)

    def score(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ids = hidden_states[..., 0].round().to(torch.long)
        texts = self.tokenizer.batch_decode(ids, skip_special_tokens=False)
        rows = []
        for text in texts:
            match = ID_RE.search(text)
            answer = self.answers.get(int(match.group(1))) if match else None
            completion = text.rsplit("<|im_start|>assistant\n", 1)[-1]
            value = score_completion(completion, answer, self.reward_config) if answer is not None else -1.0
            rows.append(torch.full((ids.shape[1], 1), value, device=ids.device))
        return torch.stack(rows)


def _dataset(problems, tokenizer) -> Dataset:  # noqa: ANN001
    rows = []
    for problem in problems:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": problem.prompt}], tokenize=False, add_generation_prompt=True
        )
        rows.append({"prompt": text, "answer": str(problem.answer), "problem_id": problem.problem_id})
    dataset = Dataset.from_list(rows)

    def tokenize(row):
        return tokenizer(row["prompt"], truncation=True, max_length=192, padding=False)

    return dataset.map(tokenize)


def _load_common(config: dict):
    model_cfg = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["base_model"], revision=model_cfg["revision"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    peft_config = LoraConfig(
        r=model_cfg["lora_rank"],
        lora_alpha=model_cfg["lora_alpha"],
        lora_dropout=model_cfg["lora_dropout"],
        target_modules=model_cfg["lora_targets"],
        task_type="CAUSAL_LM",
    )
    return tokenizer, peft_config


def _write_run_artifacts(output_dir: Path, trainer, started: float, method: str, seed: int) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    history = list(trainer.state.log_history)
    peak = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else -1.0
    manifest = {
        "method": method,
        "seed": seed,
        "training_seconds": time.perf_counter() - started,
        "peak_gpu_memory_mb": peak,
        "log_history_rows": len(history),
        "resolved_model_revision": getattr(trainer.model.config, "_commit_hash", None),
        "torch_version": torch.__version__,
    }
    (output_dir / "trajectory.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def train_grpo(config: dict, seed: int, output_dir: Path, smoke: bool = False) -> dict:
    from trl import GRPOConfig, GRPOTrainer

    tokenizer, peft_config = _load_common(config)
    budget = config["compute_budget"]
    problems = generate_problems(config["task"]["train_problem_seed"], budget["rollout_groups"])
    dataset = _dataset(problems, tokenizer)
    rewards_cfg = config["reward"]

    def verifier(completions, answer, **kwargs):  # noqa: ANN001
        _ = kwargs
        return [score_completion(text, int(gold), rewards_cfg) for text, gold in zip(completions, answer)]

    steps = 2 if smoke else budget["optimizer_steps"]
    args = GRPOConfig(
        output_dir=str(output_dir),
        max_steps=steps,
        per_device_train_batch_size=2,
        learning_rate=config["optimization"]["learning_rate"],
        adam_beta1=config["optimization"]["adam_beta1"],
        adam_beta2=config["optimization"]["adam_beta2"],
        max_grad_norm=config["optimization"]["max_grad_norm"],
        num_generations=budget["generations_per_group"],
        num_iterations=budget["updates_per_rollout_group"],
        steps_per_generation=budget["updates_per_rollout_group"],
        max_completion_length=budget["max_completion_tokens"],
        beta=config["optimization"]["kl_coefficient"],
        epsilon=config["optimization"]["clip_epsilon"],
        temperature=config["generation"]["temperature"],
        top_p=config["generation"]["top_p"],
        fp16=True,
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        seed=seed,
    )
    torch.cuda.reset_peak_memory_stats()
    trainer = GRPOTrainer(
        model=config["model"]["base_model"],
        reward_funcs=verifier,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    started = time.perf_counter()
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(output_dir)
    return _write_run_artifacts(output_dir, trainer, started, "grpo", seed)


def train_ppo(config: dict, seed: int, output_dir: Path, smoke: bool = False) -> dict:
    from trl.experimental.ppo import PPOConfig, PPOTrainer

    tokenizer, peft_config = _load_common(config)
    budget = config["compute_budget"]
    groups = 1 if smoke else budget["rollout_groups"]
    problems = generate_problems(config["task"]["train_problem_seed"], groups)
    # Four entries per group give PPO the same number of sampled completions.
    expanded = [problem for problem in problems for _ in range(budget["generations_per_group"])]
    dataset = _dataset(expanded, tokenizer)
    dataset = dataset.remove_columns(["prompt", "answer", "problem_id"])
    model_name = config["model"]["base_model"]
    policy = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    value_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=1, torch_dtype=torch.float16
    )
    for parameter in value_model.parameters():
        parameter.requires_grad = False
    for parameter in value_model.score.parameters():
        parameter.requires_grad = True
    reward_model = VerifiableRewardModel(tokenizer, answer_map(problems), config["reward"])
    total_episodes = len(expanded)
    args = PPOConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=budget["generations_per_group"],
        total_episodes=total_episodes,
        num_mini_batches=1,
        num_ppo_epochs=budget["updates_per_rollout_group"],
        response_length=budget["max_completion_tokens"],
        learning_rate=config["optimization"]["learning_rate"],
        adam_beta1=config["optimization"]["adam_beta1"],
        adam_beta2=config["optimization"]["adam_beta2"],
        max_grad_norm=config["optimization"]["max_grad_norm"],
        cliprange=config["optimization"]["clip_epsilon"],
        cliprange_value=config["optimization"]["ppo_value_clip"],
        vf_coef=config["optimization"]["ppo_value_coefficient"],
        kl_coef=config["optimization"]["kl_coefficient"],
        temperature=config["generation"]["temperature"],
        fp16=True,
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        seed=seed,
        num_sample_generations=0,
    )
    torch.cuda.reset_peak_memory_stats()
    trainer = PPOTrainer(
        args=args,
        processing_class=tokenizer,
        model=policy,
        ref_model=None,
        reward_model=reward_model,
        train_dataset=dataset,
        value_model=value_model,
        peft_config=peft_config,
    )
    started = time.perf_counter()
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(output_dir)
    return _write_run_artifacts(output_dir, trainer, started, "ppo", seed)


@torch.inference_mode()
def evaluate_adapter(config: dict, adapter_dir: Path) -> list[dict]:
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    tokenizer.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(
        config["model"]["base_model"], torch_dtype=torch.float16, device_map="cuda"
    )
    model = PeftModel.from_pretrained(base, adapter_dir).eval()
    problems = generate_problems(
        config["task"]["evaluation_problem_seed"],
        config["task"]["evaluation_examples"],
        id_offset=1_000_000,
    )
    rows = []
    for problem in problems:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": problem.prompt}], tokenize=False, add_generation_prompt=True
        )
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=192).to("cuda")
        generated = model.generate(**encoded, max_new_tokens=64, do_sample=False)
        completion = tokenizer.decode(generated[0, encoded.input_ids.shape[1] :], skip_special_tokens=True)
        prediction = extract_answer(completion)
        rows.append(
            {
                "problem_id": problem.problem_id,
                "gold": problem.answer,
                "prediction": prediction,
                "correct": prediction == problem.answer,
                "completion": completion,
            }
        )
    return rows


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
