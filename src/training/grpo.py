"""Stage 3c — Group Relative Policy Optimization (GRPO).

GRPO removes PPO's learned value baseline entirely. For each prompt we sample
G completions, score them with the reward model, and normalize each reward by
the group mean and standard deviation:

    A_i = (r_i - mean(r_1...r_G)) / std(r_1...r_G)

TRL's GRPOTrainer implements the online rollout loop and clipped update rule,
so this module mirrors the existing PPO wrapper while exposing the GRPO-specific
hyperparameters that matter for experimentation.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

import torch

from src.data.preprocessing import extract_prompt_and_response


@dataclass
class GRPOTrainingConfig:
    sft_checkpoint: str = "checkpoints/sft"
    reward_checkpoint: str = "checkpoints/reward_model"
    output_dir: str = "checkpoints/grpo"
    learning_rate: float = 1e-6
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    num_train_epochs: float = 1.0
    max_steps: int = -1
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    max_prompt_length: int = 256
    max_completion_length: int = 128
    num_train_samples: Optional[int] = 5_000
    logging_steps: int = 10
    fp16: bool = False
    bf16: bool = True
    num_generations: int = 4
    beta: float = 0.1
    epsilon: float = 0.2
    report_to: str = "none"


def _build_prompt_dataset(
    checkpoint: str,
    num_samples: Optional[int],
    max_prompt_length: int,
) -> object:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = tokenizer.eos_token

    raw = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base", split="train")
    if num_samples and num_samples > 0:
        raw = raw.select(range(min(num_samples, len(raw))))

    def to_prompt(example):
        prompt, _ = extract_prompt_and_response(example["chosen"])
        enc = tokenizer(
            prompt,
            max_length=max_prompt_length,
            truncation=True,
            padding=False,
        )
        return {"prompt": prompt, "prompt_length": len(enc["input_ids"])}

    return raw.map(to_prompt, remove_columns=raw.column_names)


def build_grpo_reward_func(
    reward_checkpoint: str,
    tokenizer,
    device: torch.device,
):
    from src.models.reward_model import GPT2RewardModel

    reward_model = GPT2RewardModel.from_pretrained(reward_checkpoint).to(device)
    reward_model.eval()

    @torch.no_grad()
    def reward_func(prompts, completions, **kwargs):
        _ = kwargs
        if completions and isinstance(completions[0], list):
            flat_completions = [completion[0]["content"] for completion in completions]
        else:
            flat_completions = [str(completion) for completion in completions]

        texts = [f"{prompt}{completion}" for prompt, completion in zip(prompts, flat_completions)]
        enc = tokenizer(
            texts,
            max_length=512,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)
        rewards = reward_model(**enc).rewards
        return rewards.detach().cpu().tolist()

    return reward_func


def train_grpo(cfg: GRPOTrainingConfig) -> dict[str, float]:
    """Train a policy with GRPO and return lightweight run stats."""
    from trl import GRPOConfig as TRLGRPOConfig, GRPOTrainer
    from transformers import AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(cfg.sft_checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    train_dataset = _build_prompt_dataset(
        cfg.sft_checkpoint,
        cfg.num_train_samples,
        cfg.max_prompt_length,
    )

    reward_func = build_grpo_reward_func(cfg.reward_checkpoint, tokenizer, device)

    training_args = TRLGRPOConfig(
        output_dir=cfg.output_dir,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        max_steps=cfg.max_steps,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        logging_steps=cfg.logging_steps,
        report_to=cfg.report_to,
        max_prompt_length=cfg.max_prompt_length,
        max_completion_length=cfg.max_completion_length,
        num_generations=cfg.num_generations,
        beta=cfg.beta,
        epsilon=cfg.epsilon,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
    )

    trainer = GRPOTrainer(
        model=cfg.sft_checkpoint,
        reward_funcs=reward_func,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    os.makedirs(cfg.output_dir, exist_ok=True)
    peak_memory_mb = None
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    trainer.train()
    wall_time = time.perf_counter() - t0

    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    if torch.cuda.is_available():
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    stats = {
        "training_time_seconds": round(wall_time, 2),
        "peak_memory_mb": round(peak_memory_mb, 2) if peak_memory_mb is not None else -1.0,
        "num_train_rows": float(len(train_dataset)),
        "num_generations": float(cfg.num_generations),
        "beta": cfg.beta,
        "epsilon": cfg.epsilon,
    }
    print(f"GRPO policy saved to {cfg.output_dir}")
    return stats
