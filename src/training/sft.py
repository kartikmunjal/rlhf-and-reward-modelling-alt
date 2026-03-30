"""
Stage 1 — Supervised Fine-Tuning (SFT).

Why SFT first?
--------------
A base language model trained purely on next-token prediction has no concept of
conversation format or helpful behavior.  SFT performs *behavioral cloning*: we
fine-tune on the human-preferred ("chosen") responses so the model learns the
right distribution *before* any RL signal is applied.

The SFT checkpoint also serves as the **frozen reference policy** (π_ref) during
PPO and DPO training.  The KL-penalty term in both algorithms measures drift from
this reference, which is what prevents reward hacking.

Implementation
--------------
We use HuggingFace's Trainer with a causal-LM objective.  Prompt tokens are masked
in the labels tensor (set to -100) so the loss gradient flows only through the
response tokens — the part we actually want to improve.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from transformers import (
    AutoTokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
)

from src.data.preprocessing import SFTDataset


@dataclass
class SFTConfig:
    model_name: str = "gpt2-medium"
    output_dir: str = "checkpoints/sft"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4      # effective batch = 4 * 4 = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    max_length: int = 512
    num_train_samples: Optional[int] = 10_000
    num_eval_samples: Optional[int] = 1_000
    logging_steps: int = 50
    eval_steps: int = 200
    save_steps: int = 500
    fp16: bool = True
    dataloader_num_workers: int = 2
    report_to: str = "none"                   # set to "wandb" for experiment tracking


def train_sft(cfg: SFTConfig) -> None:
    """Fine-tune a GPT-2 language model on chosen responses (Stage 1).

    After training the checkpoint at ``cfg.output_dir`` is used as:
      1. Initialisation point for the reward model (Stage 2)
      2. Frozen reference policy during PPO and DPO (Stage 3)
    """
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    # GPT-2 has no padding token by default; we reuse the EOS token
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = GPT2LMHeadModel.from_pretrained(cfg.model_name)
    # Tell GPT-2 which token is used for padding (affects attention mask handling)
    model.config.pad_token_id = tokenizer.pad_token_id

    train_dataset = SFTDataset(
        split="train",
        tokenizer=tokenizer,
        max_length=cfg.max_length,
        num_samples=cfg.num_train_samples,
    )
    eval_dataset = SFTDataset(
        split="test",
        tokenizer=tokenizer,
        max_length=cfg.max_length,
        num_samples=cfg.num_eval_samples,
    )

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        evaluation_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        logging_steps=cfg.logging_steps,
        fp16=cfg.fp16,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=cfg.dataloader_num_workers,
        report_to=cfg.report_to,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    print(f"Training SFT on {len(train_dataset):,} examples  →  {cfg.output_dir}")
    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print(f"SFT checkpoint saved to {cfg.output_dir}")
