"""
Process Reward Model (PRM) Training.

Training objective
------------------
Binary cross-entropy at every step-boundary position in the sequence.
Only positions where ``step_labels != -1`` contribute to the loss.

    L_PRM  =  −mean_{steps} [ y_t · log σ(z_t) + (1−y_t) · log(1−σ(z_t)) ]

where z_t is the model's logit at step boundary t, and y_t ∈ {0, 1} is the
ground-truth correctness label for that step.

The density advantage
---------------------
An ORM provides one gradient signal per training example (correct/incorrect final answer).
A PRM provides K signals per example (one per step).  For a 5-step problem, the PRM
has 5× the training signal density.  In practice this leads to faster convergence and
better calibration on individual steps.

Evaluation
----------
Beyond standard metrics (loss, accuracy), we evaluate:
  - step_accuracy_correct_solns: accuracy on steps from CORRECT solutions
    (should be near 1.0 — the model should recognise valid steps)
  - step_accuracy_wrong_solns: accuracy on steps from PERTURBED solutions
    (should be near 1.0 — the model should catch errors)
  - These can diverge: the model may be good at confirming correct steps
    but poor at detecting subtle errors, or vice versa.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm.auto import tqdm

from src.models.process_reward_model import GPT2ProcessRewardModel
from src.data.gsm8k import PRMDataset, ORMDataset


@dataclass
class PRMConfig:
    sft_checkpoint: str = "checkpoints/sft"
    output_dir: str = "checkpoints/prm"
    aggregation_mode: str = "mean"    # "mean" | "min" | "sum"
    num_epochs: int = 3
    batch_size: int = 8
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_length: int = 512
    num_train_samples: Optional[int] = 5_000
    num_eval_samples: Optional[int] = 500
    fp16: bool = True
    log_every: int = 50


@dataclass
class ORMConfig:
    sft_checkpoint: str = "checkpoints/sft"
    output_dir: str = "checkpoints/orm"
    num_epochs: int = 3
    batch_size: int = 8
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_length: int = 512
    num_train_samples: Optional[int] = 5_000
    num_eval_samples: Optional[int] = 500
    fp16: bool = True
    log_every: int = 50


def train_prm(cfg: PRMConfig) -> None:
    """Train a Process Reward Model on GSM8K step-level annotations."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(cfg.sft_checkpoint)
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2ProcessRewardModel.from_sft_checkpoint(
        cfg.sft_checkpoint,
        aggregation_mode=cfg.aggregation_mode,
        sep_token_id=tokenizer.eos_token_id,
    ).to(device)

    train_ds = PRMDataset("train", tokenizer, cfg.max_length, cfg.num_train_samples)
    eval_ds  = PRMDataset("test",  tokenizer, cfg.max_length, cfg.num_eval_samples)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2)
    eval_loader  = DataLoader(eval_ds,  batch_size=cfg.batch_size, shuffle=False, num_workers=2)

    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    total_steps = len(train_loader) * cfg.num_epochs // cfg.gradient_accumulation_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(total_steps * cfg.warmup_ratio), total_steps
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.fp16)
    os.makedirs(cfg.output_dir, exist_ok=True)
    optimizer.zero_grad()

    for epoch in range(cfg.num_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"PRM Epoch {epoch+1}/{cfg.num_epochs}")
        for step, batch in enumerate(pbar):
            ids    = batch["input_ids"].to(device)
            mask   = batch["attention_mask"].to(device)
            labels = batch["step_labels"].to(device)

            with torch.cuda.amp.autocast(enabled=cfg.fp16):
                out = model(ids, mask)
                loss, acc = model.compute_loss(out.step_logits, labels)
                loss = loss / cfg.gradient_accumulation_steps

            scaler.scale(loss).backward()
            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            if step % cfg.log_every == 0:
                pbar.set_postfix(loss=f"{loss.item()*cfg.gradient_accumulation_steps:.4f}",
                                 acc=f"{acc.item():.3f}")

        # Eval
        model.eval()
        total_loss, total_acc, n = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in eval_loader:
                ids    = batch["input_ids"].to(device)
                mask   = batch["attention_mask"].to(device)
                labels = batch["step_labels"].to(device)
                out    = model(ids, mask)
                l, a   = model.compute_loss(out.step_logits, labels)
                bs = ids.size(0)
                total_loss += l.item() * bs; total_acc += a.item() * bs; n += bs
        print(f"Epoch {epoch+1} | eval loss={total_loss/n:.4f} | step acc={total_acc/n:.4f}")

    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print(f"PRM saved to {cfg.output_dir}")


def train_orm(cfg: ORMConfig) -> None:
    """Train an Outcome Reward Model on GSM8K (final answer only).

    The ORM uses the same GPT2RewardModel architecture as the hh-rlhf reward model
    but trained on a binary (correct solution / perturbed solution) classification task.
    This gives a fair architectural comparison with the PRM.
    """
    import torch.nn.functional as F
    from src.models.reward_model import GPT2RewardModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(cfg.sft_checkpoint)
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2RewardModel.from_sft_checkpoint(cfg.sft_checkpoint).to(device)

    train_ds = ORMDataset("train", tokenizer, cfg.max_length, cfg.num_train_samples)
    eval_ds  = ORMDataset("test",  tokenizer, cfg.max_length, cfg.num_eval_samples)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2)
    eval_loader  = DataLoader(eval_ds,  batch_size=cfg.batch_size, shuffle=False, num_workers=2)

    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    total_steps = len(train_loader) * cfg.num_epochs // cfg.gradient_accumulation_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(total_steps * cfg.warmup_ratio), total_steps
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.fp16)
    os.makedirs(cfg.output_dir, exist_ok=True)
    optimizer.zero_grad()

    for epoch in range(cfg.num_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"ORM Epoch {epoch+1}/{cfg.num_epochs}")
        for step, batch in enumerate(pbar):
            ids    = batch["input_ids"].to(device)
            mask   = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast(enabled=cfg.fp16):
                rewards = model(ids, mask).rewards
                loss    = F.binary_cross_entropy_with_logits(rewards, labels)
                acc     = ((rewards > 0) == labels.bool()).float().mean()
                loss    = loss / cfg.gradient_accumulation_steps

            scaler.scale(loss).backward()
            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            if step % cfg.log_every == 0:
                pbar.set_postfix(loss=f"{loss.item()*cfg.gradient_accumulation_steps:.4f}",
                                 acc=f"{acc.item():.3f}")

        # Eval
        model.eval()
        total_loss, total_acc, n = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in eval_loader:
                ids    = batch["input_ids"].to(device)
                mask   = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                rewards = model(ids, mask).rewards
                l = F.binary_cross_entropy_with_logits(rewards, labels)
                a = ((rewards > 0) == labels.bool()).float().mean()
                bs = ids.size(0)
                total_loss += l.item() * bs; total_acc += a.item() * bs; n += bs
        print(f"Epoch {epoch+1} | eval loss={total_loss/n:.4f} | acc={total_acc/n:.4f}")

    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print(f"ORM saved to {cfg.output_dir}")
