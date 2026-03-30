"""
Stage 2 — Reward Model Training.

Goal
----
Train a model r_φ(x, y) → ℝ that scores (prompt, response) pairs such that
higher scores correlate with human preference.

We use the **Bradley-Terry** pairwise model.  For each annotated pair (y_w, y_l)
sharing prompt x, the probability of the human preferring y_w is:

    P(y_w ≻ y_l | x)  =  σ( r_φ(x, y_w) − r_φ(x, y_l) )

Maximum-likelihood estimation over the dataset gives:

    L_RM  =  −E[ log σ( r_φ(x, y_w) − r_φ(x, y_l) ) ]

This is a binary classification loss on the *margin* between chosen and
rejected rewards; it pushes r(chosen) > r(rejected) without anchoring the
absolute reward scale.

Architecture
------------
GPT2RewardModel: GPT-2 transformer (initialised from the SFT checkpoint)
with a single ``Linear(n_embd, 1, bias=False)`` head pooling the last
non-padding token hidden state.

Practical notes
---------------
- We initialise from the SFT checkpoint (not raw GPT-2) because the SFT model
  already understands the Human/Assistant dialogue format.
- A cosine LR schedule with 5 % warmup prevents reward collapse on early batches.
- We log pairwise accuracy (fraction of pairs where r_w > r_l) in addition to
  loss — it is a more interpretable training signal (random baseline = 0.50).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm.auto import tqdm

from src.models.reward_model import GPT2RewardModel, preference_loss
from src.data.preprocessing import build_preference_dataloader


@dataclass
class RewardConfig:
    sft_checkpoint: str = "checkpoints/sft"     # initialise RM from here
    output_dir: str = "checkpoints/reward_model"
    num_epochs: int = 2
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_length: int = 512
    num_train_samples: Optional[int] = 10_000
    num_eval_samples: Optional[int] = 1_000
    fp16: bool = True
    log_every: int = 50


def _eval_pass(model: GPT2RewardModel, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch in loader:
            c_ids = batch["chosen_input_ids"].to(device)
            c_mask = batch["chosen_attention_mask"].to(device)
            r_ids = batch["rejected_input_ids"].to(device)
            r_mask = batch["rejected_attention_mask"].to(device)
            r_w = model(c_ids, c_mask).rewards
            r_l = model(r_ids, r_mask).rewards
            loss, acc = preference_loss(r_w, r_l)
            bs = c_ids.size(0)
            total_loss += loss.item() * bs
            total_acc += acc.item() * bs
            n += bs
    return {"eval_loss": total_loss / n, "eval_accuracy": total_acc / n}


def train_reward_model(cfg: RewardConfig) -> None:
    """Train the reward model on human preference pairs (Stage 2).

    The trained checkpoint is used by the PPO trainer (Stage 3a) to score
    rollout responses.  DPO (Stage 3b) does *not* use the reward model at
    training time — it is only used for evaluation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(cfg.sft_checkpoint)
    tokenizer.pad_token = tokenizer.eos_token

    # Initialise from SFT checkpoint for a better starting point
    model = GPT2RewardModel.from_sft_checkpoint(cfg.sft_checkpoint).to(device)

    train_loader = build_preference_dataloader(
        "train", tokenizer, batch_size=cfg.batch_size,
        max_length=cfg.max_length, num_samples=cfg.num_train_samples,
    )
    eval_loader = build_preference_dataloader(
        "test", tokenizer, batch_size=cfg.batch_size,
        max_length=cfg.max_length, num_samples=cfg.num_eval_samples,
    )

    optimizer = AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    total_steps = len(train_loader) * cfg.num_epochs // cfg.gradient_accumulation_steps
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.fp16)
    os.makedirs(cfg.output_dir, exist_ok=True)

    global_step = 0
    optimizer.zero_grad()

    for epoch in range(cfg.num_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.num_epochs}")
        for step, batch in enumerate(pbar):
            c_ids = batch["chosen_input_ids"].to(device)
            c_mask = batch["chosen_attention_mask"].to(device)
            r_ids = batch["rejected_input_ids"].to(device)
            r_mask = batch["rejected_attention_mask"].to(device)

            with torch.cuda.amp.autocast(enabled=cfg.fp16):
                r_w = model(c_ids, c_mask).rewards
                r_l = model(r_ids, r_mask).rewards
                loss, acc = preference_loss(r_w, r_l)
                loss = loss / cfg.gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            if global_step % cfg.log_every == 0:
                pbar.set_postfix(
                    loss=f"{loss.item() * cfg.gradient_accumulation_steps:.4f}",
                    acc=f"{acc.item():.3f}",
                )

        metrics = _eval_pass(model, eval_loader, device)
        print(
            f"Epoch {epoch+1} | eval_loss={metrics['eval_loss']:.4f} "
            f"| eval_acc={metrics['eval_accuracy']:.4f}"
        )
        model.train()

    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print(f"Reward model saved to {cfg.output_dir}")
