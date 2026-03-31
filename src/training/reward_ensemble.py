"""
Reward Model Ensemble Training.

We train K reward models with identical architecture but different random seeds
(different reward-head initialisations).  Differences in seed propagate through
stochastic gradient descent, causing the models to find slightly different local
minima — enough to produce meaningfully different predictions in uncertain regions.

Why different seeds instead of different architectures?
    For a clean ablation, we want the ensemble to differ *only* in initialisation,
    not in capacity.  Architecture differences would confound the uncertainty
    interpretation: disagreement might reflect capacity rather than genuine ambiguity.

The trained ensemble is saved to:
    checkpoints/reward_ensemble/model_0/
    checkpoints/reward_ensemble/model_1/
    ...
    checkpoints/reward_ensemble/model_K-1/

Usage
-----
    from src.models.reward_ensemble import RewardEnsemble
    ensemble = RewardEnsemble.from_checkpoints([
        "checkpoints/reward_ensemble/model_0",
        "checkpoints/reward_ensemble/model_1",
        "checkpoints/reward_ensemble/model_2",
    ])
    r_mean, r_std = ensemble(input_ids, attention_mask)
    r_penalized  = ensemble.penalized_reward(input_ids, attention_mask)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm.auto import tqdm

from src.models.reward_model import GPT2RewardModel, preference_loss
from src.data.preprocessing import build_preference_dataloader


@dataclass
class EnsembleTrainingConfig:
    sft_checkpoint: str = "checkpoints/sft"
    output_dir: str = "checkpoints/reward_ensemble"
    k: int = 3                           # number of ensemble members
    # Training args (applied to each member independently)
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
    # Divergence: different head init seeds
    base_seed: int = 0                   # model_i uses seed base_seed + i


def _train_single(
    member_idx: int,
    cfg: EnsembleTrainingConfig,
    device: torch.device,
) -> str:
    """Train one ensemble member and return its checkpoint path."""
    seed = cfg.base_seed + member_idx
    torch.manual_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.sft_checkpoint)
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2RewardModel.from_sft_checkpoint(cfg.sft_checkpoint).to(device)

    # Re-initialise the reward head with this member's seed
    # (the backbone weights are the same for all members)
    torch.manual_seed(seed)
    torch.nn.init.normal_(model.reward_head.weight, std=0.02)

    train_loader = build_preference_dataloader(
        "train", tokenizer, batch_size=cfg.batch_size,
        max_length=cfg.max_length, num_samples=cfg.num_train_samples,
    )
    eval_loader = build_preference_dataloader(
        "test", tokenizer, batch_size=cfg.batch_size,
        max_length=cfg.max_length, num_samples=cfg.num_eval_samples,
    )

    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    total_steps = len(train_loader) * cfg.num_epochs // cfg.gradient_accumulation_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(total_steps * cfg.warmup_ratio), total_steps
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.fp16)

    out_dir = os.path.join(cfg.output_dir, f"model_{member_idx}")
    os.makedirs(out_dir, exist_ok=True)
    optimizer.zero_grad()

    for epoch in range(cfg.num_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Member {member_idx} | Epoch {epoch+1}")
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

        # Quick eval
        model.eval()
        total_acc, n = 0.0, 0
        with torch.no_grad():
            for batch in eval_loader:
                c_ids = batch["chosen_input_ids"].to(device)
                c_mask = batch["chosen_attention_mask"].to(device)
                r_ids = batch["rejected_input_ids"].to(device)
                r_mask = batch["rejected_attention_mask"].to(device)
                rw = model(c_ids, c_mask).rewards
                rl = model(r_ids, r_mask).rewards
                _, acc = preference_loss(rw, rl)
                total_acc += acc.item() * c_ids.size(0)
                n += c_ids.size(0)
        print(f"  Member {member_idx} | Epoch {epoch+1} eval acc: {total_acc/n:.4f}")

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    return out_dir


def train_reward_ensemble(cfg: EnsembleTrainingConfig) -> List[str]:
    """Train K reward models sequentially and return their checkpoint paths.

    Note: sequential training keeps memory usage predictable (one model at a time).
    If you have K GPUs, you could parallelise with multiprocessing — the models are
    independent.

    Returns
    -------
    List of checkpoint directory paths, one per ensemble member.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.output_dir, exist_ok=True)

    print(f"Training ensemble of {cfg.k} reward models")
    print(f"  Seeds: {cfg.base_seed} … {cfg.base_seed + cfg.k - 1}")
    print(f"  Output: {cfg.output_dir}/model_0 … model_{cfg.k-1}")
    print()

    checkpoints = []
    for i in range(cfg.k):
        print(f"{'='*50}")
        print(f"Training ensemble member {i+1}/{cfg.k}  (seed={cfg.base_seed + i})")
        print(f"{'='*50}")
        ckpt = _train_single(i, cfg, device)
        checkpoints.append(ckpt)
        print(f"Saved to {ckpt}\n")

    # Write a manifest so the ensemble can be loaded in one call
    manifest_path = os.path.join(cfg.output_dir, "manifest.txt")
    with open(manifest_path, "w") as f:
        f.write("\n".join(checkpoints) + "\n")
    print(f"Ensemble manifest: {manifest_path}")

    return checkpoints
