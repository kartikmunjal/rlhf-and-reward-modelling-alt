"""
Training loop for the audio reward model.

Directly mirrors src/training/reward.py but operates on acoustic feature vectors
(or raw audio via Wav2Vec2) rather than token-sequence embeddings.

The pairwise Bradley-Terry loss, evaluation loop, and checkpoint structure are
identical — the swap is input modality (audio features → reward) vs.
(text tokens → reward).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class TTSRewardConfig:
    jsonl_path: str = "data/tts_preferences.jsonl"
    output_dir: str = "checkpoints/tts_reward"
    model_type: str = "feature"          # "feature" (CPU) or "wav2vec2" (GPU)
    # Training
    epochs: int = 20
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    # Data
    min_delta: float = 0.02             # discard near-tie pairs
    train_split: float = 0.85
    # Feature RM
    hidden_dim: int = 64
    dropout: float = 0.1
    # Wav2Vec2 RM (if model_type="wav2vec2")
    wav2vec2_model_id: str = "facebook/wav2vec2-base"
    freeze_encoder: bool = True
    max_audio_len: int = 80000          # ~5s at 16 kHz


# ── Torch dataset wrapper ─────────────────────────────────────────────────────

class TTSPreferenceTorchDataset(Dataset):
    """Wraps TTSPreferenceDataset for DataLoader compatibility."""

    def __init__(self, data: List[Dict]) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        return {
            "chosen_features": torch.tensor(item["chosen_features"], dtype=torch.float32),
            "rejected_features": torch.tensor(item["rejected_features"], dtype=torch.float32),
            "chosen_score": torch.tensor(item["chosen_score"], dtype=torch.float32),
            "rejected_score": torch.tensor(item["rejected_score"], dtype=torch.float32),
        }


# ── Training ──────────────────────────────────────────────────────────────────

def train_tts_reward_model(cfg: TTSRewardConfig) -> Dict:
    """
    Train an audio reward model on TTS preference pairs.

    Returns
    -------
    dict with keys: best_pairwise_accuracy, final_loss, train_history, output_dir
    """
    from src.data.tts_preferences import TTSPreferenceDataset
    from src.models.audio_reward_model import (
        AudioFeatureRewardModel,
        Wav2Vec2RewardModel,
        audio_preference_loss,
        pairwise_accuracy,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\n── TTS Reward Model Training ──")
    print(f"  data:    {cfg.jsonl_path}")
    print(f"  model:   {cfg.model_type}  epochs={cfg.epochs}  lr={cfg.learning_rate}")

    base_dataset = TTSPreferenceDataset(cfg.jsonl_path, min_delta=cfg.min_delta)

    # Collect all items as plain dicts
    all_items = [base_dataset[i] for i in range(len(base_dataset))]
    n = len(all_items)
    split = int(n * cfg.train_split)

    indices = list(range(n))
    np.random.shuffle(indices)
    train_items = [all_items[i] for i in indices[:split]]
    val_items   = [all_items[i] for i in indices[split:]]

    train_ds = TTSPreferenceTorchDataset(train_items)
    val_ds   = TTSPreferenceTorchDataset(val_items)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False)

    print(f"  train: {len(train_ds)} pairs   val: {len(val_ds)} pairs")

    # ── Build model ───────────────────────────────────────────────────────────
    if cfg.model_type == "feature":
        feature_dim = base_dataset.feature_dim()
        model = AudioFeatureRewardModel(
            feature_dim=feature_dim,
            hidden_dim=cfg.hidden_dim,
            dropout=cfg.dropout,
        ).to(device)
        print(f"  AudioFeatureRewardModel: {feature_dim}-dim input, {cfg.hidden_dim}-dim hidden")
    else:
        model = Wav2Vec2RewardModel(
            model_id=cfg.wav2vec2_model_id,
            freeze_encoder=cfg.freeze_encoder,
        ).to(device)
        counts = model.count_parameters()
        print(f"  Wav2Vec2RewardModel: {counts['trainable']:,} trainable / {counts['total']:,} total params")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    history = []
    best_val_acc = 0.0
    best_epoch = 0
    os.makedirs(cfg.output_dir, exist_ok=True)

    for epoch in range(cfg.epochs):
        # Train
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            chosen_feats   = batch["chosen_features"].to(device)
            rejected_feats = batch["rejected_features"].to(device)

            loss = model.preference_loss(chosen_feats, rejected_feats)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()
        train_loss /= max(len(train_loader), 1)

        # Validate
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        n_batches = 0
        for batch in val_loader:
            chosen_feats   = batch["chosen_features"].to(device)
            rejected_feats = batch["rejected_features"].to(device)
            with torch.no_grad():
                loss = model.preference_loss(chosen_feats, rejected_feats)
                acc  = model.pairwise_accuracy(chosen_feats, rejected_feats)
            val_loss += loss.item()
            val_acc  += acc
            n_batches += 1

        val_loss /= max(n_batches, 1)
        val_acc  /= max(n_batches, 1)

        history.append({"epoch": epoch + 1, "train_loss": train_loss,
                        "val_loss": val_loss, "val_pairwise_acc": val_acc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(cfg.output_dir, "best.pt"))

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  epoch {epoch+1:3d}/{cfg.epochs}  "
                  f"train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  "
                  f"val_acc={val_acc:.4f}")

    # Save final checkpoint and config
    torch.save(model.state_dict(), os.path.join(cfg.output_dir, "final.pt"))
    with open(os.path.join(cfg.output_dir, "config.json"), "w") as f:
        json.dump(cfg.__dict__, f, indent=2)
    with open(os.path.join(cfg.output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest val pairwise accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
    print(f"Checkpoints saved → {cfg.output_dir}")

    return {
        "best_pairwise_accuracy": best_val_acc,
        "best_epoch": best_epoch,
        "final_loss": history[-1]["val_loss"] if history else None,
        "train_history": history,
        "output_dir": cfg.output_dir,
    }


# ── Evaluation helpers ────────────────────────────────────────────────────────

def evaluate_reward_model(
    model_dir: str,
    jsonl_path: str,
    model_type: str = "feature",
    device: str = "cpu",
) -> Dict:
    """Load a trained RM checkpoint and evaluate on a preference JSONL."""
    from src.data.tts_preferences import TTSPreferenceDataset
    from src.models.audio_reward_model import AudioFeatureRewardModel

    dataset = TTSPreferenceDataset(jsonl_path)
    feature_dim = dataset.feature_dim()

    if model_type == "feature":
        model = AudioFeatureRewardModel(feature_dim=feature_dim)
        model.load_state_dict(
            torch.load(os.path.join(model_dir, "best.pt"), map_location=device)
        )
    else:
        raise NotImplementedError("Wav2Vec2 eval not yet implemented here")

    model.eval()
    all_items = [dataset[i] for i in range(len(dataset))]
    ds = TTSPreferenceTorchDataset(all_items)
    loader = DataLoader(ds, batch_size=32, shuffle=False)

    total_acc = 0.0
    total_margin = 0.0
    n_batches = 0

    for batch in loader:
        chosen   = batch["chosen_features"].to(device)
        rejected = batch["rejected_features"].to(device)
        with torch.no_grad():
            r_c = model(chosen)
            r_r = model(rejected)
        total_acc    += float((r_c > r_r).float().mean().item())
        total_margin += float((r_c - r_r).mean().item())
        n_batches += 1

    return {
        "pairwise_accuracy": total_acc / max(n_batches, 1),
        "mean_margin": total_margin / max(n_batches, 1),
        "n_pairs": len(dataset),
    }
