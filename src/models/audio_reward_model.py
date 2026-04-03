"""
Audio Reward Model for TTS quality preference learning.

Two implementations with a shared interface:

1. AudioFeatureRewardModel (default, CPU-runnable)
   Input: 7-dimensional acoustic feature vector
   Architecture: 3-layer MLP with LayerNorm
   Suitable for: rapid iteration, no GPU required

2. Wav2Vec2RewardModel (high-fidelity, GPU recommended)
   Input: raw audio waveform → facebook/wav2vec2-base frozen encoder → pooled embedding
   Architecture: frozen Wav2Vec2 + trainable reward head
   Suitable for: best accuracy, learns richer audio representations

Both models expose:
  forward(features_or_audio) → scalar reward
  preference_loss(chosen, rejected) → Bradley-Terry loss
  pairwise_accuracy(chosen_rewards, rejected_rewards) → float

The interface mirrors GPT2RewardModel from src/models/reward_model.py,
so the same training loop works for both text and audio preferences.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Shared Bradley-Terry loss ──────────────────────────────────────────────────

def audio_preference_loss(
    chosen_rewards: torch.Tensor,
    rejected_rewards: torch.Tensor,
) -> torch.Tensor:
    """
    Bradley-Terry preference loss for audio quality pairs.

    L = -E[log σ(r_chosen - r_rejected)]

    Identical in form to the text reward model loss — the only difference is
    that the rewards are scalars derived from audio features / embeddings rather
    than from token-sequence hidden states.
    """
    return -F.logsigmoid(chosen_rewards - rejected_rewards).mean()


def pairwise_accuracy(
    chosen_rewards: torch.Tensor,
    rejected_rewards: torch.Tensor,
) -> float:
    """Fraction of pairs where chosen reward > rejected reward."""
    return float((chosen_rewards > rejected_rewards).float().mean().item())


# ── Model 1: Feature-based reward model ──────────────────────────────────────

class AudioFeatureRewardModel(nn.Module):
    """
    Lightweight reward model on 7-dimensional acoustic feature vectors.

    Feature dimensions (from src/data/tts_preferences.py):
      0: pitch_variance      — prosody naturalness
      1: voiced_fraction     — speech content ratio
      2: hnr                 — harmonic-to-noise ratio (voice quality)
      3: silence_fraction    — pacing proxy
      4: spectral_centroid   — brightness / clarity
      5: mfcc_stability      — voice consistency
      6: energy_dynamics     — expressiveness

    Architecture:
      Linear(7 → 64) → LayerNorm → GELU
      Linear(64 → 32) → LayerNorm → GELU
      Linear(32 → 1, bias=False)

    The final layer has no bias so the model learns relative quality
    (only differences matter in Bradley-Terry training).
    """

    FEATURE_DIM = 7

    def __init__(
        self,
        feature_dim: int = 7,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim

        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1, bias=False),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (batch, feature_dim) float tensor
        Returns:
            rewards: (batch,) scalar rewards
        """
        return self.net(features).squeeze(-1)

    def preference_loss(
        self,
        chosen_features: torch.Tensor,
        rejected_features: torch.Tensor,
    ) -> torch.Tensor:
        chosen_rewards = self.forward(chosen_features)
        rejected_rewards = self.forward(rejected_features)
        return audio_preference_loss(chosen_rewards, rejected_rewards)

    def pairwise_accuracy(
        self,
        chosen_features: torch.Tensor,
        rejected_features: torch.Tensor,
    ) -> float:
        with torch.no_grad():
            chosen_rewards = self.forward(chosen_features)
            rejected_rewards = self.forward(rejected_features)
        return pairwise_accuracy(chosen_rewards, rejected_rewards)

    def extra_repr(self) -> str:
        return f"feature_dim={self.feature_dim}"


# ── Model 2: Wav2Vec2-based reward model ──────────────────────────────────────

class Wav2Vec2RewardModel(nn.Module):
    """
    High-fidelity audio reward model using frozen Wav2Vec2 encoder + trainable head.

    Architecture:
      Frozen Wav2Vec2-base encoder (768-dim hidden states)
      → Mean pool over time dimension
      → Linear(768 → 256) → GELU
      → Linear(256 → 1, bias=False)

    Only the reward head is trained; Wav2Vec2 weights are frozen.
    This gives rich audio representations without fine-tuning the large encoder.

    GPU recommended for training (Wav2Vec2-base ≈ 90M parameters).

    Usage:
        model = Wav2Vec2RewardModel()
        reward = model(audio_values)  # (batch, seq_len) float32 at 16 kHz
    """

    MODEL_ID = "facebook/wav2vec2-base"
    INPUT_SAMPLE_RATE = 16000  # Wav2Vec2 expects 16 kHz

    def __init__(
        self,
        model_id: str = "facebook/wav2vec2-base",
        freeze_encoder: bool = True,
        head_hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        try:
            from transformers import Wav2Vec2Model
        except ImportError:
            raise ImportError("transformers is required: pip install transformers")

        self.encoder = Wav2Vec2Model.from_pretrained(model_id)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        encoder_dim = self.encoder.config.hidden_size  # 768 for wav2vec2-base

        self.reward_head = nn.Sequential(
            nn.Linear(encoder_dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, 1, bias=False),
        )

    def _encode(self, audio_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio_values: (batch, time) float32 waveform at 16 kHz
        Returns:
            pooled: (batch, encoder_dim)
        """
        outputs = self.encoder(audio_values)
        hidden_states = outputs.last_hidden_state   # (batch, T, D)
        pooled = hidden_states.mean(dim=1)          # mean pool over time
        return pooled

    def forward(self, audio_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio_values: (batch, time) float32
        Returns:
            rewards: (batch,) scalar
        """
        pooled = self._encode(audio_values)
        return self.reward_head(pooled).squeeze(-1)

    def preference_loss(
        self,
        chosen_audio: torch.Tensor,
        rejected_audio: torch.Tensor,
    ) -> torch.Tensor:
        chosen_rewards = self.forward(chosen_audio)
        rejected_rewards = self.forward(rejected_audio)
        return audio_preference_loss(chosen_rewards, rejected_rewards)

    def pairwise_accuracy(
        self,
        chosen_audio: torch.Tensor,
        rejected_audio: torch.Tensor,
    ) -> float:
        with torch.no_grad():
            chosen_rewards = self.forward(chosen_audio)
            rejected_rewards = self.forward(rejected_audio)
        return pairwise_accuracy(chosen_rewards, rejected_rewards)

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}


# ── MOS proxy predictor (wrapper around trained RM) ──────────────────────────

class MOSProxyPredictor:
    """
    Wraps a trained AudioFeatureRewardModel to produce interpretable MOS-like scores.

    The raw reward is an unconstrained scalar. This wrapper calibrates it to
    the [1, 5] MOS scale using a linear mapping fitted on a held-out validation set.

    Usage:
        mos = MOSProxyPredictor(trained_model, reward_min=-0.5, reward_max=0.8)
        score = mos.predict(features)  # returns float in [1, 5]
    """

    MOS_MIN = 1.0
    MOS_MAX = 5.0

    def __init__(
        self,
        model: AudioFeatureRewardModel,
        reward_min: float = -1.0,
        reward_max: float = 1.0,
    ) -> None:
        self.model = model
        self.reward_min = reward_min
        self.reward_max = reward_max

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """Map raw reward to [1, 5] MOS scale."""
        with torch.no_grad():
            reward = self.model(features)
        normalised = (reward - self.reward_min) / (self.reward_max - self.reward_min + 1e-9)
        mos = self.MOS_MIN + normalised.clamp(0.0, 1.0) * (self.MOS_MAX - self.MOS_MIN)
        return mos

    def calibrate(self, rewards: torch.Tensor) -> None:
        """Update min/max from a distribution of rewards."""
        self.reward_min = float(rewards.min().item())
        self.reward_max = float(rewards.max().item())
