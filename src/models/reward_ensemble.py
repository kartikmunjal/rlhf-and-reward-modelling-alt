"""
Reward Model Ensemble — uncertainty-penalised reward scoring.

Motivation
----------
A single reward model is a *point estimate* of human preference.  During PPO,
the policy is optimised against this estimate, not against ground-truth human
preference.  If the estimate is wrong in some region, the policy will exploit it.

The Ensemble approach
---------------------
Train K reward models with different random seeds (different reward-head
initialisations).  During PPO, compute the reward for each rollout as:

    r_ensemble(x, y)  =  mean_k[ r_k(x, y) ]  −  λ · std_k[ r_k(x, y) ]

The penalisation term λ·σ shrinks the effective reward in regions where the
ensemble *disagrees* — precisely the regions where the reward models are
unreliable.  This directly attacks the over-optimisation problem:

    • Ensemble agrees  → low σ → policy is rewarded as normal
    • Ensemble disagrees → high σ → reward is penalised → policy avoids the region

Connection to Anthropic's work
------------------------------
Reward model ensembles are used in Anthropic's scalable oversight research as a
practical approximation to the true (unknown) reward function, with the disagreement
term acting as a conservative uncertainty estimate.

Reference: "Scaling Laws for Reward Model Overoptimization" (Gao et al., 2022)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from src.models.reward_model import GPT2RewardModel


class RewardEnsemble(nn.Module):
    """Ensemble of GPT2RewardModels with uncertainty-penalised scoring.

    Parameters
    ----------
    models:
        List of trained :class:`GPT2RewardModel` instances.
    uncertainty_penalty:
        λ — scales the standard-deviation penalty.  λ=0 recovers plain mean;
        λ=1 is aggressive; λ=0.5 is a reasonable starting point.
    """

    def __init__(
        self,
        models: List[GPT2RewardModel],
        uncertainty_penalty: float = 0.5,
    ) -> None:
        super().__init__()
        if len(models) < 2:
            raise ValueError("Ensemble requires at least 2 models.")
        self.models = nn.ModuleList(models)
        self.uncertainty_penalty = uncertainty_penalty

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute mean and std rewards across the ensemble.

        Returns
        -------
        mean_reward : (B,) — average reward across all K models
        std_reward  : (B,) — standard deviation across K models (uncertainty proxy)
        """
        all_rewards = torch.stack(
            [m(input_ids, attention_mask).rewards for m in self.models],
            dim=0,
        )  # (K, B)
        mean_reward = all_rewards.mean(dim=0)   # (B,)
        std_reward = all_rewards.std(dim=0)     # (B,)
        return mean_reward, std_reward

    def penalized_reward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        penalty_override: Optional[float] = None,
    ) -> torch.Tensor:
        """Return uncertainty-penalised scalar reward.

            r_penalized = mean − λ · std

        Parameters
        ----------
        penalty_override:
            If provided, overrides ``self.uncertainty_penalty`` for this call.
        """
        lam = penalty_override if penalty_override is not None else self.uncertainty_penalty
        mean, std = self.forward(input_ids, attention_mask)
        return mean - lam * std

    # ------------------------------------------------------------------
    @property
    def K(self) -> int:
        return len(self.models)

    @classmethod
    def from_checkpoints(
        cls,
        checkpoint_dirs: List[str],
        uncertainty_penalty: float = 0.5,
    ) -> "RewardEnsemble":
        """Load K reward models from K checkpoint directories."""
        models = [
            GPT2RewardModel.from_pretrained(ckpt) for ckpt in checkpoint_dirs
        ]
        return cls(models, uncertainty_penalty=uncertainty_penalty)
