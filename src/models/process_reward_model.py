"""
Process Reward Model (PRM) — step-level reward signal.

Motivation
----------
An Outcome Reward Model (ORM) scores a complete response as a whole.  This
makes it blind to *how* the answer was reached: a model can arrive at the right
answer via faulty reasoning, and the ORM will reward it.

A Process Reward Model (PRM) scores each reasoning step independently.  The
policy is penalised if *any* step is incorrect, even if the final answer happens
to be right.  This provides a denser, more faithful training signal for tasks
that require multi-step reasoning.

Real-world use
--------------
OpenAI's "Let's Verify Step by Step" (Lightman et al., 2023) and Anthropic's
research on chain-of-thought reliability both argue that step-level verification
catches a large class of failures that ORM misses.

Architecture
------------
GPT-2 backbone + a binary classification head.  At inference time, the head is
applied at every *step-boundary* position in the sequence (positions where a
separator token appears in the input).  Each step gets a probability of being
correct.  The aggregate reward can be:

    sum_aggregation  : Σ log p(step_i = correct)   — penalises any wrong step
    min_aggregation  : min_i p(step_i = correct)   — the "weakest link" score
    mean_aggregation : mean_i p(step_i = correct)  — average step quality
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model, GPT2PreTrainedModel
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass


@dataclass
class PRMOutput(ModelOutput):
    """Output container for :class:`GPT2ProcessRewardModel`.

    Attributes
    ----------
    step_logits:
        Shape ``(B, T)`` — raw logit for "step is correct" at every position.
        Only step-boundary positions carry meaningful signal; all others are noise.
    step_probs:
        Shape ``(B, T)`` — sigmoid of step_logits.
    aggregate_reward:
        Shape ``(B,)`` — aggregated scalar reward (method depends on aggregation_mode).
    """

    step_logits: torch.FloatTensor = None
    step_probs: torch.FloatTensor = None
    aggregate_reward: Optional[torch.FloatTensor] = None


class GPT2ProcessRewardModel(GPT2PreTrainedModel):
    """GPT-2 that predicts step-level correctness at separator positions.

    Parameters
    ----------
    config:
        Standard GPT-2 config.
    aggregation_mode:
        How to aggregate per-step scores to a single reward.
        Options: "mean" | "min" | "sum".  "min" is most conservative.
    sep_token_id:
        Token ID used as a step separator in the input sequence.
        We identify step-boundary positions by this token.
    """

    AGGREGATION_MODES = ("mean", "min", "sum")

    def __init__(
        self,
        config: GPT2Config,
        aggregation_mode: str = "mean",
        sep_token_id: int = 50256,   # GPT-2's EOS token
    ) -> None:
        super().__init__(config)
        if aggregation_mode not in self.AGGREGATION_MODES:
            raise ValueError(f"aggregation_mode must be one of {self.AGGREGATION_MODES}")

        self.transformer = GPT2Model(config)
        # Binary head: predicts P(step_correct | context up to step boundary)
        self.step_head = nn.Linear(config.n_embd, 1, bias=False)
        nn.init.normal_(self.step_head.weight, std=0.02)

        self.aggregation_mode = aggregation_mode
        self.sep_token_id = sep_token_id
        self.post_init()

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        step_labels: Optional[torch.LongTensor] = None,
    ) -> PRMOutput:
        """
        Parameters
        ----------
        input_ids:
            ``(B, T)`` — full question + solution token sequence, with step
            boundaries marked by ``sep_token_id``.
        attention_mask:
            ``(B, T)`` — 1 for real tokens, 0 for padding.
        step_labels:
            ``(B, T)`` — binary labels at each position:
                1 = step is correct, 0 = step is incorrect, -1 = not a step boundary.
            If provided, also computes and returns the PRM training loss.

        Returns
        -------
        PRMOutput
            ``.aggregate_reward`` : shape ``(B,)``
            ``.step_logits``      : shape ``(B, T)``
        """
        hidden = self.transformer(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state  # (B, T, H)

        step_logits = self.step_head(hidden).squeeze(-1)   # (B, T)
        step_probs = torch.sigmoid(step_logits)            # (B, T)

        # Aggregate over actual step-boundary positions
        sep_mask = (input_ids == self.sep_token_id)        # (B, T)

        # For each sequence, gather probs at step positions then aggregate
        batch_size = input_ids.size(0)
        aggregate_rewards = []

        for b in range(batch_size):
            sep_probs = step_probs[b][sep_mask[b]]   # (n_steps_b,)
            if sep_probs.numel() == 0:
                # No separators found — fall back to final-token reward
                agg = step_probs[b, (attention_mask[b].sum() - 1).long()]
            elif self.aggregation_mode == "mean":
                agg = sep_probs.mean()
            elif self.aggregation_mode == "min":
                agg = sep_probs.min()
            elif self.aggregation_mode == "sum":
                agg = sep_probs.sum()
            aggregate_rewards.append(agg)

        aggregate_reward = torch.stack(aggregate_rewards)  # (B,)

        return PRMOutput(
            step_logits=step_logits,
            step_probs=step_probs,
            aggregate_reward=aggregate_reward,
        )

    # ------------------------------------------------------------------
    def compute_loss(
        self,
        step_logits: torch.Tensor,    # (B, T)
        step_labels: torch.LongTensor,  # (B, T), -1 at non-boundary positions
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Binary cross-entropy loss at step-boundary positions.

        Returns
        -------
        loss:
            Scalar training loss.
        accuracy:
            Fraction of step-boundary positions correctly classified.
        """
        mask = step_labels != -1                  # (B, T) — True at step boundaries
        if not mask.any():
            return step_logits.sum() * 0, torch.tensor(0.0)

        logits_at_steps = step_logits[mask]       # (N_steps,)
        labels_at_steps = step_labels[mask].float()

        loss = F.binary_cross_entropy_with_logits(logits_at_steps, labels_at_steps)
        accuracy = ((logits_at_steps > 0) == labels_at_steps.bool()).float().mean()
        return loss, accuracy

    # ------------------------------------------------------------------
    @classmethod
    def from_sft_checkpoint(
        cls,
        checkpoint_path: str,
        aggregation_mode: str = "mean",
        sep_token_id: int = 50256,
    ) -> "GPT2ProcessRewardModel":
        """Initialise PRM from an SFT checkpoint."""
        from transformers import GPT2LMHeadModel
        sft = GPT2LMHeadModel.from_pretrained(checkpoint_path)
        model = cls(sft.config, aggregation_mode=aggregation_mode, sep_token_id=sep_token_id)
        model.transformer.load_state_dict(sft.transformer.state_dict())
        return model
