"""
Reward Model Architecture — GPT-2 with Bradley-Terry preference head.

Architecture
------------
We wrap GPT-2 (transformer backbone only — no LM head) with a single linear
projection that maps the last non-padding hidden state to a scalar reward.

    input_ids  ──►  GPT-2 Transformer  ──►  last-token hidden state  ──►  Linear(H→1)  ──►  r ∈ ℝ

Why the last token?
    GPT-2 uses *causal* (left-to-right) self-attention, so the final position has
    attended to every prior token.  It provides the richest summary of the whole
    sequence — analogous to [CLS] in BERT.

Training objective — Bradley-Terry model
-----------------------------------------
Given a preference pair (x, y_w, y_l) where humans prefer y_w over y_l:

    P(y_w ≻ y_l | x)  =  σ( r(x, y_w) − r(x, y_l) )

Maximum-likelihood estimation over the dataset gives the loss:

    L_RM  =  −E[(x,y_w,y_l)] [ log σ( r(x,y_w) − r(x,y_l) ) ]

This is identical to binary cross-entropy on the reward *margin*.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model, GPT2LMHeadModel, GPT2PreTrainedModel
from transformers.modeling_outputs import ModelOutput


@dataclass
class RewardModelOutput(ModelOutput):
    """Output container for :class:`GPT2RewardModel`.

    Attributes
    ----------
    rewards:
        Shape ``(batch_size,)`` scalar reward for each input sequence.
    hidden_states:
        Shape ``(batch_size, seq_len, hidden_size)`` — only populated when
        ``return_hidden_states=True``.
    """

    rewards: torch.FloatTensor = None
    hidden_states: Optional[torch.FloatTensor] = None


class GPT2RewardModel(GPT2PreTrainedModel):
    """GPT-2 with a scalar regression head for preference learning.

    Inheriting from :class:`GPT2PreTrainedModel` gives us HuggingFace's
    ``save_pretrained`` / ``from_pretrained`` infrastructure for free.

    Parameters
    ----------
    config:
        Standard GPT-2 config.  ``config.n_embd`` determines the head size.
    """

    def __init__(self, config: GPT2Config) -> None:
        super().__init__(config)
        self.transformer = GPT2Model(config)

        # Single linear layer, no bias — keeps the reward scale well-conditioned
        self.reward_head = nn.Linear(config.n_embd, 1, bias=False)

        # Small init: avoids pathologically large rewards at the start of training
        nn.init.normal_(self.reward_head.weight, std=0.02)

        self.post_init()

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        return_hidden_states: bool = False,
    ) -> RewardModelOutput:
        """Compute a scalar reward for each sequence in the batch.

        Parameters
        ----------
        input_ids:
            ``(batch, seq_len)`` — token IDs for the full prompt + response.
        attention_mask:
            ``(batch, seq_len)`` — 1 for real tokens, 0 for padding.
        return_hidden_states:
            If True, include the full hidden state tensor in the output.

        Returns
        -------
        RewardModelOutput
            ``.rewards`` has shape ``(batch,)``.
        """
        out = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state  # (B, T, H)

        # Locate the last *real* token in each sequence.
        # For right-padded inputs: last real token = (sum of mask) - 1.
        if attention_mask is not None:
            last_idx = attention_mask.sum(dim=1) - 1  # (B,)
        else:
            last_idx = torch.full(
                (hidden.size(0),), hidden.size(1) - 1, device=hidden.device
            )

        batch_size = hidden.size(0)
        # Gather: shape (B, H)
        pooled = hidden[torch.arange(batch_size, device=hidden.device), last_idx]

        rewards = self.reward_head(pooled).squeeze(-1)  # (B,)

        return RewardModelOutput(
            rewards=rewards,
            hidden_states=hidden if return_hidden_states else None,
        )

    # ------------------------------------------------------------------
    @classmethod
    def from_pretrained_backbone(
        cls,
        model_name: str = "gpt2-medium",
    ) -> "GPT2RewardModel":
        """Initialise reward model from a raw GPT-2 checkpoint.

        The transformer weights are copied from the pre-trained checkpoint;
        the reward head is randomly initialised.
        """
        config = GPT2Config.from_pretrained(model_name)
        model = cls(config)
        backbone = GPT2Model.from_pretrained(model_name)
        model.transformer.load_state_dict(backbone.state_dict())
        return model

    @classmethod
    def from_sft_checkpoint(cls, checkpoint_path: str) -> "GPT2RewardModel":
        """Initialise reward model from an SFT checkpoint.

        Initialising from the SFT model (rather than raw GPT-2) is the standard
        practice: the SFT model already understands the Human/Assistant format,
        giving the reward model a better starting point.
        """
        sft = GPT2LMHeadModel.from_pretrained(checkpoint_path)
        model = cls(sft.config)
        model.transformer.load_state_dict(sft.transformer.state_dict())
        return model


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def preference_loss(
    chosen_rewards: torch.Tensor,
    rejected_rewards: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Bradley-Terry preference loss.

    For a batch of preference pairs:

        loss  =  −mean[ log σ( r_w − r_l ) ]

    Parameters
    ----------
    chosen_rewards:
        Scalar rewards for the preferred responses, shape ``(B,)``.
    rejected_rewards:
        Scalar rewards for the dispreferred responses, shape ``(B,)``.

    Returns
    -------
    loss:
        Scalar training loss.
    accuracy:
        Fraction of pairs in which the model correctly ranks chosen > rejected.
        Use as a training-progress metric (random ≈ 0.5, trained ≥ 0.7 is good).
    """
    margin = chosen_rewards - rejected_rewards
    loss = -F.logsigmoid(margin).mean()
    accuracy = (margin > 0).float().mean()
    return loss, accuracy
