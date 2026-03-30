"""
Stage 3b — Direct Preference Optimization (DPO).

Motivation
----------
PPO requires four models in memory simultaneously (policy, value, reference, reward
model) and an outer rollout loop.  DPO is a single-model, regression-style algorithm
that achieves the same alignment goal without any of that infrastructure.

Derivation sketch
-----------------
The RLHF objective is:

    max_π  E_{x~D, y~π}[ r(x,y) ]  −  β · KL(π(·|x) ‖ π_ref(·|x))

The optimal policy under this objective can be written in closed form:

    π*(y|x)  ∝  π_ref(y|x) · exp( r(x,y) / β )

Rearranging gives the reward as a function of the optimal policy:

    r*(x,y)  =  β · log( π*(y|x) / π_ref(y|x) )  +  β · log Z(x)

Substituting into the Bradley-Terry pairwise loss and noting that Z(x) cancels:

    L_DPO  =  −E[(x,y_w,y_l)] [
        log σ(  β · log(π_θ(y_w|x)/π_ref(y_w|x))
              − β · log(π_θ(y_l|x)/π_ref(y_l|x))  )
    ]

Key insight: DPO *implicitly* learns a reward without ever calling a reward model.
The ratio log(π_θ/π_ref) acts as an implicit reward: if the policy assigns higher
probability to y_w than the reference does, it is effectively up-voting y_w.

Practical advantages over PPO
------------------------------
  + No reward model needed at training time
  + No separate value network
  + No on-policy rollouts — trains on fixed preference pairs (offline)
  + 2× fewer GPU memory requirements
  − Less robust to distribution shift (offline data only)
  − Quality heavily depends on the SFT model's initial distribution
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from trl import DPOTrainer, DPOConfig

from src.data.preprocessing import DPODataset


@dataclass
class DPOTrainingConfig:
    sft_checkpoint: str = "checkpoints/sft"
    output_dir: str = "checkpoints/dpo"
    beta: float = 0.1                         # KL coefficient (temperature of implicit reward)
    learning_rate: float = 5e-7               # much smaller than SFT — we nudge, not retrain
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_length: int = 512
    max_prompt_length: int = 256
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    num_train_samples: Optional[int] = 10_000
    num_eval_samples: Optional[int] = 1_000
    fp16: bool = True
    logging_steps: int = 50
    eval_steps: int = 200
    save_steps: int = 500
    report_to: str = "none"


# ---------------------------------------------------------------------------
# First-principles DPO loss — for educational use / unit tests
# ---------------------------------------------------------------------------

def sequence_logprob(
    logits: torch.Tensor,          # (B, T, V)
    input_ids: torch.Tensor,       # (B, T)
    attention_mask: torch.Tensor,  # (B, T)
) -> torch.Tensor:
    """Compute the average log-probability of the response tokens.

    We sum log p(y_t | y_{<t}, x) over response positions and average by
    response length to remove length bias.

    Parameters
    ----------
    logits:
        Raw logits from the language model, shape (B, T, V).
    input_ids:
        Token IDs, shape (B, T).
    attention_mask:
        1 for real tokens (prompt + response), 0 for padding.

    Returns
    -------
    Tensor of shape (B,) — mean per-token log-prob of each sequence.
    """
    # Shift: logits[t] predicts token[t+1]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_ids = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()

    log_probs = F.log_softmax(shift_logits, dim=-1)  # (B, T-1, V)
    # Gather the log-prob for each actual next token
    token_logps = log_probs.gather(2, shift_ids.unsqueeze(-1)).squeeze(-1)  # (B, T-1)
    # Mask out padding and average
    token_logps = token_logps * shift_mask
    return token_logps.sum(dim=1) / shift_mask.sum(dim=1).clamp(min=1)


def dpo_loss(
    policy_chosen_logps: torch.Tensor,    # (B,)
    policy_rejected_logps: torch.Tensor,  # (B,)
    ref_chosen_logps: torch.Tensor,       # (B,)
    ref_rejected_logps: torch.Tensor,     # (B,)
    beta: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Direct Preference Optimization loss (from scratch).

    Computes:

        L_DPO  =  −mean[ log σ( β · (log π_θ(y_w|x) − log π_ref(y_w|x))
                                − β · (log π_θ(y_l|x) − log π_ref(y_l|x)) ) ]

    Parameters
    ----------
    policy_chosen_logps, policy_rejected_logps:
        Average per-token log-probs from the *policy being trained*.
    ref_chosen_logps, ref_rejected_logps:
        Average per-token log-probs from the *frozen reference policy* (SFT model).
    beta:
        Controls how much the policy can deviate from the reference.
        β → 0 recovers pure preference learning; β → ∞ freezes the policy.

    Returns
    -------
    loss:
        Scalar DPO loss.
    chosen_rewards:
        Implicit rewards β · (log π_θ(y_w) − log π_ref(y_w)), shape (B,).
    rejected_rewards:
        Implicit rewards β · (log π_θ(y_l) − log π_ref(y_l)), shape (B,).
    """
    # Implicit reward: how much more (or less) the policy favours y over the reference
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)

    logits = chosen_rewards - rejected_rewards
    loss = -F.logsigmoid(logits).mean()
    return loss, chosen_rewards.detach(), rejected_rewards.detach()


# ---------------------------------------------------------------------------
# High-level training entry point (uses trl.DPOTrainer)
# ---------------------------------------------------------------------------

def train_dpo(cfg: DPOTrainingConfig) -> None:
    """Fine-tune the SFT model via DPO on human preference pairs (Stage 3b).

    Unlike PPO, DPO does not require:
      - Rollout generation loops
      - A value network
      - An explicit reward model at training time

    The reference model (frozen SFT) is kept in memory but has its gradients
    disabled; it only contributes the log π_ref term to the DPO loss.
    """
    tokenizer = AutoTokenizer.from_pretrained(cfg.sft_checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    from transformers import GPT2LMHeadModel

    model = GPT2LMHeadModel.from_pretrained(cfg.sft_checkpoint)
    ref_model = GPT2LMHeadModel.from_pretrained(cfg.sft_checkpoint)
    # Reference model: frozen — only used for log π_ref computation
    for p in ref_model.parameters():
        p.requires_grad_(False)

    train_dataset = DPODataset(
        "train", tokenizer, max_length=cfg.max_length,
        num_samples=cfg.num_train_samples,
    )
    eval_dataset = DPODataset(
        "test", tokenizer, max_length=cfg.max_length,
        num_samples=cfg.num_eval_samples,
    )

    # Convert DPODataset to HuggingFace Dataset for trl compatibility
    from datasets import Dataset as HFDataset
    train_hf = HFDataset.from_list([train_dataset[i] for i in range(len(train_dataset))])
    eval_hf = HFDataset.from_list([eval_dataset[i] for i in range(len(eval_dataset))])

    dpo_config = DPOConfig(
        beta=cfg.beta,
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        max_length=cfg.max_length,
        max_prompt_length=cfg.max_prompt_length,
        evaluation_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        logging_steps=cfg.logging_steps,
        fp16=cfg.fp16,
        load_best_model_at_end=True,
        report_to=cfg.report_to,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=train_hf,
        eval_dataset=eval_hf,
        tokenizer=tokenizer,
    )

    print(f"Training DPO on {len(train_hf):,} preference pairs  →  {cfg.output_dir}")
    print(f"  β={cfg.beta}  (higher β = stronger KL constraint toward reference policy)")
    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print(f"DPO policy saved to {cfg.output_dir}")
