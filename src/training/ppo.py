"""
Stage 3a — Proximal Policy Optimization (PPO).

How PPO works for RLHF
-----------------------
We treat the language model as a policy π_θ that, given a prompt x, produces a
response y token-by-token.  The reward model r_φ(x, y) provides a scalar signal
after the response is complete.

The PPO objective (clipped surrogate + KL penalty) is:

    L_PPO = E[ min(ρ_t · A_t,  clip(ρ_t, 1−ε, 1+ε) · A_t) ]  −  β · KL(π_θ ‖ π_ref)

where:
  ρ_t = π_θ(a_t | s_t) / π_θ_old(a_t | s_t)   — importance-sampling ratio
  A_t = r_φ(x, y) − V(s_t)                      — advantage (RM reward − value baseline)
  ε   = 0.2                                       — clipping radius
  β   = 0.05–0.1                                  — KL coefficient

The KL penalty is *crucial*: without it, the policy quickly exploits the reward
model's blind spots and produces degenerate text (reward hacking).

Implementation
--------------
We wrap trl.PPOTrainer, which handles:
  - Rollout generation from the current policy
  - Value estimation and advantage computation (GAE)
  - Clipped surrogate loss + entropy bonus
  - KL tracking

We add a custom `score_responses` function that calls our reward model,
not trl's built-in pipeline — this keeps the reward model decoupled.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from datasets import load_dataset, Dataset as HFDataset
from transformers import AutoTokenizer, GPT2LMHeadModel, pipeline
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead

from src.models.reward_model import GPT2RewardModel


@dataclass
class PPOTrainingConfig:
    sft_checkpoint: str = "checkpoints/sft"
    reward_checkpoint: str = "checkpoints/reward_model"
    output_dir: str = "checkpoints/ppo"
    # PPO hyper-parameters
    learning_rate: float = 1.41e-5
    batch_size: int = 16             # number of (prompt, response) pairs per PPO step
    mini_batch_size: int = 4         # gradient accumulation mini-batch
    gradient_accumulation_steps: int = 1
    ppo_epochs: int = 4              # number of PPO update passes per rollout batch
    kl_penalty: str = "kl"           # "kl" or "abs" or "mse"
    init_kl_coef: float = 0.2        # initial β for KL penalty
    adap_kl_ctrl: bool = True        # adaptively control β to hit target KL
    target_kl: float = 6.0           # target KL divergence per step
    clip_range: float = 0.2          # ε for clipped surrogate
    vf_coef: float = 0.1             # value-function loss coefficient
    max_new_tokens: int = 128        # max response length during rollout
    max_prompt_length: int = 256
    num_train_samples: Optional[int] = 5_000
    log_every: int = 10
    fp16: bool = True


def _build_prompt_dataset(
    sft_checkpoint: str,
    num_samples: Optional[int],
    max_prompt_length: int,
) -> HFDataset:
    """Extract prompts from hh-rlhf for use as rollout inputs."""
    from src.data.preprocessing import extract_prompt_and_response

    tokenizer = AutoTokenizer.from_pretrained(sft_checkpoint)
    tokenizer.pad_token = tokenizer.eos_token

    raw = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base", split="train")
    if num_samples and num_samples > 0:
        raw = raw.select(range(min(num_samples, len(raw))))

    def tokenize_prompt(example):
        prompt, _ = extract_prompt_and_response(example["chosen"])
        enc = tokenizer(
            prompt,
            max_length=max_prompt_length,
            truncation=True,
            padding=False,
        )
        return {"input_ids": enc["input_ids"], "query": prompt}

    return raw.map(tokenize_prompt, remove_columns=raw.column_names)


@torch.no_grad()
def score_responses(
    reward_model: GPT2RewardModel,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    responses: List[str],
    device: torch.device,
    max_length: int = 512,
) -> List[torch.Tensor]:
    """Score a batch of (prompt, response) pairs with the reward model.

    Returns a list of scalar tensors, one per pair, as required by trl.PPOTrainer.
    """
    texts = [p + r for p, r in zip(prompts, responses)]
    enc = tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding=True,
        return_tensors="pt",
    ).to(device)
    rewards = reward_model(**enc).rewards  # (B,)
    return [r for r in rewards]


def train_ppo(cfg: PPOTrainingConfig) -> None:
    """Fine-tune the SFT model via PPO using the trained reward model (Stage 3a).

    Training loop overview:
      1. Sample a batch of prompts from the dataset.
      2. Generate responses from the current policy (rollout).
      3. Score each (prompt, response) pair with the reward model.
      4. Run PPO_EPOCHS passes of clipped-surrogate + KL update.
      5. Repeat.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(cfg.sft_checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # PPO requires left-padding for generation

    # Policy + value head (trl wraps GPT-2 with an additional linear value head)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(cfg.sft_checkpoint)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(cfg.sft_checkpoint)

    reward_model = GPT2RewardModel.from_pretrained(cfg.reward_checkpoint).to(device)
    reward_model.eval()

    ppo_config = PPOConfig(
        model_name=cfg.sft_checkpoint,
        learning_rate=cfg.learning_rate,
        batch_size=cfg.batch_size,
        mini_batch_size=cfg.mini_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        ppo_epochs=cfg.ppo_epochs,
        kl_penalty=cfg.kl_penalty,
        init_kl_coef=cfg.init_kl_coef,
        adap_kl_ctrl=cfg.adap_kl_ctrl,
        target_kl=cfg.target_kl,
        cliprange=cfg.clip_range,
        vf_coef=cfg.vf_coef,
        log_with=None,
    )

    prompt_dataset = _build_prompt_dataset(
        cfg.sft_checkpoint, cfg.num_train_samples, cfg.max_prompt_length
    )

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=prompt_dataset,
        data_collator=lambda data: {
            "input_ids": [d["input_ids"] for d in data],
            "query": [d["query"] for d in data],
        },
    )

    generation_kwargs = dict(
        max_new_tokens=cfg.max_new_tokens,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        temperature=0.7,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    os.makedirs(cfg.output_dir, exist_ok=True)

    for step, batch in enumerate(ppo_trainer.dataloader):
        query_tensors = batch["input_ids"]

        # --- Rollout: generate responses from the current policy ---
        response_tensors = ppo_trainer.generate(
            query_tensors, return_prompt=False, **generation_kwargs
        )
        batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

        # --- Score: use reward model to assign scalar rewards ---
        rewards = score_responses(
            reward_model,
            tokenizer,
            batch["query"],
            batch["response"],
            device,
        )

        # --- PPO update step ---
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

        if step % cfg.log_every == 0:
            mean_reward = torch.stack(rewards).mean().item()
            ppo_kl = stats.get("objective/kl", float("nan"))
            print(f"Step {step:4d}  reward={mean_reward:.4f}  kl={ppo_kl:.4f}")

    ppo_trainer.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print(f"PPO policy saved to {cfg.output_dir}")
