#!/usr/bin/env python3
"""
PPO training with uncertainty-penalised ensemble reward.

The key difference from train_ppo.py:
  - Reward = mean(r_1, ..., r_K) - λ * std(r_1, ..., r_K)
  - High ensemble disagreement (σ) → penalised reward → policy avoids uncertain regions

This directly prevents the reward hacking documented in the baseline PPO run, where
the policy exploited a single reward model's blind spots.

Usage
-----
# Train with ensemble reward (requires checkpoints/reward_ensemble/)
python scripts/train_ppo_ensemble.py \\
    --sft_checkpoint checkpoints/sft \\
    --ensemble_dir checkpoints/reward_ensemble \\
    --uncertainty_penalty 0.5

# Ablation: vary uncertainty penalty
for lam in 0.0 0.25 0.5 1.0; do
    python scripts/train_ppo_ensemble.py \\
        --uncertainty_penalty $lam \\
        --output_dir checkpoints/ppo_ensemble_lam$lam
done
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead

from src.models.reward_ensemble import RewardEnsemble
from src.data.preprocessing import extract_prompt_and_response
from src.training.ppo import _build_prompt_dataset


def score_with_ensemble(
    ensemble: RewardEnsemble,
    tokenizer,
    prompts: list,
    responses: list,
    device: torch.device,
    penalty: float,
    max_length: int = 512,
) -> list:
    texts = [p + r for p, r in zip(prompts, responses)]
    enc = tokenizer(
        texts, max_length=max_length, truncation=True,
        padding=True, return_tensors="pt",
    ).to(device)
    penalized = ensemble.penalized_reward(
        enc["input_ids"], enc["attention_mask"],
        penalty_override=penalty,
    )
    return [r for r in penalized]


def main():
    p = argparse.ArgumentParser(description="PPO with ensemble uncertainty penalty")
    p.add_argument("--sft_checkpoint", default="checkpoints/sft")
    p.add_argument("--ensemble_dir", default="checkpoints/reward_ensemble",
                   help="Directory containing model_0/, model_1/, ... subdirectories")
    p.add_argument("--output_dir", default="checkpoints/ppo_ensemble")
    p.add_argument("--uncertainty_penalty", type=float, default=0.5,
                   help="λ — how much to penalise ensemble disagreement. "
                        "0=plain mean, 1=aggressive uncertainty penalty.")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_samples", type=int, default=5_000)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--lr", type=float, default=1.41e-5)
    p.add_argument("--init_kl_coef", type=float, default=0.2)
    p.add_argument("--log_every", type=int, default=10)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load ensemble from manifest
    manifest = os.path.join(args.ensemble_dir, "manifest.txt")
    if os.path.exists(manifest):
        with open(manifest) as f:
            ckpts = [l.strip() for l in f if l.strip()]
    else:
        # Fall back: scan for model_* subdirs
        ckpts = sorted([
            os.path.join(args.ensemble_dir, d)
            for d in os.listdir(args.ensemble_dir)
            if d.startswith("model_")
        ])

    print(f"Loading ensemble of {len(ckpts)} reward models:")
    for c in ckpts:
        print(f"  {c}")

    ensemble = RewardEnsemble.from_checkpoints(ckpts, uncertainty_penalty=args.uncertainty_penalty)
    ensemble.eval()
    for param in ensemble.parameters():
        param.requires_grad_(False)
    ensemble.to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.sft_checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model     = AutoModelForCausalLMWithValueHead.from_pretrained(args.sft_checkpoint)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(args.sft_checkpoint)

    ppo_config = PPOConfig(
        model_name=args.sft_checkpoint,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        mini_batch_size=4,
        ppo_epochs=4,
        init_kl_coef=args.init_kl_coef,
        adap_kl_ctrl=True,
        target_kl=6.0,
        log_with=None,
    )

    prompt_dataset = _build_prompt_dataset(
        args.sft_checkpoint, args.num_samples, max_prompt_length=256
    )

    trainer = PPOTrainer(
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

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens, do_sample=True,
        top_k=50, top_p=0.9, temperature=0.7,
        pad_token_id=tokenizer.pad_token_id,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    for step, batch in enumerate(trainer.dataloader):
        query_tensors = batch["input_ids"]
        response_tensors = trainer.generate(
            query_tensors, return_prompt=False, **gen_kwargs
        )
        batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

        rewards = score_with_ensemble(
            ensemble, tokenizer,
            batch["query"], batch["response"],
            device, penalty=args.uncertainty_penalty,
        )

        stats = trainer.step(query_tensors, response_tensors, rewards)

        if step % args.log_every == 0:
            mean_r = torch.stack(rewards).mean().item()
            kl = stats.get("objective/kl", float("nan"))
            print(f"Step {step:4d}  reward={mean_r:.4f}  kl={kl:.4f}  λ={args.uncertainty_penalty}")

    trainer.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Ensemble-PPO policy saved to {args.output_dir}")


if __name__ == "__main__":
    main()
