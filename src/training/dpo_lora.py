"""
Extension 4 — LoRA DPO: Parameter-Efficient Direct Preference Optimization.

Design
------
DPO requires keeping two models in memory: the policy being trained and the
frozen reference policy.  With full fine-tuning this doubles GPU memory pressure.

LoRA DPO cuts trainable parameters to ~0.5% of the policy while the reference
policy is *not* adapted at all — it remains the raw SFT checkpoint.  Memory
breakdown for GPT-2-medium:

    Full DPO  : 2 × 1.4 GB  ≈ 2.8 GB model weights  +  optimizer state
    LoRA DPO  : 1.4 GB ref  +  1.4 GB frozen base  +  ~0.007 GB adapters
              ≈ same memory for weights but 200× smaller optimizer state

Practical note
--------------
After LoRA DPO training we optionally merge the adapter into the base model so
downstream PPO / evaluation scripts can load it without PEFT.

Expected finding
----------------
LoRA DPO (r=16) reaches within 1 pp of full DPO pairwise accuracy on the
held-out preference set, training only ~1.8M parameters instead of 355M.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from datasets import Dataset as HFDataset
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import AutoTokenizer, GPT2LMHeadModel
from trl import DPOTrainer, DPOConfig

from src.data.preprocessing import DPODataset
from src.training.sft_lora import count_parameters, merge_and_save


@dataclass
class LoRADPOConfig:
    sft_checkpoint: str = "checkpoints/sft"  # frozen reference + adapter init point
    output_dir: str = "checkpoints/dpo_lora_r16"

    # LoRA adapter hyper-parameters
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: ["c_attn", "c_proj"]
    )

    # DPO training schedule
    beta: float = 0.1
    learning_rate: float = 1e-4          # 200× larger than full DPO (tiny adapter)
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

    # Post-training: merge adapter into base model for easy downstream use
    merge_after_training: bool = True
    merged_output_dir: str = "checkpoints/dpo_lora_r16_merged"


def train_dpo_lora(cfg: LoRADPOConfig) -> Dict[str, int]:
    """Fine-tune SFT model with LoRA adapters using DPO objective (Stage 3b variant).

    The reference policy is the *unmodified* SFT checkpoint — it never receives
    LoRA adapters and has all its parameters frozen.  Only the policy model gets
    the trainable low-rank matrices.

    Returns
    -------
    dict with trainable / total parameter counts.
    """
    tokenizer = AutoTokenizer.from_pretrained(cfg.sft_checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Reference model: frozen full SFT checkpoint (no LoRA)
    ref_model = GPT2LMHeadModel.from_pretrained(cfg.sft_checkpoint)
    for p in ref_model.parameters():
        p.requires_grad_(False)

    # Policy model: SFT checkpoint + LoRA adapters
    base_model = GPT2LMHeadModel.from_pretrained(cfg.sft_checkpoint)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        bias="none",
        inference_mode=False,
    )
    policy_model = get_peft_model(base_model, lora_config)

    param_stats = count_parameters(policy_model)
    print(
        f"\nLoRA DPO (r={cfg.lora_r})  |  "
        f"Trainable: {param_stats['trainable']:,}  |  "
        f"Total: {param_stats['total']:,}  |  "
        f"Fraction: {param_stats['fraction_pct']:.2f}%"
    )

    train_dataset = DPODataset(
        "train", tokenizer, max_length=cfg.max_length,
        num_samples=cfg.num_train_samples,
    )
    eval_dataset = DPODataset(
        "test", tokenizer, max_length=cfg.max_length,
        num_samples=cfg.num_eval_samples,
    )

    # Convert to HF Dataset for trl compatibility
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
        model=policy_model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=train_hf,
        eval_dataset=eval_hf,
        tokenizer=tokenizer,
    )

    print(f"Training LoRA DPO on {len(train_hf):,} pairs  →  {cfg.output_dir}")
    print(f"  β={cfg.beta}  |  adapter rank r={cfg.lora_r}")
    trainer.train()

    # Save adapter weights
    policy_model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print(f"LoRA DPO adapter saved to {cfg.output_dir}")

    if cfg.merge_after_training:
        merge_and_save(
            cfg.output_dir,
            cfg.merged_output_dir,
            model_name=cfg.sft_checkpoint,
        )

    return param_stats


def compare_lora_vs_full_dpo(
    lora_dpo_dir: str,
    full_dpo_dir: str,
    reward_model_dir: str,
    num_eval: int = 500,
    device: str = "cuda",
) -> Dict[str, float]:
    """Score LoRA-DPO and full-DPO on held-out preference pairs using a RM judge.

    Both models generate a response for each prompt; the reward model scores
    them.  Returns win rates and mean reward for the ablation table.
    """
    from src.evaluation.metrics import compute_win_rate_rm

    results = {}
    for name, ckpt in [("lora_dpo", lora_dpo_dir), ("full_dpo", full_dpo_dir)]:
        wr, mean_r = compute_win_rate_rm(
            policy_checkpoint=ckpt,
            reward_checkpoint=reward_model_dir,
            num_eval=num_eval,
            device=device,
        )
        results[name] = {"win_rate": wr, "mean_reward": mean_r}
        print(f"{name:12s} | win_rate={wr:.3f} | mean_reward={mean_r:.4f}")

    return results
