"""
Extension 12: FSDP-wrapped DPO training.

DPO memory challenge: both policy and reference model must be resident during
the forward pass to compute log-ratio rewards. This doubles the model-state
memory vs SFT on the same hardware.

Three reference model strategies, in order of memory efficiency:

  Strategy A — Policy FULL_SHARD, Reference unsharded bf16 (default):
    Policy memory : 14/N bytes/param  (2/N params + 4/N grad + 8/N optim)
    Reference mem :  2   bytes/param  (bf16, no grad, no optim — fixed cost)
    Best when N is large (reference becomes negligible) or model is small.

  Strategy B — Both FULL_SHARD:
    Policy memory : 14/N bytes/param
    Reference mem :  2/N bytes/param  (sharded bf16, no grad/optim needed)
    Best for very large models where even bf16 reference doesn't fit on N GPUs.

  Strategy C — Reference CPU offload:
    Policy on GPU (FULL_SHARD); reference on CPU, moved to GPU per batch.
    Memory optimal at cost of CPU-GPU transfer latency (~2× slower training).
    Use when GPU count is constrained but model is large.

Implementation:
  We use Strategy A (unsharded bf16 reference) as default. The reference
  model holds no gradients and no optimizer state — only 2P bytes.
  The policy is wrapped with FSDP FULL_SHARD, same as fsdp_sft.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from src.training.fsdp_sft import (
    FSDPSFTConfig,
    apply_activation_checkpointing,
    memory_stats_mb,
    wrap_model_with_fsdp,
)


@dataclass
class FSDPDPOConfig:
    sft_checkpoint: str = "checkpoints/sft_fsdp/fsdp_sft.pt"
    model_id: str = "gpt2-medium"
    num_samples: int = 5000
    max_length: int = 512
    max_prompt_length: int = 256
    batch_size: int = 1                    # DPO: 3 forward passes per pair (chosen, rejected × 2 models)
    gradient_accumulation_steps: int = 16
    num_epochs: int = 1
    learning_rate: float = 5e-7
    beta: float = 0.1
    output_dir: str = "checkpoints/dpo_fsdp"

    # FSDP (policy only)
    sharding_strategy: str = "FULL_SHARD"
    use_mixed_precision: bool = True
    use_activation_checkpointing: bool = True

    # Reference model strategy
    ref_cpu_offload: bool = False     # Strategy C: offload reference to CPU

    simulate_num_gpus: int = 1
    profile_memory: bool = True


# ---------------------------------------------------------------------------
# DPO loss
# ---------------------------------------------------------------------------

def dpo_loss(
    policy_chosen_lp: torch.Tensor,
    policy_rejected_lp: torch.Tensor,
    ref_chosen_lp: torch.Tensor,
    ref_rejected_lp: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Standard DPO loss (Rafailov et al. 2023):
      L = -E[ log σ( β(log π_θ(y_w|x) - log π_ref(y_w|x))
                   - β(log π_θ(y_l|x) - log π_ref(y_l|x)) ) ]

    Returns (loss, chosen_rewards, rejected_rewards).
    """
    chosen_rewards   = beta * (policy_chosen_lp   - ref_chosen_lp)
    rejected_rewards = beta * (policy_rejected_lp - ref_rejected_lp)
    loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
    return loss, chosen_rewards.detach(), rejected_rewards.detach()


def _sequence_log_probs(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    no_grad: bool = False,
) -> torch.Tensor:
    """
    Compute per-sequence sum of log P(token | prefix) for response tokens only.
    Labels have -100 at prompt positions (masked); we sum only unmasked positions.

    Shift: logits[t] predicts labels[t+1], so we shift both by 1.
    """
    ctx = torch.no_grad() if no_grad else torch.enable_grad()
    with ctx:
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    shift_logits = logits[:, :-1, :]               # (B, L-1, V)
    shift_labels = labels[:, 1:]                   # (B, L-1)
    log_probs    = F.log_softmax(shift_logits, dim=-1)
    token_lp     = log_probs.gather(2, shift_labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    mask         = (shift_labels != -100).float()
    return (token_lp * mask).sum(-1)               # (B,)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_dpo_fsdp(cfg: FSDPDPOConfig) -> dict:
    """
    FSDP DPO: policy is FULL_SHARD, reference is frozen bf16 (Strategy A).

    Memory accounting at GPT-2-medium (355M), 1 GPU, bf16 mixed + grad-ckpt:
      Reference (unsharded bf16):  355M × 2 bytes   ≈  0.71 GB
      Policy FULL_SHARD (1 GPU):   355M × 18 bytes  ≈  6.39 GB  (full 18 bytes/param)
      Activations (ckpt, B=1):                       ≈  0.10 GB
      Total peak                                     ≈  7.20 GB  — fits on 1× RTX 3080 (10GB)

    On 4× A100: policy sharded to 1.6 GB/GPU; reference still 0.71 GB (unsharded replica)
    → Total ≈ 2.31 GB/GPU, leaving 77 GB free for larger batches or longer sequences.
    """
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from transformers import (
        GPT2LMHeadModel,
        GPT2Tokenizer,
        get_linear_schedule_with_warmup,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out    = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    eff_accum = cfg.gradient_accumulation_steps * cfg.simulate_num_gpus
    print(f"[FSDP DPO] strategy={cfg.sharding_strategy} | beta={cfg.beta} | "
          f"eff_batch={cfg.batch_size * eff_accum}")

    tokenizer = GPT2Tokenizer.from_pretrained(cfg.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # --- Reference model (Strategy A: unsharded bf16, no grad, fixed memory) ---
    print("  Loading reference model (bf16, frozen)...")
    ref_model = GPT2LMHeadModel.from_pretrained(cfg.model_id, torch_dtype=torch.bfloat16)
    if Path(cfg.sft_checkpoint).exists():
        sd = torch.load(cfg.sft_checkpoint, map_location="cpu", weights_only=True)
        ref_model.load_state_dict(sd, strict=False)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    if cfg.ref_cpu_offload:
        ref_device = torch.device("cpu")
        ref_model  = ref_model.cpu()
        print("  Reference offloaded to CPU (Strategy C).")
    else:
        ref_device = device
        ref_model  = ref_model.to(device)

    if cfg.profile_memory:
        print(f"  Memory after ref: {memory_stats_mb(device)['allocated_mb']:.0f} MB")

    # --- Policy model (FSDP FULL_SHARD) ---
    print("  Loading policy model (FSDP)...")
    policy = GPT2LMHeadModel.from_pretrained(cfg.model_id)
    if Path(cfg.sft_checkpoint).exists():
        sd = torch.load(cfg.sft_checkpoint, map_location="cpu", weights_only=True)
        policy.load_state_dict(sd, strict=False)

    fsdp_cfg = FSDPSFTConfig(
        model_id=cfg.model_id,
        sharding_strategy=cfg.sharding_strategy,
        use_mixed_precision=cfg.use_mixed_precision,
        use_activation_checkpointing=cfg.use_activation_checkpointing,
    )
    policy = wrap_model_with_fsdp(policy, fsdp_cfg)
    if cfg.use_activation_checkpointing:
        apply_activation_checkpointing(policy)
    policy.to(device)

    if cfg.profile_memory:
        print(f"  Memory after policy FSDP: {memory_stats_mb(device)['allocated_mb']:.0f} MB")

    # --- Dataset ---
    ds = load_dataset("Anthropic/hh-rlhf", split="train")
    if cfg.num_samples < len(ds):
        ds = ds.select(range(cfg.num_samples))

    def tokenize(batch):
        split_token = "\n\nAssistant:"
        prompts = [row.split(split_token)[0] for row in batch["chosen"]]

        def pair_encode(base_texts, responses, max_len):
            full = [p + split_token + r.split(split_token)[-1]
                    for p, r in zip(prompts, responses)]
            enc = tokenizer(full, truncation=True, max_length=max_len,
                            padding="max_length", return_tensors="pt")
            labels = enc["input_ids"].clone()
            # Mask prompt tokens: determine prompt length per example
            for i, p in enumerate(prompts):
                p_enc = tokenizer(p, truncation=True, max_length=cfg.max_prompt_length)
                prompt_len = len(p_enc["input_ids"])
                labels[i, :prompt_len] = -100
            return enc["input_ids"], enc["attention_mask"], labels

        out = {}
        for prefix, texts in [("chosen", batch["chosen"]), ("rejected", batch["rejected"])]:
            iids, amask, lbls = pair_encode(prompts, texts, cfg.max_length)
            out[f"{prefix}_input_ids"]      = iids
            out[f"{prefix}_attention_mask"] = amask
            out[f"{prefix}_labels"]         = lbls
        return out

    ds = ds.map(tokenize, batched=True, remove_columns=ds.column_names)
    ds.set_format("torch")
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    # Optimizer created AFTER FSDP wrap
    optimizer    = torch.optim.AdamW(policy.parameters(), lr=cfg.learning_rate)
    total_steps  = (len(loader) // eff_accum) * cfg.num_epochs
    warmup_steps = max(1, int(total_steps * 0.05))
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    history = {
        "train_loss": [], "chosen_reward": [], "rejected_reward": [],
        "reward_margin": [], "peak_memory_mb": [],
    }
    optimizer.zero_grad()

    for epoch in range(cfg.num_epochs):
        policy.train()
        torch.cuda.reset_peak_memory_stats()
        e_loss = e_chosen = e_rejected = 0.0

        for step, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            # Reference log-probs (no grad; move to ref_device if CPU-offloading)
            with torch.no_grad():
                ref_chosen_lp   = _sequence_log_probs(
                    ref_model,
                    batch["chosen_input_ids"].to(ref_device),
                    batch["chosen_attention_mask"].to(ref_device),
                    batch["chosen_labels"].to(ref_device),
                    no_grad=True,
                ).to(device)
                ref_rejected_lp = _sequence_log_probs(
                    ref_model,
                    batch["rejected_input_ids"].to(ref_device),
                    batch["rejected_attention_mask"].to(ref_device),
                    batch["rejected_labels"].to(ref_device),
                    no_grad=True,
                ).to(device)

            # Policy log-probs (with grad)
            pol_chosen_lp   = _sequence_log_probs(
                policy, batch["chosen_input_ids"], batch["chosen_attention_mask"],
                batch["chosen_labels"])
            pol_rejected_lp = _sequence_log_probs(
                policy, batch["rejected_input_ids"], batch["rejected_attention_mask"],
                batch["rejected_labels"])

            loss, chosen_r, rejected_r = dpo_loss(
                pol_chosen_lp, pol_rejected_lp,
                ref_chosen_lp, ref_rejected_lp,
                cfg.beta,
            )
            (loss / eff_accum).backward()
            e_loss     += loss.item()
            e_chosen   += chosen_r.mean().item()
            e_rejected += rejected_r.mean().item()

            if (step + 1) % eff_accum == 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        n    = len(loader)
        peak = memory_stats_mb(device)["peak_mb"]
        history["train_loss"].append(e_loss / n)
        history["chosen_reward"].append(e_chosen / n)
        history["rejected_reward"].append(e_rejected / n)
        history["reward_margin"].append((e_chosen - e_rejected) / n)
        history["peak_memory_mb"].append(peak)
        print(f"  Epoch {epoch+1}/{cfg.num_epochs} | loss={e_loss/n:.4f} | "
              f"margin={( e_chosen - e_rejected)/n:.3f} | peak={peak:.0f} MB")

    torch.save(policy.state_dict(), out / "fsdp_dpo.pt")
    with open(out / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"  Saved to {out}")
    return history
