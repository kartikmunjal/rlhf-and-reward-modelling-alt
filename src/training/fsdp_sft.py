"""
Extension 12: FSDP-wrapped SFT training.

PyTorch FSDP sharding strategies (ZeRO equivalences):
  NO_SHARD      ≡ ZeRO-0/DDP — API validation on single GPU; no memory benefit
  SHARD_GRAD_OP ≡ ZeRO-2     — shard gradients + optimizer state; replicate params
  FULL_SHARD    ≡ ZeRO-3     — shard params + gradients + optimizer state

Single-GPU simulation:
  Run with ShardingStrategy.NO_SHARD and gradient_accumulation_steps = N
  to simulate N-GPU effective batch sizes. Code is multi-GPU-ready; swap in
  torch.distributed.init_process_group() and update world_size to deploy.

Design notes:
  1. FSDP wrap happens BEFORE optimizer creation — FSDP.parameters() differ
     from base model parameters after wrapping (ShardedTensor handles).
  2. auto_wrap_policy shards at GPT-2Block boundaries — each block is one
     FSDP unit, so all-gather cost scales with block size, not full model.
  3. backward_prefetch=BACKWARD_PRE: overlap next-block all-gather with
     current-block backward computation — reduces idle GPU time by ~15%.
  4. Checkpoint saving: FSDP.state_dict() on multi-GPU requires
     StateDictType.FULL_STATE_DICT to consolidate shards onto rank 0 before
     saving. On single GPU, standard state_dict() works directly.
"""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.distributed.fsdp import (
    BackwardPrefetch,
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class FSDPSFTConfig:
    model_id: str = "gpt2-medium"
    num_samples: int = 5000
    max_length: int = 512
    batch_size: int = 2                    # per-GPU (or per-step before grad accum)
    gradient_accumulation_steps: int = 8   # effective_batch = batch_size × accum
    num_epochs: int = 2
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.05
    output_dir: str = "checkpoints/sft_fsdp"

    # FSDP settings
    sharding_strategy: str = "FULL_SHARD"        # NO_SHARD | SHARD_GRAD_OP | FULL_SHARD
    use_mixed_precision: bool = True              # bf16 fwd/bwd, fp32 optimizer
    use_cpu_offload: bool = False                 # offload optimizer state to CPU
    use_activation_checkpointing: bool = True     # per-block gradient checkpointing
    min_params_for_wrap: int = 1_000_000          # size threshold for size_based policy

    # Simulation
    simulate_num_gpus: int = 1    # multiply accum_steps to simulate N-GPU batch
    profile_memory: bool = True


# ---------------------------------------------------------------------------
# FSDP utilities
# ---------------------------------------------------------------------------

_STRATEGY_MAP = {
    "NO_SHARD":      ShardingStrategy.NO_SHARD,
    "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
    "FULL_SHARD":    ShardingStrategy.FULL_SHARD,
}


def get_gpt2_wrap_policy(model: nn.Module):
    """Shard at GPT-2 transformer block boundaries for optimal communication granularity."""
    try:
        from transformers.models.gpt2.modeling_gpt2 import GPT2Block
        return functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={GPT2Block},
        )
    except ImportError:
        return functools.partial(
            size_based_auto_wrap_policy,
            min_num_params=1_000_000,
        )


def bf16_mixed_precision() -> MixedPrecision:
    """
    Standard mixed-precision policy:
      - param_dtype=bf16: forward pass in bf16 (halves bandwidth, faster matmuls)
      - reduce_dtype=bf16: gradient all-reduce in bf16 (halves communication volume)
      - buffer_dtype=fp32: layer norms, embeddings stay in fp32 (numerical stability)
    """
    return MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.float32,
    )


def wrap_model_with_fsdp(model: nn.Module, cfg: FSDPSFTConfig) -> FSDP:
    """
    Wrap a GPT-2 model with FSDP.

    Key parameter: backward_prefetch=BACKWARD_PRE
    While computing gradients for layer i, FSDP pre-fetches (all-gathers) the
    parameters for layer i-1 so they are ready when needed. This overlaps
    compute and communication on GPUs with NVLink or fast PCIe interconnect.
    """
    mp_policy  = bf16_mixed_precision() if cfg.use_mixed_precision else None
    cpu_off    = CPUOffload(offload_params=True) if cfg.use_cpu_offload else None

    wrapped = FSDP(
        model,
        sharding_strategy=_STRATEGY_MAP[cfg.sharding_strategy],
        auto_wrap_policy=get_gpt2_wrap_policy(model),
        mixed_precision=mp_policy,
        cpu_offload=cpu_off,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=torch.cuda.current_device() if torch.cuda.is_available() else None,
        sync_module_states=False,  # set True when launching with torchrun
    )
    return wrapped


def apply_activation_checkpointing(model: FSDP) -> None:
    """
    Apply gradient checkpointing at each GPT-2Block.

    Trade-off: saves ~6× activation memory at cost of one extra forward pass
    per block during backward. At B=2, S=512, H=1024, 24 layers:
      Without: 24 × 12 × 2 × 512 × 1024 × 2 bytes ≈ 600 MB
      With:    24 ×  2 × 2 × 512 × 1024 × 2 bytes ≈ 100 MB
    """
    try:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointImpl,
            apply_activation_checkpointing as _apply,
            checkpoint_wrapper,
        )
        from transformers.models.gpt2.modeling_gpt2 import GPT2Block

        non_reentrant_wrapper = functools.partial(
            checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
        )
        _apply(model, checkpoint_wrapper_fn=non_reentrant_wrapper,
               check_fn=lambda m: isinstance(m, GPT2Block))
    except (ImportError, AttributeError):
        # Fallback to HuggingFace gradient checkpointing
        base = model.module if hasattr(model, "module") else model
        if hasattr(base, "gradient_checkpointing_enable"):
            base.gradient_checkpointing_enable()


def memory_stats_mb(device=None) -> dict:
    """Return current and peak GPU memory in MB."""
    if not torch.cuda.is_available():
        return {"allocated_mb": 0, "reserved_mb": 0, "peak_mb": 0}
    dev = device or torch.cuda.current_device()
    return {
        "allocated_mb": torch.cuda.memory_allocated(dev) / 1e6,
        "reserved_mb":  torch.cuda.memory_reserved(dev) / 1e6,
        "peak_mb":      torch.cuda.max_memory_allocated(dev) / 1e6,
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_sft_fsdp(cfg: FSDPSFTConfig) -> dict:
    """
    FSDP-wrapped SFT training. Key differences from base train_sft():

    1. Model is wrapped with FSDP *before* optimizer creation. Creating the
       optimizer on the unwrapped model then wrapping causes parameter mismatch
       because FSDP replaces parameter storage with ShardedTensors.

    2. Gradient accumulation steps are multiplied by simulate_num_gpus so a
       single GPU can produce the same effective batch size as a multi-GPU run.

    3. Checkpoint consolidation: on true multi-GPU, FSDP shards are spread
       across devices. StateDictType.FULL_STATE_DICT gathers them onto rank 0
       before saving. On single GPU, standard state_dict() works directly.

    Memory profile logged each epoch for comparison across sharding strategies.
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

    # Effective gradient accumulation accounts for simulated GPU count
    eff_accum = cfg.gradient_accumulation_steps * cfg.simulate_num_gpus
    eff_batch = cfg.batch_size * eff_accum
    print(f"[FSDP SFT] strategy={cfg.sharding_strategy} | "
          f"simulate_gpus={cfg.simulate_num_gpus} | eff_batch={eff_batch}")

    # --- Model ---
    tokenizer = GPT2Tokenizer.from_pretrained(cfg.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(cfg.model_id)
    model.resize_token_embeddings(len(tokenizer))
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  {cfg.model_id}: {num_params/1e6:.1f}M params")

    if cfg.profile_memory:
        torch.cuda.reset_peak_memory_stats()
        pre_wrap = memory_stats_mb(device)

    # --- FSDP wrap (must happen before optimizer) ---
    model = wrap_model_with_fsdp(model, cfg)
    if cfg.use_activation_checkpointing:
        apply_activation_checkpointing(model)
    model.to(device)

    if cfg.profile_memory:
        post_wrap = memory_stats_mb(device)
        print(f"  Memory pre-wrap:  {pre_wrap['allocated_mb']:.0f} MB")
        print(f"  Memory post-wrap: {post_wrap['allocated_mb']:.0f} MB  "
              f"(FSDP overhead: {post_wrap['allocated_mb'] - pre_wrap['allocated_mb']:.0f} MB)")

    # --- Dataset ---
    ds = load_dataset("Anthropic/hh-rlhf", split="train")
    if cfg.num_samples < len(ds):
        ds = ds.select(range(cfg.num_samples))

    def tokenize(batch):
        enc = tokenizer(
            batch["chosen"], truncation=True, max_length=cfg.max_length,
            padding="max_length", return_tensors="pt",
        )
        enc["labels"] = enc["input_ids"].clone()
        enc["labels"][enc["attention_mask"] == 0] = -100
        return enc

    ds = ds.map(tokenize, batched=True, remove_columns=ds.column_names)
    ds.set_format("torch")
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    # --- Optimizer (created AFTER FSDP wrap) ---
    optimizer    = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=0.01)
    total_steps  = (len(loader) // eff_accum) * cfg.num_epochs
    warmup_steps = max(1, int(total_steps * cfg.warmup_ratio))
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    history = {"train_loss": [], "peak_memory_mb": [], "config": cfg.__dict__.copy()}
    optimizer.zero_grad()

    for epoch in range(cfg.num_epochs):
        model.train()
        torch.cuda.reset_peak_memory_stats()
        epoch_loss = 0.0

        for step, batch in enumerate(loader):
            batch   = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss    = outputs.loss / eff_accum
            loss.backward()
            epoch_loss += outputs.loss.item()

            if (step + 1) % eff_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_loss = epoch_loss / len(loader)
        peak_mb  = memory_stats_mb(device)["peak_mb"]
        history["train_loss"].append(avg_loss)
        history["peak_memory_mb"].append(peak_mb)
        print(f"  Epoch {epoch+1}/{cfg.num_epochs} | loss={avg_loss:.4f} | "
              f"peak_mem={peak_mb:.0f} MB")

    # --- Save checkpoint ---
    # On multi-GPU: use StateDictType.FULL_STATE_DICT (consolidates shards on rank 0)
    # On single GPU: standard state_dict works
    is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    if is_distributed:
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        cfg_sd = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg_sd):
            state = model.state_dict()
    else:
        state = model.state_dict()

    torch.save(state, out / "fsdp_sft.pt")
    with open(out / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"  Saved to {out}")
    return history
