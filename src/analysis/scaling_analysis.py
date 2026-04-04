"""
Extension 12: Memory and compute scaling analysis for distributed transformer training.

Derives per-GPU memory requirements across model scales, sharding strategies,
and optimisation configurations (fp32, bf16 mixed, FSDP ZeRO stages, LoRA, pipeline parallelism).

Key memory formulas (fp32 Adam, N-GPU FSDP FULL_SHARD):
  BF16 weights            : 2P bytes
  FP32 gradients          : 4P bytes
  FP32 master weights     : 4P bytes        (mixed-precision optimizer copy)
  Adam optimizer (m+v)    : 8P bytes        (two fp32 states per param)
  ─────────────────────────────────────────
  Total per GPU (DDP)     : 18P bytes       (bf16 fwd, fp32 optim — often cited as 16P)
  Total per GPU (ZeRO-3)  : ceil(18P/N) bytes

  Activation memory per transformer layer (bf16, Megatron-LM formula):
    A_layer = 12 × B × S × H bytes          (without checkpointing)
    A_ckpt  =  2 × B × S × H bytes          (with per-block gradient checkpointing)

References:
  Rajbhandari et al. (2020) "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"
    https://arxiv.org/abs/1910.02054
  Shoeybi et al. (2019) "Megatron-LM: Training Multi-Billion Parameter Language Models"
    https://arxiv.org/abs/1909.08053
  Korthikanti et al. (2022) "Reducing Activation Recomputation in Large Transformer Models"
    https://arxiv.org/abs/2205.05198
  Hu et al. (2021) "LoRA: Low-Rank Adaptation of Large Language Models"
    https://arxiv.org/abs/2106.09685
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Model specifications
# ---------------------------------------------------------------------------

@dataclass
class ModelSpec:
    name: str
    num_params: int              # total trainable parameters
    hidden_dim: int              # d_model
    num_layers: int              # number of transformer decoder blocks
    num_heads: int               # multi-head attention heads
    vocab_size: int
    seq_len: int = 512           # typical training sequence length
    intermediate_dim: Optional[int] = None   # FFN intermediate; default 4 × hidden_dim

    def __post_init__(self):
        if self.intermediate_dim is None:
            self.intermediate_dim = 4 * self.hidden_dim


# ---------------------------------------------------------------------------
# Memory breakdown
# ---------------------------------------------------------------------------

@dataclass
class MemoryBreakdown:
    # Static memory (model state)
    weights_bf16_gb: float        # bf16 forward weights
    weights_fp32_gb: float        # fp32 weights (full-precision baseline)
    gradients_gb: float           # fp32 gradients
    optimizer_state_gb: float     # Adam fp32 first + second moment
    master_weights_gb: float      # fp32 master copy for mixed-precision optimizer update

    # Total static
    total_ddp_fp32_gb: float      # full fp32, no sharding  (16 bytes/param)
    total_ddp_mixed_gb: float     # bf16 fwd + fp32 optim  (18 bytes/param)

    # Activation memory (per layer, per batch)
    activations_per_layer_gb: float         # without checkpointing
    activations_full_gb: float              # all layers, no checkpointing
    activations_checkpointed_gb: float      # with per-block gradient checkpointing

    # Peak totals (static + activations)
    peak_fp32_gb: float           # fp32, no checkpointing
    peak_mixed_gb: float          # bf16 mixed, no checkpointing
    peak_mixed_ckpt_gb: float     # bf16 mixed, with checkpointing


def compute_memory_breakdown(
    spec: ModelSpec,
    batch_size: int = 1,
    bf16_bytes: int = 2,
    fp32_bytes: int = 4,
) -> MemoryBreakdown:
    """
    Compute per-GPU memory breakdown for single-device (DDP-equivalent) training.

    Activation formula: Megatron-LM Appendix B
      - Without checkpointing: 12 * B * S * H bytes per layer (bf16)
      - With checkpointing   :  2 * B * S * H bytes per layer
        (only block inputs/outputs stored; internals recomputed on backward)
    """
    P = spec.num_params
    L = spec.num_layers
    H = spec.hidden_dim
    S = spec.seq_len
    B = batch_size
    GB = 1e9

    weights_bf16     = P * bf16_bytes / GB
    weights_fp32     = P * fp32_bytes / GB
    gradients        = P * fp32_bytes / GB        # fp32 grads (numerical stability)
    master_weights   = P * fp32_bytes / GB        # fp32 copy for optimizer update
    optimizer_state  = P * 8 / GB                 # Adam: fp32 m + fp32 v = 8 bytes

    # Standard full-fp32 DDP: weights + grads + optim = 16 bytes/param
    total_ddp_fp32  = weights_fp32 + gradients + optimizer_state
    # bf16 mixed: bf16 weights + fp32 grads + fp32 master + fp32 optim = 18 bytes/param
    total_ddp_mixed = weights_bf16 + gradients + master_weights + optimizer_state

    # Activations
    act_per_layer_no_ckpt = 12 * B * S * H * bf16_bytes / GB
    act_per_layer_ckpt    =  2 * B * S * H * bf16_bytes / GB
    act_full              = act_per_layer_no_ckpt * L
    act_ckpt              = act_per_layer_ckpt    * L

    return MemoryBreakdown(
        weights_bf16_gb=weights_bf16,
        weights_fp32_gb=weights_fp32,
        gradients_gb=gradients,
        optimizer_state_gb=optimizer_state,
        master_weights_gb=master_weights,
        total_ddp_fp32_gb=total_ddp_fp32,
        total_ddp_mixed_gb=total_ddp_mixed,
        activations_per_layer_gb=act_per_layer_no_ckpt,
        activations_full_gb=act_full,
        activations_checkpointed_gb=act_ckpt,
        peak_fp32_gb=total_ddp_fp32 + act_full,
        peak_mixed_gb=total_ddp_mixed + act_full,
        peak_mixed_ckpt_gb=total_ddp_mixed + act_ckpt,
    )


# ---------------------------------------------------------------------------
# FSDP / ZeRO sharding
# ---------------------------------------------------------------------------

def compute_fsdp_per_gpu(
    breakdown: MemoryBreakdown,
    num_gpus: int,
    sharding_strategy: str = "FULL_SHARD",
    use_checkpointing: bool = True,
) -> Dict:
    """
    Estimate per-GPU peak memory with FSDP sharding.

    PyTorch FSDP sharding strategies map directly to ZeRO stages:
      NO_SHARD      ≡ ZeRO-0 / DDP    — all state replicated
      SHARD_GRAD_OP ≡ ZeRO-2          — shard gradients + optimizer; replicate params
      FULL_SHARD    ≡ ZeRO-3          — shard params + gradients + optimizer

    Activations are NOT sharded — each GPU processes its own micro-batch independently.
    During all-gather in FULL_SHARD, one transformer block's params are unshard at a time
    (FSDP's backward_prefetch avoids holding two blocks simultaneously).
    We model this as: stored shards = 1/N, peak includes one gathered block.
    """
    N   = max(1, num_gpus)
    W   = breakdown.weights_bf16_gb
    G   = breakdown.gradients_gb
    O   = breakdown.optimizer_state_gb
    M   = breakdown.master_weights_gb
    act = breakdown.activations_checkpointed_gb if use_checkpointing \
          else breakdown.activations_full_gb

    if sharding_strategy == "NO_SHARD":
        params_gpu = W
        grad_gpu   = G
        optim_gpu  = O + M
    elif sharding_strategy == "SHARD_GRAD_OP":
        # ZeRO-2: params replicated, grad+optim sharded
        params_gpu = W
        grad_gpu   = G / N
        optim_gpu  = (O + M) / N
    elif sharding_strategy == "FULL_SHARD":
        # ZeRO-3: all state sharded; params gathered one block at a time
        params_gpu = W / N
        grad_gpu   = G / N
        optim_gpu  = (O + M) / N
    else:
        raise ValueError(f"Unknown sharding_strategy: {sharding_strategy!r}")

    total_static  = params_gpu + grad_gpu + optim_gpu
    total_peak    = total_static + act
    savings_ratio = (W + G + O + M) / total_static if total_static > 0 else 1.0

    return {
        "strategy":           sharding_strategy,
        "num_gpus":           N,
        "params_gb":          params_gpu,
        "gradients_gb":       grad_gpu,
        "optimizer_gb":       optim_gpu,
        "total_static_gb":    total_static,
        "activations_gb":     act,
        "total_peak_gb":      total_peak,
        "savings_vs_ddp":     savings_ratio,
    }


# ---------------------------------------------------------------------------
# LoRA adapter memory savings
# ---------------------------------------------------------------------------

def lora_memory_savings(
    spec: ModelSpec,
    lora_r: int = 16,
    target_modules_per_layer: int = 4,   # q, k, v, o projections
    bf16_bytes: int = 2,
) -> Dict:
    """
    Estimate memory savings from LoRA vs full fine-tuning.

    LoRA adds A ∈ R^(r×k) and B ∈ R^(d×r) to each target weight W ∈ R^(d×k).
    Only A and B are trained; W is frozen. For attention projections d=k=H.

    Full fine-tune optimizer memory = 12 bytes/param (fp32 grad + m + v)
    LoRA optimizer memory           = 12 bytes × lora_params only
    Frozen weights memory           = 2 bytes/param (bf16, no grad)
    """
    H = spec.hidden_dim
    P = spec.num_params
    GB = 1e9

    lora_params_per_layer  = target_modules_per_layer * lora_r * (H + H)
    total_lora_params      = lora_params_per_layer * spec.num_layers

    full_ft_optim_gb  = P * 12 / GB              # grad + m + v in fp32
    lora_optim_gb     = total_lora_params * 12 / GB
    frozen_weights_gb = P * bf16_bytes / GB       # frozen weights always in memory

    return {
        "model_params":           P,
        "lora_trainable_params":  total_lora_params,
        "full_ft_trainable":      P,
        "trainable_ratio":        total_lora_params / P,
        "frozen_weights_gb":      frozen_weights_gb,
        "full_ft_optim_gb":       full_ft_optim_gb,
        "lora_optim_gb":          lora_optim_gb,
        "optim_savings_gb":       full_ft_optim_gb - lora_optim_gb,
        "optim_savings_ratio":    1 - lora_optim_gb / full_ft_optim_gb,
        "total_full_ft_gb":       frozen_weights_gb + full_ft_optim_gb,
        "total_lora_gb":          frozen_weights_gb + lora_optim_gb,
    }


# ---------------------------------------------------------------------------
# Pipeline parallelism
# ---------------------------------------------------------------------------

def pipeline_stages(
    spec: ModelSpec,
    num_stages: int,
    micro_batch_size: int = 1,
    num_micro_batches: int = 4,
    bf16_bytes: int = 2,
) -> Dict:
    """
    Estimate per-stage memory under GPipe-style pipeline parallelism.

    Pipeline bubble efficiency = 1 - (p-1)/(p+m-1)
    where p = num_stages, m = num_micro_batches.

    Each stage stores activations for all in-flight micro-batches at its layer
    boundary (the "bubble"). We use per-block gradient checkpointing within stages.
    """
    layers_per_stage   = math.ceil(spec.num_layers / num_stages)
    params_per_stage   = spec.num_params / num_stages
    optim_per_stage_gb = params_per_stage * 12 / 1e9   # fp32 grad + m + v

    # Activations at stage boundaries: num_micro_batches × B × S × H (bf16)
    act_boundary_gb = (num_micro_batches * micro_batch_size *
                       spec.seq_len * spec.hidden_dim * bf16_bytes / 1e9)
    # Internal activations with checkpointing
    act_internal_gb = (layers_per_stage * 2 * micro_batch_size *
                       spec.seq_len * spec.hidden_dim * bf16_bytes / 1e9)

    total_per_stage_gb = (params_per_stage * bf16_bytes / 1e9 +
                          optim_per_stage_gb +
                          act_boundary_gb +
                          act_internal_gb)

    efficiency = 1 - (num_stages - 1) / (num_stages + num_micro_batches - 1)

    return {
        "num_stages":           num_stages,
        "layers_per_stage":     layers_per_stage,
        "params_per_stage":     params_per_stage,
        "params_per_stage_gb":  params_per_stage * bf16_bytes / 1e9,
        "optim_per_stage_gb":   optim_per_stage_gb,
        "act_boundary_gb":      act_boundary_gb,
        "act_internal_gb":      act_internal_gb,
        "total_per_stage_gb":   total_per_stage_gb,
        "pipeline_efficiency":  efficiency,
        "bubble_fraction":      1 - efficiency,
    }


# ---------------------------------------------------------------------------
# Tensor parallelism (Megatron-style)
# ---------------------------------------------------------------------------

def tensor_parallel_memory(
    spec: ModelSpec,
    tp_degree: int,
    dp_degree: int = 1,
    bf16_bytes: int = 2,
) -> Dict:
    """
    Estimate per-GPU memory with Megatron-style tensor parallelism.

    In TP, each GPU holds 1/tp_degree of each weight matrix column/row.
    Optimizer state and gradients are correspondingly sharded.
    Communication: all-reduce (2 × B × S × H bytes) at each attention + FFN.
    """
    P = spec.num_params
    H = spec.hidden_dim
    GB = 1e9

    # Weight sharding: each GPU holds 1/tp of params
    weights_gpu_gb   = P * bf16_bytes / (GB * tp_degree)
    optim_gpu_gb     = P * 12 / (GB * tp_degree)          # grad + m + v

    # All-reduce volume per layer: 2 passes × B × S × H (forward) + same (backward)
    comm_per_layer_gb = 4 * spec.seq_len * H * bf16_bytes / GB  # per batch element

    return {
        "tp_degree":          tp_degree,
        "dp_degree":          dp_degree,
        "total_gpus":         tp_degree * dp_degree,
        "weights_per_gpu_gb": weights_gpu_gb,
        "optim_per_gpu_gb":   optim_gpu_gb,
        "total_static_gb":    weights_gpu_gb + optim_gpu_gb,
        "comm_per_layer_gb":  comm_per_layer_gb,
        "tp_savings_ratio":   tp_degree,
    }


# ---------------------------------------------------------------------------
# Benchmark model catalogue
# ---------------------------------------------------------------------------

BENCHMARK_MODELS: List[ModelSpec] = [
    ModelSpec("GPT-2-small",    117_000_000,  768, 12, 12, 50257),
    ModelSpec("GPT-2-medium",   355_000_000, 1024, 24, 16, 50257),
    ModelSpec("GPT-2-large",    774_000_000, 1280, 36, 20, 50257),
    ModelSpec("GPT-2-XL",     1_542_000_000, 1600, 48, 25, 50257),
    ModelSpec("LLaMA-7B",     6_738_000_000, 4096, 32, 32, 32000, intermediate_dim=11008),
    ModelSpec("LLaMA-13B",   13_016_000_000, 5120, 40, 40, 32000, intermediate_dim=13824),
    ModelSpec("LLaMA-70B",   69_510_000_000, 8192, 80, 64, 32000, intermediate_dim=28672),
]


def format_memory_table(
    models: List[ModelSpec] = None,
    batch_size: int = 1,
    gpu_counts: List[int] = (1, 2, 4, 8),
) -> str:
    if models is None:
        models = BENCHMARK_MODELS

    lines = []
    hdr = (f"{'Model':<16} {'Params':>9} {'BF16 Wt':>8} {'DDP Total':>10} "
           + "  ".join(f"{n}×GPU" for n in gpu_counts)
           + f"  {'Min A100s':>9}")
    lines.append(hdr)
    lines.append("─" * len(hdr))

    for spec in models:
        bd = compute_memory_breakdown(spec, batch_size=batch_size)
        gpu_cols = []
        for n in gpu_counts:
            r = compute_fsdp_per_gpu(bd, n, "FULL_SHARD", use_checkpointing=True)
            gpu_cols.append(f"{r['total_peak_gb']:>5.1f}G")

        # Minimum 80 GB A100s needed
        min_gpus = 1
        for n in [1, 2, 4, 8, 16, 32]:
            r = compute_fsdp_per_gpu(bd, n, "FULL_SHARD", use_checkpointing=True)
            min_gpus = n
            if r["total_peak_gb"] < 72:
                break

        lines.append(
            f"{spec.name:<16} {spec.num_params/1e9:>8.2f}B "
            f"{bd.weights_bf16_gb:>7.1f}G "
            f"{bd.total_ddp_mixed_gb:>9.1f}G "
            + "  ".join(gpu_cols)
            + f"  {'≥'+str(min_gpus)+'×A100':>9}"
        )

    return "\n".join(lines)
