#!/usr/bin/env python3
"""
Extension 12: Distributed training scaling analysis.

Prints memory requirements, GPU counts, LoRA savings, and 7B+ engineering
considerations across the full model scale range (GPT-2-small to LLaMA-70B).

Usage:
  python scripts/analyze_scaling.py                      # full summary table
  python scripts/analyze_scaling.py --model LLaMA-7B     # deep-dive on one model
  python scripts/analyze_scaling.py --lora               # LoRA savings table
  python scripts/analyze_scaling.py --pipeline           # pipeline parallelism table
  python scripts/analyze_scaling.py --what_scales_to 7b  # "what would change" checklist
"""

import argparse
import math
import sys

sys.path.insert(0, ".")

from src.analysis.scaling_analysis import (
    BENCHMARK_MODELS,
    ModelSpec,
    compute_fsdp_per_gpu,
    compute_memory_breakdown,
    lora_memory_savings,
    pipeline_stages,
    tensor_parallel_memory,
)


# ---------------------------------------------------------------------------
# Full memory table
# ---------------------------------------------------------------------------

def print_full_table(batch_size: int = 1):
    print("=" * 95)
    print(f"MEMORY SCALING TABLE (batch={batch_size}, seq=512, bf16 mixed, Adam, grad-checkpointing)")
    print("Columns: per-GPU peak memory with FSDP FULL_SHARD on N GPUs")
    print("=" * 95)
    print(f"{'Model':<16} {'Params':>9} {'BF16 Wt':>8} {'DDP Total':>10}"
          f"  {'1 GPU':>7}  {'2 GPU':>7}  {'4 GPU':>7}  {'8 GPU':>7}  {'Min A100(80G)':>13}")
    print("─" * 95)

    for spec in BENCHMARK_MODELS:
        bd = compute_memory_breakdown(spec, batch_size=batch_size)
        cols = []
        for n in [1, 2, 4, 8]:
            r = compute_fsdp_per_gpu(bd, n, "FULL_SHARD", use_checkpointing=True)
            cols.append(f"{r['total_peak_gb']:>6.1f}G")

        # Minimum A100s (80GB) needed
        min_gpus = 64  # start pessimistic
        for n in [1, 2, 4, 8, 16, 32, 64]:
            r = compute_fsdp_per_gpu(bd, n, "FULL_SHARD", use_checkpointing=True)
            if r["total_peak_gb"] < 72:
                min_gpus = n
                break

        ddp_total = bd.total_ddp_mixed_gb + bd.activations_checkpointed_gb
        print(f"{spec.name:<16} {spec.num_params/1e9:>8.2f}B "
              f"{bd.weights_bf16_gb:>7.1f}G {ddp_total:>9.1f}G"
              + "".join(f"  {c}" for c in cols)
              + f"  {'≥'+str(min_gpus)+'× A100':>13}")

    print()
    print("Notes:")
    print("  BF16 Wt   = weight-only memory (inference lower bound)")
    print("  DDP Total = DDP per-GPU peak (18 bytes/param + activations, no sharding)")
    print("  N GPU     = FSDP FULL_SHARD (ZeRO-3) per-GPU peak with gradient checkpointing")
    print("  Min A100  = minimum 80GB A100s with FSDP FULL_SHARD + gradient checkpointing")


# ---------------------------------------------------------------------------
# Single-model deep-dive
# ---------------------------------------------------------------------------

def print_deep_dive(model_name: str):
    spec = next((m for m in BENCHMARK_MODELS if m.name.lower() == model_name.lower()), None)
    if spec is None:
        print(f"Model '{model_name}' not found.")
        print(f"Available: {[m.name for m in BENCHMARK_MODELS]}")
        return

    bd = compute_memory_breakdown(spec, batch_size=1)
    print(f"\n{'='*70}")
    print(f"DEEP DIVE: {spec.name}  ({spec.num_params/1e9:.2f}B params)")
    print(f"Architecture: d_model={spec.hidden_dim}, layers={spec.num_layers}, "
          f"heads={spec.num_heads}, FFN={spec.intermediate_dim}")
    print(f"{'='*70}")

    # 1. Static memory breakdown
    print("\n1. STATIC MEMORY (per GPU, single device, bf16 mixed precision)")
    items = [
        ("BF16 weights",      bd.weights_bf16_gb),
        ("FP32 gradients",    bd.gradients_gb),
        ("FP32 master copy",  bd.master_weights_gb),
        ("Adam m+v (fp32)",   bd.optimizer_state_gb),
    ]
    for label, gb in items:
        print(f"   {label:<24}: {gb:7.2f} GB  ({gb*1024:.0f} MB)")
    total_mixed = sum(v for _, v in items)
    print(f"   {'─'*36}")
    print(f"   {'Total (mixed prec.)':<24}: {total_mixed:7.2f} GB  (~18 bytes/param)")

    # 2. Activation memory
    print(f"\n2. ACTIVATION MEMORY (B=1, S={spec.seq_len}, L={spec.num_layers}, H={spec.hidden_dim})")
    print(f"   Per layer, no ckpt  : {bd.activations_per_layer_gb*1000:7.1f} MB  (formula: 12×B×S×H bytes)")
    print(f"   All layers, no ckpt : {bd.activations_full_gb:7.3f} GB")
    print(f"   All layers, w/ ckpt : {bd.activations_checkpointed_gb:7.3f} GB  (6× reduction)")
    print(f"   Checkpointing trade-off: ~30% increase in compute; saves "
          f"{(bd.activations_full_gb - bd.activations_checkpointed_gb):.2f} GB")

    # 3. FSDP sharding comparison
    print(f"\n3. FSDP SHARDING (bf16 mixed, gradient checkpointing)")
    print(f"   {'Strategy':<14} {'GPUs':>4} {'Static/GPU':>11} {'Activations':>12} "
          f"{'Peak/GPU':>10} {'Fits 80GB?':>11} {'vs DDP':>8}")
    print(f"   {'─'*14} {'─'*4} {'─'*11} {'─'*12} {'─'*10} {'─'*11} {'─'*8}")
    for strat in ["NO_SHARD", "SHARD_GRAD_OP", "FULL_SHARD"]:
        for n in ([1] if strat == "NO_SHARD" else [1, 2, 4, 8]):
            r     = compute_fsdp_per_gpu(bd, n, strat, use_checkpointing=True)
            fits  = "✓" if r["total_peak_gb"] < 72 else "✗"
            ratio = r["savings_vs_ddp"]
            print(f"   {strat:<14} {n:>4} {r['total_static_gb']:>10.2f}G "
                  f"{r['activations_gb']:>11.3f}G {r['total_peak_gb']:>9.2f}G "
                  f"{fits:>11} {ratio:>6.1f}×")

    # 4. LoRA savings
    print(f"\n4. LoRA OPTIMIZER STATE SAVINGS (vs full fine-tuning)")
    print(f"   {'Rank':>6} {'Trainable':>12} {'% of P':>8} "
          f"{'Full FT Optim':>14} {'LoRA Optim':>11} {'Savings':>8}")
    print(f"   {'─'*6} {'─'*12} {'─'*8} {'─'*14} {'─'*11} {'─'*8}")
    for r in [8, 16, 32, 64]:
        lo = lora_memory_savings(spec, lora_r=r)
        print(f"   {r:>6} {lo['lora_trainable_params']/1e6:>10.1f}M "
              f"{lo['trainable_ratio']*100:>7.3f}% "
              f"{lo['full_ft_optim_gb']:>13.2f}G "
              f"{lo['lora_optim_gb']:>10.3f}G "
              f"{lo['optim_savings_ratio']*100:>6.1f}%")

    # 5. Pipeline parallelism
    print(f"\n5. PIPELINE PARALLELISM (GPipe, 4 micro-batches, B=1)")
    print(f"   {'Stages':>6} {'Layers/Stage':>13} {'Params/Stage':>13} "
          f"{'Mem/Stage':>10} {'PP Efficiency':>14}")
    print(f"   {'─'*6} {'─'*13} {'─'*13} {'─'*10} {'─'*14}")
    for s in [2, 4, 8]:
        if s <= spec.num_layers:
            pp = pipeline_stages(spec, num_stages=s, micro_batch_size=1, num_micro_batches=4)
            print(f"   {s:>6} {pp['layers_per_stage']:>13} "
                  f"{pp['params_per_stage']/1e9:>11.2f}B "
                  f"{pp['total_per_stage_gb']:>9.2f}G "
                  f"{pp['pipeline_efficiency']*100:>12.0f}%  "
                  f"(bubble: {pp['bubble_fraction']*100:.0f}%)")

    # 6. Tensor parallelism
    print(f"\n6. TENSOR PARALLELISM (Megatron-style, no pipeline, B=1)")
    print(f"   {'TP Degree':>9} {'Weights/GPU':>12} {'Optim/GPU':>10} {'Total/GPU':>10}")
    print(f"   {'─'*9} {'─'*12} {'─'*10} {'─'*10}")
    for tp in [1, 2, 4, 8]:
        t = tensor_parallel_memory(spec, tp_degree=tp)
        print(f"   {tp:>9} {t['weights_per_gpu_gb']:>11.2f}G "
              f"{t['optim_per_gpu_gb']:>9.2f}G "
              f"{t['total_static_gb']:>9.2f}G")

    # 7. Recommendation
    print(f"\n7. RECOMMENDED TRAINING CONFIGURATION")
    # Find minimum FSDP config that fits on A100
    rec = None
    for n in [1, 2, 4, 8, 16]:
        r = compute_fsdp_per_gpu(bd, n, "FULL_SHARD", use_checkpointing=True)
        if r["total_peak_gb"] < 72:
            rec = (n, r)
            break

    if rec:
        n, r = rec
        lo16 = lora_memory_savings(spec, lora_r=16)
        print(f"   Data parallel: {n}× A100 (80GB) with FSDP FULL_SHARD + grad-checkpointing")
        print(f"   → {r['total_peak_gb']:.1f} GB/GPU  (leaves {72-r['total_peak_gb']:.1f} GB headroom)")
        print(f"   LoRA r=16 reduces optimizer state by {lo16['optim_savings_gb']:.1f} GB "
              f"({lo16['optim_savings_ratio']*100:.0f}%) — highly recommended")
        if spec.num_params > 10e9:
            print(f"   For >10B: add tensor parallelism TP=4 to halve per-GPU weight size")
            print(f"   Full 3D parallelism: DP × TP × PP = 8 × 4 × 2 = 64 GPUs (LLM production)")
    else:
        print(f"   Requires 3D parallelism (DP + TP + PP) on 64+ A100s")
        print(f"   Recommended: FSDP FULL_SHARD (ZeRO-3) + TP=8 + PP=4 + LoRA r=16")


# ---------------------------------------------------------------------------
# LoRA comparison table
# ---------------------------------------------------------------------------

def print_lora_table():
    print("=" * 75)
    print("LoRA OPTIMIZER STATE SAVINGS (r=16, 4 attention projections/layer)")
    print("Optimizer memory = fp32 gradients + Adam m + Adam v = 12 bytes/param")
    print("=" * 75)
    print(f"{'Model':<16} {'Params':>9} {'Full FT Optim':>14} "
          f"{'LoRA Optim':>11} {'Savings':>10} {'Ratio':>8}")
    print("─" * 75)
    for spec in BENCHMARK_MODELS:
        lo = lora_memory_savings(spec, lora_r=16)
        print(f"{spec.name:<16} {spec.num_params/1e9:>8.2f}B "
              f"{lo['full_ft_optim_gb']:>13.2f}G "
              f"{lo['lora_optim_gb']:>10.4f}G "
              f"{lo['optim_savings_gb']:>9.2f}G "
              f"{lo['optim_savings_ratio']*100:>7.1f}%")
    print()
    print("Key insight: at 7B+, LoRA r=16 reduces optimizer state by >99.9%.")
    print("Full fine-tuning a 7B model requires 80 GB for optimizer state alone.")
    print("LoRA reduces this to <0.1 GB — the difference between needing 8×A100s vs 1×A100.")


# ---------------------------------------------------------------------------
# "What would change" engineering checklist
# ---------------------------------------------------------------------------

def print_what_scales():
    print("=" * 80)
    print("ENGINEERING CHECKLIST: Scaling the RLHF Pipeline to 7B+")
    print("=" * 80)

    items = [
        ("Model loading",
         "GPT-2-medium fits in 0.7 GB bf16. LLaMA-7B is 13.4 GB.\n"
         "    Use `from_pretrained(..., torch_dtype=torch.bfloat16)` to avoid\n"
         "    materialising fp32 weights. For 70B, use device_map='auto' (Accelerate)\n"
         "    or load sharded checkpoints with HuggingFace SafeTensors."),

        ("FSDP wrapping",
         "At 7B: FULL_SHARD on 4× A100 → 3.4 GB params/GPU + optim.\n"
         "    auto_wrap_policy must match the target architecture's block class\n"
         "    (LlamaDecoderLayer for LLaMA, GPT2Block for GPT-2).\n"
         "    Wrong wrap policy → FSDP wraps the whole model as one unit → no savings."),

        ("Optimizer state",
         "7B full fine-tuning: 6.7B × 12 bytes = 80 GB optimizer state alone.\n"
         "    Three solutions (pick one or combine):\n"
         "    a) FSDP FULL_SHARD: shards optimizer state N-way → 80/N GB per GPU\n"
         "    b) LoRA r=16: 99.9% optimizer savings; 80 GB → 0.06 GB\n"
         "    c) 8-bit Adam (bitsandbytes): fp32 state → int8 → 4× reduction"),

        ("Activation checkpointing",
         "7B, B=1, S=2048: activations without ckpt ≈ 32 × 12 × 1 × 2048 × 4096 × 2B ≈ 64 GB.\n"
         "    With per-block checkpointing: 32 ×  2 × 1 × 2048 × 4096 × 2B ≈ 11 GB.\n"
         "    Essential. Compute overhead: ~30% slower backward pass (one extra fwd/block)."),

        ("Sequence length scaling",
         "Attention score matrix is B × H × S × S in fp32 (for numerical stability).\n"
         "    At S=2048, H=32: 1 × 32 × 2048 × 2048 × 4B = 1.0 GB per layer.\n"
         "    At S=4096: 4.0 GB per layer. This is why Flash Attention 2 is non-optional\n"
         "    at 7B+: it computes attention in tiles without materialising the full matrix,\n"
         "    reducing S² memory to O(S)."),

        ("Gradient communication",
         "DDP all-reduce: 2P bytes per step (forward + backward accumulation).\n"
         "    At 7B: 13.4 GB per backward pass over the interconnect.\n"
         "    FSDP FULL_SHARD: all-gather per block (reduces peak communication to 1 block\n"
         "    at a time). Requires high-bandwidth interconnect (NVLink 600 GB/s vs PCIe 32 GB/s).\n"
         "    On multi-node: NCCL_IB_DISABLE=0 and InfiniBand are critical."),

        ("LoRA becomes essential (not optional)",
         "At 7B full fine-tuning, A100 memory split:\n"
         "    Weights (bf16): 13.4 GB | Gradients (fp32): 26.8 GB | Adam: 53.6 GB = 93.8 GB\n"
         "    Exceeds a single 80GB A100 before even loading one training example.\n"
         "    LoRA r=16 on 4 attention projections × 32 layers = 8.4M trainable params:\n"
         "    Optimizer state: 8.4M × 12B = 0.10 GB vs 53.6 GB full fine-tuning."),

        ("DPO memory doubles",
         "DPO holds policy + reference simultaneously. At 7B:\n"
         "    Reference (bf16, frozen): 13.4 GB per GPU (unsharded, Strategy A)\n"
         "    Policy (FSDP FULL_SHARD, 8 GPUs): 13.4 GB / 8 ≈ 1.7 GB + optim\n"
         "    Total policy+ref: ~15 GB/GPU on 8× A100 with LoRA on policy.\n"
         "    Without LoRA: 93.8/8 + 13.4 = 25.1 GB/GPU — still fits on A100."),

        ("Checkpoint saving",
         "FSDP shards are held on different GPUs. To save a consolidated checkpoint:\n"
         "    Use StateDictType.FULL_STATE_DICT with rank0_only=True (gathers to rank 0).\n"
         "    For 7B: requires 13.4 GB free CPU RAM on rank 0 for consolidation.\n"
         "    Alternative: StateDictType.LOCAL_STATE_DICT (saves per-rank shards,\n"
         "    requires matching world_size at load time)."),

        ("torchrun launch",
         "Multi-GPU requires init_process_group() before FSDP.wrap():\n"
         "    torch.distributed.init_process_group(backend='nccl')\n"
         "    rank = torch.distributed.get_rank()\n"
         "    world_size = torch.distributed.get_world_size()\n"
         "    torchrun --nproc_per_node=8 scripts/train_sft_fsdp.py\n"
         "    Current code: add these 3 lines and remove device_id hardcoding."),
    ]

    for i, (title, detail) in enumerate(items, 1):
        print(f"\n{i:2d}. {title}")
        for line in detail.split("\n"):
            print(f"    {line}")

    print("\n" + "─" * 80)
    print("TL;DR: The RLHF math is unchanged. The engineering changes are:")
    print("  • Wrap policy with FSDP FULL_SHARD (change auto_wrap_policy block class)")
    print("  • Enable LoRA r=16 on policy (99.9% optimizer savings)")
    print("  • Enable activation checkpointing (6× activation savings)")
    print("  • Use Flash Attention 2 (O(S) vs O(S²) attention memory)")
    print("  • Add torchrun launcher + init_process_group (3 lines)")
    print("  • Save checkpoints with FULL_STATE_DICT (1 line)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       default=None, help="Model name for deep-dive")
    parser.add_argument("--batch_size",  type=int, default=1)
    parser.add_argument("--lora",        action="store_true")
    parser.add_argument("--pipeline",    action="store_true")
    parser.add_argument("--what_scales", action="store_true",
                        help="Print 7B+ engineering checklist")
    args = parser.parse_args()

    if args.model:
        print_deep_dive(args.model)
    elif args.lora:
        print_lora_table()
    elif args.what_scales:
        print_what_scales()
    else:
        print_full_table(args.batch_size)
        print()
        print_lora_table()
        if not args.pipeline:
            print("\nFor 7B+ engineering checklist: --what_scales")
            print("For single-model deep-dive:    --model LLaMA-7B")


if __name__ == "__main__":
    main()
