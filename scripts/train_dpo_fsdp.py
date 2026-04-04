#!/usr/bin/env python3
"""
Extension 12: DPO with FSDP (policy sharded, reference frozen in bf16).

Standard run (policy FULL_SHARD, reference unsharded bf16):
  python scripts/train_dpo_fsdp.py --sft_checkpoint checkpoints/sft_fsdp/fsdp_sft.pt

Reference on CPU (Strategy C — maximum GPU memory savings, slower):
  python scripts/train_dpo_fsdp.py --ref_cpu_offload

Multi-GPU (4 GPUs):
  torchrun --nproc_per_node=4 scripts/train_dpo_fsdp.py --sharding FULL_SHARD
"""

import argparse
import sys

sys.path.insert(0, ".")

from src.training.fsdp_dpo import FSDPDPOConfig, train_dpo_fsdp


def main():
    parser = argparse.ArgumentParser(description="FSDP DPO training (Extension 12)")
    parser.add_argument("--sft_checkpoint", default="checkpoints/sft_fsdp/fsdp_sft.pt")
    parser.add_argument("--model",          default="gpt2-medium")
    parser.add_argument("--num_samples",    type=int, default=5000)
    parser.add_argument("--batch_size",     type=int, default=1)
    parser.add_argument("--epochs",         type=int, default=1)
    parser.add_argument("--beta",           type=float, default=0.1)
    parser.add_argument("--sharding",       default="FULL_SHARD",
                        choices=["NO_SHARD", "SHARD_GRAD_OP", "FULL_SHARD"])
    parser.add_argument("--ref_cpu_offload", action="store_true",
                        help="Offload reference model to CPU (slower, saves GPU memory)")
    parser.add_argument("--grad_ckpt",      action="store_true")
    parser.add_argument("--simulate_gpus",  type=int, default=1)
    parser.add_argument("--output_dir",     default="checkpoints/dpo_fsdp")
    args = parser.parse_args()

    cfg = FSDPDPOConfig(
        sft_checkpoint=args.sft_checkpoint,
        model_id=args.model,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        beta=args.beta,
        sharding_strategy=args.sharding,
        use_activation_checkpointing=args.grad_ckpt,
        ref_cpu_offload=args.ref_cpu_offload,
        simulate_num_gpus=args.simulate_gpus,
        output_dir=args.output_dir,
    )
    history = train_dpo_fsdp(cfg)

    print(f"\nFinal DPO loss:    {history['train_loss'][-1]:.4f}")
    print(f"Final reward margin: {history['reward_margin'][-1]:.4f}")
    if history["peak_memory_mb"]:
        print(f"Peak memory:       {max(history['peak_memory_mb']):.0f} MB")


if __name__ == "__main__":
    main()
