"""
CLI script: Fine-tune GPT-2 on synthetic SFT data and compare to hh-rlhf.

This script trains three variants and produces a comparison table:
  A. SFT on hh-rlhf chosen responses  (human baseline)
  B. SFT on synthetic Claude data      (constitution-grounded)
  C. SFT on mixed 50/50               (human + synthetic)

Each model is then scored by the reward model on a held-out prompt set.
The reward model acts as a proxy for human preference.

Usage
-----
    # Train all three variants (requires reward model at checkpoints/reward)
    python scripts/train_sft_synthetic.py \
        --synthetic_data data/synthetic_sft.jsonl \
        --reward_checkpoint checkpoints/reward \
        --variants hh_rlhf synthetic mixed

    # Train only synthetic variant (quick run)
    python scripts/train_sft_synthetic.py \
        --variants synthetic \
        --num_samples 2000 --epochs 1

    # Smoke test
    python scripts/train_sft_synthetic.py --num_samples 200 --epochs 1 \
        --variants synthetic --no_eval
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def parse_args():
    p = argparse.ArgumentParser(
        description="Train SFT on synthetic data and compare to hh-rlhf"
    )
    p.add_argument("--synthetic_data", default="data/synthetic_sft.jsonl",
                   help="Path to JSONL file from generate_synthetic_sft.py")
    p.add_argument("--variants", nargs="+",
                   default=["hh_rlhf", "synthetic", "mixed"],
                   choices=["hh_rlhf", "synthetic", "mixed"],
                   help="Which SFT variants to train")
    p.add_argument("--model", default="gpt2-medium")
    p.add_argument("--num_samples", type=int, default=10_000,
                   help="Training samples per variant (total for mixed = 2x)")
    p.add_argument("--num_eval_samples", type=int, default=1_000)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--output_prefix", default="checkpoints/sft_synthetic")
    p.add_argument("--reward_checkpoint", default="checkpoints/reward",
                   help="Reward model for RM-judged evaluation")
    p.add_argument("--num_eval", type=int, default=500)
    p.add_argument("--no_eval", action="store_true",
                   help="Skip RM evaluation (just train)")
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--report_to", default="none", choices=["none", "wandb"])
    return p.parse_args()


def main():
    args = parse_args()

    from transformers import AutoTokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
    from src.data.preprocessing import SFTDataset
    from src.data.synthetic_sft import SyntheticSFTDataset

    results = {}

    for variant in args.variants:
        output_dir = f"{args.output_prefix}/{variant}"
        print(f"\n{'='*60}")
        print(f"  Training SFT variant: {variant}  →  {output_dir}")
        print(f"{'='*60}")

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        model = GPT2LMHeadModel.from_pretrained(args.model)
        model.config.pad_token_id = tokenizer.pad_token_id

        # Build dataset for this variant
        if variant == "hh_rlhf":
            train_dataset = SFTDataset(
                split="train", tokenizer=tokenizer,
                max_length=512, num_samples=args.num_samples,
            )
            eval_dataset = SFTDataset(
                split="test", tokenizer=tokenizer,
                max_length=512, num_samples=args.num_eval_samples,
            )
        elif variant == "synthetic":
            if not os.path.exists(args.synthetic_data):
                print(f"ERROR: synthetic data file not found: {args.synthetic_data}")
                print(f"Run: python scripts/generate_synthetic_sft.py --num_samples {args.num_samples}")
                continue
            train_dataset = SyntheticSFTDataset(
                jsonl_path=args.synthetic_data,
                tokenizer=tokenizer,
                max_length=512,
            )
            if len(train_dataset) > args.num_samples:
                from torch.utils.data import Subset
                import random
                indices = random.sample(range(len(train_dataset)), args.num_samples)
                train_dataset = Subset(train_dataset, indices)
            eval_dataset = SFTDataset(
                split="test", tokenizer=tokenizer,
                max_length=512, num_samples=args.num_eval_samples,
            )
        elif variant == "mixed":
            # 50% hh-rlhf + 50% synthetic
            from torch.utils.data import ConcatDataset, Subset
            import random
            n_each = args.num_samples // 2
            hh_ds = SFTDataset(
                split="train", tokenizer=tokenizer,
                max_length=512, num_samples=n_each,
            )
            if not os.path.exists(args.synthetic_data):
                print(f"ERROR: synthetic data not found: {args.synthetic_data}")
                continue
            syn_ds = SyntheticSFTDataset(
                jsonl_path=args.synthetic_data, tokenizer=tokenizer, max_length=512,
            )
            if len(syn_ds) > n_each:
                syn_ds = Subset(syn_ds, random.sample(range(len(syn_ds)), n_each))
            train_dataset = ConcatDataset([hh_ds, syn_ds])
            eval_dataset = SFTDataset(
                split="test", tokenizer=tokenizer,
                max_length=512, num_samples=args.num_eval_samples,
            )

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-5,
            weight_decay=0.01,
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            evaluation_strategy="steps",
            eval_steps=200,
            save_strategy="steps",
            save_steps=500,
            logging_steps=50,
            fp16=args.fp16,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=args.report_to,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        trainer.train()
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"  Saved SFT ({variant}) → {output_dir}")

        results[variant] = {"output_dir": output_dir, "n_train": len(train_dataset)}

    # ── RM-judged evaluation ───────────────────────────────────────────────────
    if not args.no_eval and os.path.exists(args.reward_checkpoint):
        print(f"\n{'='*60}")
        print("  RM-judged evaluation: mean reward per SFT variant")
        print(f"{'='*60}")

        from src.evaluation.metrics import compute_win_rate_rm

        print(f"{'Variant':<16} {'Mean Reward':>14} {'Win vs hh_rlhf':>16}")
        print("-" * 50)

        baseline_reward = None
        eval_results = {}
        for variant in args.variants:
            if variant not in results:
                continue
            ckpt = results[variant]["output_dir"]
            try:
                win_rate, mean_r = compute_win_rate_rm(
                    policy_checkpoint=ckpt,
                    reward_checkpoint=args.reward_checkpoint,
                    num_eval=args.num_eval,
                )
                eval_results[variant] = mean_r
                if variant == "hh_rlhf":
                    baseline_reward = mean_r
                delta = f"{(mean_r - baseline_reward):+.4f}" if baseline_reward is not None else "baseline"
                print(f"{variant:<16} {mean_r:>14.4f} {delta:>16}")
            except Exception as e:
                print(f"{variant:<16} {'ERROR':>14} {str(e)[:30]:>16}")

        results["eval"] = eval_results

    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/synthetic_sft_results.json", "w") as f:
        json.dump({k: str(v) if not isinstance(v, (int, float, str, dict)) else v
                   for k, v in results.items()}, f, indent=2)
    print("\nResults saved to results/synthetic_sft_results.json")


if __name__ == "__main__":
    main()
