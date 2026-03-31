#!/usr/bin/env python3
"""
Compare Process Reward Model (PRM) vs Outcome Reward Model (ORM) on GSM8K.

This script runs the core ablation:
  1. Generate solutions to test problems using the SFT model
  2. Score each solution with both the ORM and PRM
  3. For solutions where the final answer is correct, check if the PRM
     detects any incorrect intermediate steps the ORM would have missed
  4. For solutions where the final answer is wrong, compare how confidently
     each model flagged the solution

The central hypothesis: PRM catches errors the ORM misses (particularly
correct final answers reached via faulty reasoning).

Usage
-----
python scripts/compare_prm_orm.py \\
    --sft_checkpoint checkpoints/sft \\
    --prm_checkpoint checkpoints/prm \\
    --orm_checkpoint checkpoints/orm \\
    --num_eval 500
"""

import argparse
import json
import os
import re
import sys

import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from transformers import AutoTokenizer, GPT2LMHeadModel
from datasets import load_dataset

from src.models.process_reward_model import GPT2ProcessRewardModel
from src.models.reward_model import GPT2RewardModel
from src.data.gsm8k import extract_final_answer, parse_steps, verify_step


def generate_solution(model, tokenizer, question: str, device, max_new_tokens=256) -> str:
    prompt = f"Question: {question}\nSolution:"
    enc = tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True).to(device)
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=max_new_tokens,
            do_sample=True, top_p=0.9, temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )
    full = tokenizer.decode(out[0], skip_special_tokens=True)
    return full[len(prompt):]


def score_with_orm(orm, tokenizer, question, solution, device, max_length=512) -> float:
    text = f"Question: {question}\nSolution:{solution}"
    enc = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True).to(device)
    with torch.no_grad():
        return orm(**enc).rewards.item()


def score_with_prm(prm, tokenizer, question, solution, device, max_length=512) -> float:
    text = f"Question: {question}\nSolution:{solution}"
    enc = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True).to(device)
    with torch.no_grad():
        return prm(**enc).aggregate_reward.item()


def main():
    p = argparse.ArgumentParser(description="PRM vs ORM ablation on GSM8K")
    p.add_argument("--sft_checkpoint", default="checkpoints/sft")
    p.add_argument("--prm_checkpoint", default="checkpoints/prm")
    p.add_argument("--orm_checkpoint", default="checkpoints/orm")
    p.add_argument("--num_eval", type=int, default=500)
    p.add_argument("--output_dir", default="results/prm_vs_orm")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.sft_checkpoint)
    tokenizer.pad_token = tokenizer.eos_token

    sft_model = GPT2LMHeadModel.from_pretrained(args.sft_checkpoint).to(device)
    sft_model.eval()

    prm = GPT2ProcessRewardModel.from_pretrained(args.prm_checkpoint).to(device)
    prm.eval()

    orm = GPT2RewardModel.from_pretrained(args.orm_checkpoint).to(device)
    orm.eval()

    raw = load_dataset("openai/gsm8k", "main", split="test")
    raw = raw.select(range(min(args.num_eval, len(raw))))

    results = []
    for ex in raw:
        q = ex["question"]
        ref_answer = extract_final_answer(ex["answer"])

        solution = generate_solution(sft_model, tokenizer, q, device)
        generated_answer = extract_final_answer(solution)
        answer_correct = (generated_answer == ref_answer)

        # Check if any step is arithmetically wrong
        steps = parse_steps(solution)
        step_correctness = [verify_step(s) for s in steps]
        any_wrong_step = not all(step_correctness)

        orm_score = score_with_orm(orm, tokenizer, q, solution, device)
        prm_score = score_with_prm(prm, tokenizer, q, solution, device)

        results.append({
            "question": q[:80],
            "answer_correct": answer_correct,
            "any_wrong_step": any_wrong_step,
            "orm_score": orm_score,
            "prm_score": prm_score,
            # The critical case: right answer, wrong steps
            "correct_answer_wrong_steps": answer_correct and any_wrong_step,
        })

    # ── Analysis ──────────────────────────────────────────────────────
    n = len(results)
    correct = [r for r in results if r["answer_correct"]]
    wrong   = [r for r in results if not r["answer_correct"]]
    # Cases where PRM should disagree with ORM
    tricky  = [r for r in results if r["correct_answer_wrong_steps"]]

    print(f"\n{'='*60}")
    print(f"PRM vs ORM Ablation — GSM8K ({n} examples)")
    print(f"{'='*60}")
    print(f"Correct final answers  : {len(correct)/n*100:.1f}%")
    print(f"  of which: any wrong intermediate step : {len(tricky)/max(len(correct),1)*100:.1f}%")
    print()

    if tricky:
        orm_caught = sum(1 for r in tricky if r["orm_score"] < 0)
        prm_caught = sum(1 for r in tricky if r["prm_score"] < 0.5)
        print("Tricky cases (correct answer, wrong intermediate steps):")
        print(f"  ORM flagged as suspicious : {orm_caught}/{len(tricky)} ({orm_caught/len(tricky)*100:.1f}%)")
        print(f"  PRM flagged as suspicious : {prm_caught}/{len(tricky)} ({prm_caught/len(tricky)*100:.1f}%)")
        print()

    for label, subset in [("Correct solutions", correct), ("Wrong solutions", wrong)]:
        if not subset:
            continue
        orm_scores = [r["orm_score"] for r in subset]
        prm_scores = [r["prm_score"] for r in subset]
        print(f"{label} (n={len(subset)}):")
        print(f"  ORM mean score : {np.mean(orm_scores):.4f}  (std={np.std(orm_scores):.4f})")
        print(f"  PRM mean score : {np.mean(prm_scores):.4f}  (std={np.std(prm_scores):.4f})")
        print()

    # ── Plot ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Score distributions by outcome
    for ax, model_key, title in zip(
        axes[:2],
        ["orm_score", "prm_score"],
        ["ORM Scores", "PRM Scores"],
    ):
        ax.hist([r[model_key] for r in correct], bins=30, alpha=0.6,
                label="Correct answer", color="steelblue")
        ax.hist([r[model_key] for r in wrong], bins=30, alpha=0.6,
                label="Wrong answer", color="tomato")
        if tricky:
            ax.hist([r[model_key] for r in tricky], bins=20, alpha=0.8,
                    label="Correct answer,\nwrong steps", color="orange", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

    # ORM vs PRM scatter coloured by answer correctness
    orm_scores = [r["orm_score"] for r in results]
    prm_scores = [r["prm_score"] for r in results]
    colors = ["steelblue" if r["answer_correct"] else "tomato" for r in results]
    axes[2].scatter(orm_scores, prm_scores, c=colors, alpha=0.4, s=15)
    axes[2].set_xlabel("ORM score")
    axes[2].set_ylabel("PRM score")
    axes[2].set_title("ORM vs PRM (blue=correct, red=wrong)")

    if tricky:
        tricky_orm = [r["orm_score"] for r in tricky]
        tricky_prm = [r["prm_score"] for r in tricky]
        axes[2].scatter(tricky_orm, tricky_prm, c="orange", s=40,
                        label="Correct ans,\nwrong steps", zorder=5, marker="^")
        axes[2].legend(fontsize=8)

    plt.suptitle("Process vs Outcome Reward Model — GSM8K Ablation", y=1.02)
    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, "prm_vs_orm.png")
    plt.savefig(plot_path, dpi=100, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")

    # Save full results
    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Full results saved to {results_path}")


if __name__ == "__main__":
    main()
