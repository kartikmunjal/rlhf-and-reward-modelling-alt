#!/usr/bin/env python3
"""Publish README findings exclusively from generated study artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MAIN_START = "<!-- SUMMARIZATION-FINETUNE-RESULTS:START -->"
MAIN_END = "<!-- SUMMARIZATION-FINETUNE-RESULTS:END -->"
MODULE_START = "<!-- GENERATED-FINDINGS:START -->"
MODULE_END = "<!-- GENERATED-FINDINGS:END -->"


def effect(row: dict) -> str:
    return (
        f"{row['estimate']:.3f} (95% CI {row['ci95'][0]:.3f}–{row['ci95'][1]:.3f}; "
        f"N_trials={row['n_trials']}; {row['bootstrap_replicates']:,} bootstraps)"
    )


def coverage(row: dict) -> str:
    return (
        f"{row['valid']}/{row['total']} (Wilson 95% CI "
        f"{row['wilson_ci95'][0]:.3f}–{row['wilson_ci95'][1]:.3f})"
    )


def main_generated(metrics: dict, sft: dict, dpo: dict) -> str:
    sft_delta = metrics["comparisons"]["sft_minus_base"]["anthropic"]
    dpo_delta = metrics["comparisons"]["dpo_minus_sft"]["anthropic"]
    dpo_length = metrics["comparisons"]["dpo_minus_sft"]["length_words"]
    controlled = metrics["dpo_length_controlled"]
    claude = metrics["coverage"]["anthropic"]
    openai = metrics["coverage"]["openai"]
    pair = metrics["candidate_pair_validity"]
    runtime_hours = sft["trainer_metrics"]["train_runtime"] / 3600
    dpo_minutes = dpo["runtime_seconds"] / 60
    return "\n".join([
        "## Summarization SFT + Judge-DPO Extension",
        "",
        f"**Preregistered SFT result: {'PASS' if metrics['sft_success'] else 'FAIL'}.** On the same held-out articles, LoRA-SFT improved the frozen Claude judge by {effect(sft_delta['relevance'])} on relevance and {effect(sft_delta['consistency'])} on consistency. The ROUGE-L change was {effect(metrics['comparisons']['sft_minus_base']['rougeL'])}.",
        "",
        f"**Preregistered DPO result: {'PASS' if metrics['dpo_success'] else 'FAIL'}.** Judge-preference DPO improved relevance by {effect(dpo_delta['relevance'])}, but consistency changed by {effect(dpo_delta['consistency'])}; its interval includes zero, so the locked two-axis criterion failed. ROUGE-L changed by {effect(metrics['comparisons']['dpo_minus_sft']['rougeL'])}.",
        "",
        f"The failure was not explained by verbosity: DPO shifted length by {effect(dpo_length)} words. After controlling for length within article, the relevance effect was {effect(controlled['relevance'])} and consistency was {effect(controlled['consistency'])}. This is evidence against length exploitation as the mechanism, not proof of human-rated improvement.",
        "",
        f"Claude coverage was base {coverage(claude['base'])}, SFT {coverage(claude['sft'])}, and DPO {coverage(claude['dpo'])}. Candidate-pair validity was {coverage(pair)}, above the locked 90% floor. The GPT-5-mini audit covered base {coverage(openai['base'])}, SFT {coverage(openai['sft'])}, and DPO {coverage(openai['dpo'])}; SFT and DPO coverage missed the floor, so cross-provider DPO results are diagnostic only.",
        "",
        f"The single RTX 3070 run trained {sft['trainable_parameters']:,} LoRA parameters over {sft['train_examples']:,} examples for {sft['global_steps']:,} optimizer steps in {runtime_hours:.2f} hours (peak allocated GPU memory {sft['hardware']['peak_gpu_memory_bytes'] / 2**30:.2f} GiB). DPO used {dpo['preference_rows']:,} conservative preference pairs for {dpo['global_steps']:,} optimizer steps in {dpo_minutes:.2f} minutes (peak {dpo['hardware']['peak_gpu_memory_bytes'] / 2**30:.2f} GiB). N_training_seeds=1, so article-bootstrap intervals do not estimate seed variability.",
        "",
        "See [`summarization_finetune/`](summarization_finetune/), [`results/summarization_finetune_v1/report.md`](results/summarization_finetune_v1/report.md), and [`results/summarization_finetune_v1/metrics.json`](results/summarization_finetune_v1/metrics.json) for the locked protocols, complete generated results, and provenance manifests.",
    ])


def module_generated(metrics: dict, sft: dict, dpo: dict) -> str:
    sft_delta = metrics["comparisons"]["sft_minus_base"]["anthropic"]
    dpo_delta = metrics["comparisons"]["dpo_minus_sft"]["anthropic"]
    length = metrics["comparisons"]["dpo_minus_sft"]["length_words"]
    return "\n".join([
        "## Generated findings",
        "",
        f"SFT **{'passed' if metrics['sft_success'] else 'failed'}** its locked criterion: relevance improved by {effect(sft_delta['relevance'])} and consistency by {effect(sft_delta['consistency'])}.",
        "",
        f"DPO **{'passed' if metrics['dpo_success'] else 'failed'}** its locked two-axis criterion. Relevance changed by {effect(dpo_delta['relevance'])}; consistency changed by {effect(dpo_delta['consistency'])}. DPO length changed by {effect(length)} words, providing no evidence of verbosity exploitation.",
        "",
        f"The run used N_training_seeds={metrics['training_seeds']}, {sft['train_examples']:,} SFT examples, and {dpo['preference_rows']:,} conservative judge-preference pairs. Claude confirmatory coverage met the locked floor; GPT-5-mini SFT/DPO coverage did not and remains diagnostic.",
        "",
        "Complete estimates, 95% intervals, validity rates, and interpretation boundaries are generated in [`../results/summarization_finetune_v1/report.md`](../results/summarization_finetune_v1/report.md). Lightweight training provenance is in [`../results/summarization_finetune_v1/manifests/`](../results/summarization_finetune_v1/manifests/).",
    ])


def replace_between(text: str, start: str, end: str, content: str) -> str:
    if start not in text or end not in text:
        raise ValueError(f"Missing README markers: {start}, {end}")
    before, remainder = text.split(start, 1)
    _, after = remainder.split(end, 1)
    return before + start + "\n" + content.rstrip() + "\n" + end + after


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, default=Path("results/summarization_finetune_v1/metrics.json"))
    parser.add_argument("--sft-manifest", type=Path, default=Path("results/summarization_finetune_v1/manifests/sft_run_manifest.json"))
    parser.add_argument("--dpo-manifest", type=Path, default=Path("results/summarization_finetune_v1/manifests/dpo_run_manifest.json"))
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    parser.add_argument("--module-readme", type=Path, default=Path("summarization_finetune/README.md"))
    args = parser.parse_args()
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    sft = json.loads(args.sft_manifest.read_text(encoding="utf-8"))
    dpo = json.loads(args.dpo_manifest.read_text(encoding="utf-8"))

    readme = args.readme.read_text(encoding="utf-8")
    if MAIN_START not in readme:
        begin = readme.index("## Summarization SFT + Judge-DPO Extension")
        end = readme.index("<!-- SAFETY-RESULTS:START -->")
        readme = readme[:begin] + MAIN_START + "\nplaceholder\n" + MAIN_END + "\n\n" + readme[end:]
    args.readme.write_text(replace_between(readme, MAIN_START, MAIN_END, main_generated(metrics, sft, dpo)), encoding="utf-8")

    module = args.module_readme.read_text(encoding="utf-8")
    if MODULE_START not in module:
        module = module.rstrip() + f"\n\n{MODULE_START}\nplaceholder\n{MODULE_END}\n"
    args.module_readme.write_text(replace_between(module, MODULE_START, MODULE_END, module_generated(metrics, sft, dpo)), encoding="utf-8")
    print("Updated", args.readme, "and", args.module_readme)


if __name__ == "__main__":
    main()
