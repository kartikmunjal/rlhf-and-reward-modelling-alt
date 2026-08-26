#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from summarization_finetune.analysis import analyze


def cell(row):
    return f"{row['estimate']:.3f} [{row['ci95'][0]:.3f}, {row['ci95'][1]:.3f}] (N_trials={row['n_trials']}; 2,000 bootstraps)"


def render(m):
    lines = ["# Summarization SFT + Judge-DPO v1 result", "", "## Preregistered decisions", "",
             f"- SFT primary success: **{m['sft_success']}** ({m['sft_success_by_axis']}).",
             f"- DPO primary success: **{m['dpo_success']}** ({m['dpo_success_by_axis']}).",
             f"- Training seeds: {m['training_seeds']} (article bootstrap does not estimate seed variability).", "",
             "## Infrastructure validity", "",
             "| Provider/model | Valid / planned | Wilson 95% CI |", "|---|---:|---:|"]
    for provider in ("anthropic", "openai"):
        for model in ("base", "sft", "dpo"):
            row=m["coverage"][provider][model]; lines.append(f"| {provider}/{model} | {row['valid']}/{row['total']} | [{row['wilson_ci95'][0]:.3f}, {row['wilson_ci95'][1]:.3f}] |")
    pair=m["candidate_pair_validity"]; lines.append(f"| anthropic/DPO candidate pairs | {pair['valid']}/{pair['total']} | [{pair['wilson_ci95'][0]:.3f}, {pair['wilson_ci95'][1]:.3f}] |")
    lines += ["", "## Paired score changes", "", "| Comparison | Provider | Axis | Mean change (95% CI) |", "|---|---|---|---:|"]
    for comparison in ("sft_minus_base", "dpo_minus_sft"):
        for provider in ("anthropic", "openai"):
            for axis in ("relevance", "consistency", "coherence", "fluency"):
                lines.append(f"| {comparison} | {provider} | {axis} | {cell(m['comparisons'][comparison][provider][axis])} |")
        lines.append(f"| {comparison} | reference proxy | ROUGE-L | {cell(m['comparisons'][comparison]['rougeL'])} |")
        lines.append(f"| {comparison} | diagnostic | length words | {cell(m['comparisons'][comparison]['length_words'])} |")
    lines += ["", "## DPO length-hacking diagnostic", "", "| Axis | Length-controlled DPO treatment effect | Classification |", "|---|---:|---|"]
    for axis in ("relevance", "consistency"):
        lines.append(f"| {axis} | {cell(m['dpo_length_controlled'][axis])} | `{m['reward_hacking'][axis]['classification']}` |")
    lines += ["", "## Interpretation boundary", "",
              "Claude pointwise scores are confirmatory. GPT-5-mini and ROUGE-L are independent but imperfect proxies. A positive length-controlled effect is evidence against verbosity as this particular exploitation mechanism, not proof equivalent to new human ratings. Pairwise preference or evaluation coverage below the locked 90% floor makes the affected DPO claim infrastructure-incomplete. Results generalize over sampled articles, not training seeds."]
    return "\n".join(lines)+"\n"


parser=argparse.ArgumentParser()
parser.add_argument("--config",type=Path,default=Path("summarization_finetune/study_config.json"))
parser.add_argument("--data-dir",type=Path,default=Path("data/processed/summarization_finetune_v1"))
parser.add_argument("--generations-dir",type=Path,default=Path("results/summarization_finetune_v1/generations"))
parser.add_argument("--judge-dir",type=Path,default=Path("results/summarization_finetune_v1/judge"))
parser.add_argument("--output-dir",type=Path,default=Path("results/summarization_finetune_v1"))
args=parser.parse_args(); root=Path(__file__).resolve().parents[1]
metrics=analyze(config=json.loads(args.config.read_text()),data_dir=args.data_dir,generations_dir=args.generations_dir,judge_dir=args.judge_dir,output_dir=args.output_dir,root=root)
(args.output_dir/"report.md").write_text(render(metrics),encoding="utf-8"); print("Wrote",args.output_dir/"report.md")
