#!/usr/bin/env python3
"""Generate the locked SummEval held-out analysis and research report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_judge_summeval.analysis import analyze
from llm_judge_summeval.data import AXES


def cell(metric: dict) -> str:
    return (f"{metric['estimate']:.3f} [{metric['ci95'][0]:.3f}, {metric['ci95'][1]:.3f}] "
            f"(N={metric['n_observations']}; {metric['n_clusters']} {metric['cluster_unit']} clusters)")


def render(metrics: dict) -> str:
    lines = ["# SummEval LLM-as-Judge v1 — preregistered result", "", "## Integrity", "",
             f"- Study status: `{metrics['status']}`.", "- Primary model: pinned Claude Haiku 4.5 snapshot.",
             "- Secondary model: pinned GPT-5 mini snapshot.", "- Confidence intervals: 2,000 source-article cluster-bootstrap replicates.",
             "", "## Data completeness and API usage", "",
             "| Component | Valid / preregistered N_trials | Validity rate (Wilson 95% CI) | Input tokens | Output tokens |",
             "|---|---:|---:|---:|---:|"]
    for label, row in (("Claude pointwise", metrics["pointwise"]["anthropic"]),
                       ("GPT-5 mini pointwise", metrics["pointwise"]["openai"]),
                       ("Claude pairwise", metrics["pairwise"])):
        ci = row["valid_rate_wilson_ci95"]
        lines.append(f"| {label} | {row['valid']} / {row['total']} | {row['valid_rate']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | {row['usage']['input_tokens']} | {row['usage']['output_tokens']} |")
    costs = metrics["actual_api_cost_at_recorded_rates"]
    lines += ["", f"Recorded-token API cost at the versioned runtime rates: Claude ${costs['anthropic_usd']:.4f}, OpenAI ${costs['openai_usd']:.4f}, total ${costs['total_usd']:.4f}.",
             "", "## Pointwise human correlation", "",
             "| Provider | Axis | Judge vs human Spearman ρ (95% CI) | ROUGE-L vs human ρ (95% CI) | Judge − ROUGE-L (95% CI) | Length bias ρ (95% CI) | Judge vs human, controlling length (95% CI) |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for provider in ("anthropic", "openai"):
        if not metrics["pointwise"][provider]["axes"]:
            lines.append(f"| {provider} | all | infrastructure incomplete | infrastructure incomplete | infrastructure incomplete | infrastructure incomplete | infrastructure incomplete |")
            continue
        for axis in AXES:
            row = metrics["pointwise"][provider]["axes"][axis]
            lines.append(f"| {provider} | {axis} | {cell(row['judge_human'])} | {cell(row['rouge_human'])} | {cell(row['judge_minus_rouge'])} | {cell(row['length_bias'])} | {cell(row['judge_human_controlling_length'])} |")
    lines += ["", f"Primary success: **{metrics['primary_success']}**. " + ", ".join(f"{axis}={value}" for axis, value in metrics["primary_success_by_axis"].items()),
              "", "## Cross-provider agreement", "", "| Axis | Claude vs GPT-5 mini Spearman ρ (95% CI) |", "|---|---:|"]
    if metrics["cross_provider"]["axes"]:
        for axis in AXES:
            lines.append(f"| {axis} | {cell(metrics['cross_provider']['axes'][axis])} |")
    else:
        lines.append("| all | infrastructure incomplete |")
    lines += ["", "## Pairwise position bias and mitigation", "",
              "| Axis | Preference flip rate | Tie instability | First-order human agreement | Symmetrized human agreement | BT vs human ρ (95% CI) | Symmetrized − first-order agreement (95% CI) |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for axis in AXES:
        row = metrics["pairwise"]["axes"][axis]
        lines.append(f"| {axis} | {cell(row['preference_flip_cluster_bootstrap'])} | {cell(row['tie_instability_cluster_bootstrap'])} | "
                     f"{cell(row['first_order_human_agreement'])} | {cell(row['symmetrized_human_agreement'])} | "
                     f"{cell(row['bradley_terry_human_spearman'])} | {cell(row['symmetrized_minus_first'])} |")
    lines += ["", "## Interpretation boundary", "",
              "The confirmatory claim applies only to the pinned models, frozen prompts, expert-mean SummEval labels, and document-disjoint held-out split. Secondary axes, cross-provider agreement, pairwise diagnostics, and mitigation results cannot rescue failure of either primary axis."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("llm_judge_summeval/study_config.json"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/summeval_judge_v1"))
    parser.add_argument("--prompts", type=Path, default=Path("llm_judge_summeval/prompts.json"))
    parser.add_argument("--prompt-manifest", type=Path, default=Path("llm_judge_summeval/final_prompt_manifest.json"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/summeval_judge_v1"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    metrics = analyze(args.results_dir, args.processed_dir, args.prompts, args.prompt_manifest, config)
    if metrics["primary_status"] != "complete":
        raise SystemExit("Primary Claude response coverage is below the preregistered 90% threshold; no final report published")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.results_dir / "report.md").write_text(render(metrics), encoding="utf-8")
    print("Wrote", args.results_dir / "report.md")


if __name__ == "__main__":
    main()
