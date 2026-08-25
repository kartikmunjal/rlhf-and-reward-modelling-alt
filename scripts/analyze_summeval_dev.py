#!/usr/bin/env python3
"""Generate development-only diagnostics without accessing held-out labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_judge_summeval.analysis import pointwise_analysis
from llm_judge_summeval.data import AXES, load_prompt_development_rows


def cell(metric: dict) -> str:
    return (f"{metric['estimate']:.3f} [{metric['ci95'][0]:.3f}, {metric['ci95'][1]:.3f}] "
            f"(N={metric['n_observations']}; {metric['n_clusters']} {metric['cluster_unit']} clusters)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("llm_judge_summeval/study_config.json"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/summeval_judge_v1"))
    parser.add_argument("--ledger", type=Path, default=Path("results/summeval_judge_v1/anthropic_dev_pointwise.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/summeval_judge_v1/dev_report.md"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows = load_prompt_development_rows(args.processed_dir)
    labels = {row["summary_id"]: row for row in rows}
    metrics = pointwise_analysis("anthropic", args.ledger, rows, labels, config)
    lines = ["# SummEval prompt-development diagnostics", "",
             "Development partition only; these results are not confirmatory and do not use held-out labels.", "",
             f"Valid responses: {metrics['valid']} / {metrics['total']} "
             f"(Wilson 95% CI {metrics['valid_rate_wilson_ci95'][0]:.3f}–{metrics['valid_rate_wilson_ci95'][1]:.3f}).",
             "", "| Axis | Judge vs human Spearman rho (95% CI) | ROUGE-L vs human (95% CI) | Judge - ROUGE-L (95% CI) |",
             "|---|---:|---:|---:|"]
    for axis in AXES:
        row = metrics["axes"][axis]
        lines.append(f"| {axis} | {cell(row['judge_human'])} | {cell(row['rouge_human'])} | {cell(row['judge_minus_rouge'])} |")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output.parent / "dev_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Wrote", args.output)


if __name__ == "__main__":
    main()
