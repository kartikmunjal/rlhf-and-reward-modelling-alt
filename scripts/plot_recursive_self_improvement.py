#!/usr/bin/env python3
"""Regenerate all recursive self-improvement figures from aggregate artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def interval(metric):
    return metric["estimate"], metric["estimate"] - metric["ci95"][0], metric["ci95"][1] - metric["estimate"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, default=Path("results/recursive_self_improvement_v1/metrics.json"))
    parser.add_argument("--compute", type=Path, default=Path("results/recursive_self_improvement_v1/compute.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/recursive_self_improvement_v1/figures"))
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = load(args.metrics)
    compute = {json.loads(line)["checkpoint"]: json.loads(line) for line in args.compute.read_text(encoding="utf-8").splitlines() if line}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rounds = [f"iterative_dpo_round_{index}" for index in range(1, 9)]
    values = [interval(metrics["capability"][name]["win_rate"])[0] for name in rounds]
    errors = [[interval(metrics["capability"][name]["win_rate"])[1] for name in rounds], [interval(metrics["capability"][name]["win_rate"])[2] for name in rounds]]
    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.errorbar(range(1, 9), values, yerr=errors, marker="o", capsize=3, label="Observed vs SFT")
    selected = metrics["stage1_curve"]["selected"]
    fitted = metrics["stage1_curve"]["candidates"][selected]["fitted"]
    axis.plot(range(1, 9), fitted, linestyle="--", label=f"Selected {selected.replace('_', ' ')} fit")
    axis.axhline(0.5, color="0.6", linewidth=1)
    axis.set(xlabel="Iterative-DPO round", ylabel="Pairwise win rate vs SFT", title="Stage 1: longer-horizon iterative DPO")
    axis.legend(); fig.tight_layout()
    fig.savefig(args.output_dir / "stage1_round_curve.svg", metadata={"Date": None}); plt.close(fig)

    checkpoints = metrics["stage2_checkpoints_in_fit"]
    x = [compute[name]["cumulative_non_padding_training_tokens"] for name in checkpoints]
    y = [interval(metrics["capability"][name]["win_rate"])[0] for name in checkpoints]
    yerr = [[interval(metrics["capability"][name]["win_rate"])[1] for name in checkpoints], [interval(metrics["capability"][name]["win_rate"])[2] for name in checkpoints]]
    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.errorbar(x, y, yerr=yerr, marker="o", linestyle="none", capsize=3)
    for x_value, y_value, name in zip(x, y, checkpoints):
        axis.annotate(name.replace("iterative_dpo_", "r"), (x_value, y_value), xytext=(3, 4), textcoords="offset points", fontsize=7)
    axis.set(xlabel="Cumulative non-padding training tokens", ylabel="Pairwise win rate vs SFT", title="Stage 2: capability versus measured training compute")
    axis.ticklabel_format(axis="x", style="sci", scilimits=(0, 0)); fig.tight_layout()
    fig.savefig(args.output_dir / "stage2_capability_compute.svg", metadata={"Date": None}); plt.close(fig)

    if any(metrics["stage3_round_metrics"].get(str(percent), {}).get("1", {}).get("external_evaluator_win_rate") for percent in (0, 25, 50, 75, 100)):
        fig, axis = plt.subplots(figsize=(7.2, 4.2))
        for percent in (0, 25, 50, 75, 100):
            series = [metrics["stage3_round_metrics"][str(percent)][str(round_index)]["external_evaluator_win_rate"] for round_index in range(1, 5)]
            values = [interval(item)[0] for item in series]
            errors = [[interval(item)[1] for item in series], [interval(item)[2] for item in series]]
            axis.errorbar(range(1, 5), values, yerr=errors, marker="o", capsize=3, label=f"{percent}% self")
        axis.set(xlabel="Stage-3 round", ylabel="Win rate vs common round-3 start", title="Stage 3: increasing self-label reliance")
        axis.set_xticks(range(1, 5)); axis.legend(ncol=2); fig.tight_layout()
        fig.savefig(args.output_dir / "stage3_self_reliance.svg", metadata={"Date": None}); plt.close(fig)


if __name__ == "__main__":
    main()
