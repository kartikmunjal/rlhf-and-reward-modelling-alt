#!/usr/bin/env python3
"""Publish artifact-backed recursive-study findings into both README files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recursive_self_improvement.config import load_effective_config

START = "<!-- recursive-self-improvement-results:start -->"
END = "<!-- recursive-self-improvement-results:end -->"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def replace_block(path: Path, block: str, *, insertion_anchor: str | None = None) -> None:
    text = path.read_text(encoding="utf-8")
    rendered = f"{START}\n{block.rstrip()}\n{END}"
    if START in text:
        before, remainder = text.split(START, 1)
        _, after = remainder.split(END, 1)
        text = before + rendered + after
    elif insertion_anchor:
        if insertion_anchor not in text:
            raise ValueError(f"Missing insertion anchor in {path}: {insertion_anchor}")
        text = text.replace(insertion_anchor, rendered + "\n\n" + insertion_anchor, 1)
    else:
        text = text.rstrip() + "\n\n" + rendered + "\n"
    path.write_text(text, encoding="utf-8")


def ci(metric: dict) -> str:
    return f"{metric['estimate']:.3f} [{metric['ci95'][0]:.3f}, {metric['ci95'][1]:.3f}] (N={metric['n_trials']:,})"


def curve_line(label: str, curve: dict) -> str:
    linear = curve["candidates"]["linear"]["loo_mse"]
    saturation = curve["candidates"]["saturating_exponential"]["loo_mse"]
    return f"- {label}: `{curve['selected']}` selected by frozen LOO-MSE rule (linear {linear:.6g}; saturating exponential {saturation:.6g})."


def build(metrics: dict, integrity: dict, config: dict) -> str:
    if integrity.get("status") != "pass":
        raise ValueError("Stage-3 integrity audit has not passed")
    comparisons = metrics["stage3_vs_zero_percent"]
    percentages = config["stage3"]["label_mixture_percent_self"]
    nonzero_percentages = [percent for percent in percentages if percent]
    if any(comparisons.get(str(percent)) is None for percent in nonzero_percentages):
        raise ValueError("Stage-3 primary comparisons are incomplete")
    lines = [
        "## Recursive Self-Improvement Extension",
        "",
        "This artifact-backed GPT-2-medium study replaces the repository's old illustrative iterative-DPO figures. It extends one rolling-preference loop to eight rounds, measures capability against cumulative observed training tokens, and separately tests increasing reliance on self-generated likelihood labels.",
        "",
        "### Registered design and execution",
        "",
        f"- Stage 1: {config['stage1']['rounds']} rolling-2 DPO rounds; {config['stage1']['rollout_prompts_per_round']} on-policy prompts and {config['stage1']['dpo_steps_per_round']} optimizer steps per round.",
        f"- Stage 2: base → SFT → ordinary DPO → iterative-DPO rounds 1–{config['stage1']['rounds']} on one disjoint held-out suite.",
        f"- Stage 3: {integrity['conditions_verified']} verified cells ({'/'.join(map(str, percentages))}% self labels × {config['stage3']['rounds_per_condition']} rounds), {integrity['prompts_per_round']} prompts per cell, from the identical round-{config['stage3']['starting_checkpoint'].rsplit('_', 1)[-1]} start.",
        "- The frozen K=3 human-preference reward ensemble was evaluation-only in Stage 3; it never generated a Stage-3 training label.",
        "",
        "### Stage 1–2 findings",
        "",
        "| Checkpoint | Win rate vs SFT (paired prompt bootstrap 95% CI) |",
        "|---|---:|",
    ]
    for checkpoint, entry in metrics["capability"].items():
        if entry["win_rate"] is not None:
            lines.append(f"| `{checkpoint}` | {ci(entry['win_rate'])} |")
    lines.extend(
        [
            "",
            f"- The preregistered plateau rule first fires at round {metrics['stage1_plateau_round'] or 'none'}; this is a local decision-rule result, not proof that later escape is impossible.",
            curve_line("Round trajectory", metrics["stage1_curve"]),
            curve_line("Capability–compute trajectory", metrics["stage2_density_curve"]),
            "",
            "### Stage 3 primary result",
            "",
            "Each entry is the paired round-4 external-evaluator win-rate difference from the 0%-self control. P values are two-sided paired sign-flip tests with Holm correction across the four planned comparisons.",
            "",
            "| Self labels | Difference vs 0% (95% CI) | N | Holm p | Degradation rule met? |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for percent in nonzero_percentages:
        value = comparisons[str(percent)]
        degraded = value["ci95"][1] < 0
        lines.append(
            f"| {percent}% | {value['estimate']:.3f} [{value['ci95'][0]:.3f}, {value['ci95'][1]:.3f}] | {value['n_trials']:,} | {value['holm_adjusted_p_value']:.4f} | {'yes' if degraded else 'no'} |"
        )
    lines.extend(
        [
            "",
            "### Reward-hacking diagnostics at round 4",
            "",
            "| Self labels | K=3 win rate vs common start | Length shift vs 0% (tokens) | Mean KL from common start | Mean ensemble disagreement |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for percent in percentages:
        values = metrics["stage3_round_metrics"][str(percent)]["4"]
        shift = "reference" if percent == 0 else ci(metrics["stage3_round4_length_shift_vs_zero"][str(percent)])
        lines.append(
            f"| {percent}% | {ci(values['external_evaluator_win_rate'])} | {shift} | {ci(values['kl_from_stage3_start'])} | {ci(values['ensemble_disagreement'])} |"
        )
    lines.extend(
        [
            "",
            "Training-label agreement with the frozen evaluator is reported by round and label source in the generated metrics, so apparent improvement can be checked for circular self-label behavior.",
            "",
            "### Scope boundaries",
            "",
            "This is a one-seed, small-model controlled study. Prompt-bootstrap intervals quantify held-out-prompt uncertainty, not training-seed variation. Self-likelihood ranking is a narrow operationalization of self-reliance, not deliberative self-judgment. The selected curves describe only the observed checkpoints and are not extrapolation laws. PPO/GRPO remain excluded because the existing artifacts use a different model family; no new checkpoint was manufactured to fill that matrix.",
            "",
            "Reproduce from [`recursive_self_improvement/preregistration.md`](recursive_self_improvement/preregistration.md), [`recursive_self_improvement/study_config.json`](recursive_self_improvement/study_config.json), and [`scripts/analyze_recursive_self_improvement.py`](scripts/analyze_recursive_self_improvement.py). Full generated results are in [`results/recursive_self_improvement_v1/report.md`](results/recursive_self_improvement_v1/report.md) and [`metrics.json`](results/recursive_self_improvement_v1/metrics.json).",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, default=Path("results/recursive_self_improvement_v1/metrics.json"))
    parser.add_argument("--integrity", type=Path, default=Path("results/recursive_self_improvement_v1/stage3_integrity_audit.json"))
    parser.add_argument("--main-readme", type=Path, default=Path("README.md"))
    parser.add_argument("--module-readme", type=Path, default=Path("recursive_self_improvement/README.md"))
    args = parser.parse_args()
    block = build(load(args.metrics), load(args.integrity), load_effective_config(ROOT))
    replace_block(args.main_readme, block, insertion_anchor="## Safety Classifier & Fairness Extension")
    replace_block(args.module_readme, block)
    print(args.main_readme)
    print(args.module_readme)


if __name__ == "__main__":
    main()
