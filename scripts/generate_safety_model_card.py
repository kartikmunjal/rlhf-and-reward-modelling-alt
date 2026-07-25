#!/usr/bin/env python3
"""Generate a model card strictly from manifests and completed result files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_safety_classifier import write_text_artifact  # noqa: E402
from src.safety.taxonomy import TARGET_LABELS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aggregate", default="results/safety_v2/aggregate/aggregate_metrics.json"
    )
    parser.add_argument("--results-root", default="results/safety_v2")
    parser.add_argument(
        "--data-manifest", default="data/processed/safety_v2/data_manifest.json"
    )
    parser.add_argument("--output", default="results/safety_v2/MODEL_CARD.md")
    return parser.parse_args()


def cell(metric: dict) -> str:
    return (
        f'{metric["value"]:.3f} '
        f'[{metric["ci95"][0]:.3f}, {metric["ci95"][1]:.3f}]'
    )


def main() -> None:
    args = parse_args()
    aggregate = json.loads(Path(args.aggregate).read_text(encoding="utf-8"))
    if aggregate["n_trials"] != aggregate["n_trials_planned"]:
        raise ValueError("Model card requires all preregistered trials")
    trial_id = aggregate["selected_trial_id"]
    metrics = json.loads(
        (Path(args.results_root) / trial_id / "metrics.json").read_text(encoding="utf-8")
    )
    data_manifest = json.loads(Path(args.data_manifest).read_text(encoding="utf-8"))
    performance = metrics["jigsaw_performance"]
    fairness = metrics["civil_comments_identity_fairness"]
    lines = [
        "# Safety classifier v2 model card",
        "",
        f'- Selected trial: `{trial_id}`',
        f'- N_trials: {aggregate["n_trials"]}',
        f'- Taxonomy: `{metrics["manifest"]["taxonomy_version"]}`',
        f'- Base model: `{metrics["manifest"]["config"]["base_model"]}`',
        "",
        "## Intended use",
        "",
        "Research evaluation of three overlapping English text-safety categories:",
        "hate/harassment, sexualized content, and harmful/violent content. Outputs",
        "are decision-support signals, not autonomous moderation decisions.",
        "",
        "## Prohibited use",
        "",
        "Do not use this model as the sole basis for account sanctions, law-enforcement",
        "action, employment decisions, medical decisions, or demographic profiling.",
        "It is not validated outside English or for population-level fairness.",
        "",
        "## Primary frozen-test results",
        "",
        "| Category | F1 (95% CI) | PR-AUC (95% CI) |",
        "|---|---:|---:|",
    ]
    for name in TARGET_LABELS:
        lines.append(
            f'| {name} | {cell(performance["per_category"][name]["f1"])} | '
            f'{cell(metrics["jigsaw_calibration"]["per_category"][name]["pr_auc"])} |'
        )
    lines.extend(
        [
            "",
            f'Paired macro-F1 change vs v1: '
            f'{cell(aggregate["paired_macro_f1_improvement_vs_v1"])}.',
            "",
            "## Fairness and diagnostics",
            "",
            "Identity-slice metrics use original Civil Comments identity annotations.",
            "Worst-minus-best gaps are diagnostic and may reflect overlapping groups,",
            "label prevalence, annotation noise, or domain shift; they are not causal.",
            f'Observed identity FPR gap: {cell(fairness["fpr_gap_worst_minus_best"])}.'
            if fairness["fpr_gap_worst_minus_best"]
            else "Identity FPR gap unavailable.",
            "",
            "HateCheck, ToxiGen, BeaverTails, threshold-sensitivity, calibration, and",
            "per-identity results are retained in the machine-readable trial metrics.",
            "",
            "## Training data and lineage",
            "",
            "Training uses the frozen Jigsaw train partition plus BeaverTails",
            "`330k_train`. ToxiGen, HateCheck, Civil Comments, Jigsaw test, and",
            "BeaverTails test are evaluation-only. Exact source revisions:",
            "",
        ]
    )
    for source, item in data_manifest["sources"].items():
        lines.append(f'- {source}: `{item["repository"]}@{item["revision"]}`')
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- Operational labels are proxies, not universal definitions of harm.",
            "- BeaverTails is pair-level QA data and differs from comment moderation.",
            "- ToxiGen includes machine-generated text.",
            "- Identity categories overlap and do not establish causal discrimination.",
            "- Context beyond 256 tokens is truncated.",
            "- Human review and a documented appeal process remain necessary.",
            "",
            "## Reproducibility",
            "",
            "The preregistration, ordered trial ledger, dataset manifest, per-trial",
            "manifests, predictions, aggregate report, and error-analysis sheets form",
            "the audit trail. No result in this card is hand-entered.",
            "",
        ]
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_text_artifact(output, "\n".join(lines))
    print(f"Wrote model card: {output}")


if __name__ == "__main__":
    main()
