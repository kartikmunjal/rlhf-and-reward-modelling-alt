#!/usr/bin/env python3
"""Generate a reproducible interim safety-results report and README block."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

LABELS = ("hate_harassment", "sexualized", "harmful_violent")
START = "<!-- SAFETY-RESULTS:START -->"
END = "<!-- SAFETY-RESULTS:END -->"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--snapshot-dir", default="results/safety_v2_interim_2026-07-26"
    )
    parser.add_argument(
        "--v1-metrics", default="results/safety_classifier_v1/metrics.json"
    )
    parser.add_argument("--readme", default="README.md")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def metric(cell: dict, digits: int = 3) -> str:
    value = cell["value"]
    low, high = cell["ci95"]
    return f"{value:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def percent(cell: dict) -> str:
    value = 100 * cell["value"]
    low, high = (100 * number for number in cell["ci95"])
    return f"{value:.2f}% [{low:.2f}%, {high:.2f}%]"


def mean_sd(values: list[float]) -> str:
    if len(values) == 1:
        return f"{values[0]:.3f} (N_seeds=1; SD unavailable)"
    return (
        f"{statistics.mean(values):.3f} ± {statistics.stdev(values):.3f} SD "
        f"(N_seeds={len(values)})"
    )


def family_name(trial_id: str) -> str:
    if trial_id.startswith("unweighted_bce"):
        return "Unweighted BCE"
    if trial_id.startswith("raw_weighted_bce"):
        return "Raw weighted BCE"
    if trial_id.startswith("capped_weighted_bce"):
        return "Capped weighted BCE"
    if trial_id.startswith("focal"):
        return "Focal"
    raise ValueError(f"Unknown trial family: {trial_id}")


def build_report(snapshot: dict, payloads: list[dict], v1: dict) -> str:
    statuses = {row["trial_id"]: row["status"] for row in snapshot["trials"]}
    completed = sum(status == "complete" for status in statuses.values())
    trained = sum(status in {"trained", "evaluating", "complete"} for status in statuses.values())
    planned = int(snapshot["n_trials_planned"])
    first = payloads[0]
    manifest = first["manifest"]
    v1_perf = v1["performance"]
    v1_macro = statistics.mean(
        v1_perf["per_category"][label]["f1"]["value"] for label in LABELS
    )

    lines = [
        "## Safety Classifier & Fairness Extension",
        "",
        "This extension turns the reward-model infrastructure into a real",
        "three-label safety classifier: `hate_harassment`, `sexualized`, and",
        "`harmful_violent`. V1 is a completed Jigsaw baseline. V2 is a",
        "preregistered 4-loss × 3-seed study; the table below is an interim",
        "snapshot and is not a final model-selection result.",
        "",
        "### Experiment status",
        "",
        "| Item | Recorded result |",
        "|---|---|",
        "| V1 | Complete; N_trials=1 |",
        f"| V2 training | {trained}/{planned} trials trained |",
        f"| V2 evaluation snapshot | {completed}/{planned} trials evaluated; phase `{snapshot['phase']}` |",
        f"| V2 training examples | {manifest['n_train']:,}: "
        f"{manifest['train_source_counts']['jigsaw']:,} Jigsaw + "
        f"{manifest['train_source_counts']['beavertails']:,} unique BeaverTails pairs |",
        f"| V2 positive labels | hate/harassment {manifest['positive_counts']['hate_harassment']:,}; "
        f"sexualized {manifest['positive_counts']['sexualized']:,}; "
        f"harmful/violent {manifest['positive_counts']['harmful_violent']:,} |",
        f"| External evaluation | BeaverTails {first['beavertails_external']['n_examples']:,}; "
        f"ToxiGen {first['toxigen_external']['overall']['n_examples']:,}; "
        f"HateCheck {first['hatecheck_external']['overall']['n_examples']:,}; "
        f"Civil Comments {first['civil_comments_identity_fairness']['n_examples']:,} |",
        "| Uncertainty | 2,000 deterministic bootstrap replicates for model metrics; "
        "Wilson/Newcombe intervals for rates and gaps |",
        "",
        "The v2 data manifest pins source revisions and hashes. BeaverTails repeated",
        "annotation rows are aggregated by strict per-target majority vote; pairs",
        "tied on any mapped label are excluded under preregistration amendment 1.",
        "External sets never enter training or threshold selection.",
        "",
        "### Published v1 baseline",
        "",
        "| Category | Precision (95% CI) | Recall (95% CI) | F1 (95% CI) | Test support |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in LABELS:
        row = v1_perf["per_category"][label]
        lines.append(
            f"| `{label}` | {metric(row['precision'])} | {metric(row['recall'])} | "
            f"{metric(row['f1'])} | {row['support']:,} |"
        )
    lines.extend(
        [
            "",
            f"V1 descriptive macro F1 is **{v1_macro:.3f}** (arithmetic mean of the",
            "three category point estimates; category bootstrap intervals are shown",
            "above). Any-label FPR on 14,256 negative Jigsaw rows is",
            f"**{percent(v1['fairness']['overall_test_fpr'])}**. The frozen",
            "adjacent-benign set has 0/60 flags, or",
            f"**{percent(v1['fairness']['adjacent_benign_fpr'])}**; its small size",
            "makes the upper confidence bound important.",
            "",
            "### Interim v2 results — completed evaluations only",
            "",
            "The locked selection metric is model-selection macro F1. Jigsaw test",
            "metrics are disclosed for transparency but cannot choose the winner.",
            "",
            "| Trial | Selection macro F1 (95% CI) | Jigsaw test macro F1 (95% CI) | "
            "Beaver macro F1 (95% CI) | Beaver violent F1 (95% CI) | Jigsaw FPR (95% CI) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for payload in payloads:
        lines.append(
            f"| `{payload['trial_id']}` | "
            f"{metric(payload['selection_performance']['macro_f1'])} | "
            f"{metric(payload['jigsaw_performance']['macro_f1'])} | "
            f"{metric(payload['beavertails_external']['macro_f1'])} | "
            f"{metric(payload['beavertails_external']['per_category']['harmful_violent']['f1'])} | "
            f"{percent(payload['jigsaw_fairness']['overall_test_fpr'])} |"
        )

    lines.extend(
        [
            "",
            "### Interim external-validity and fairness diagnostics",
            "",
            "| Trial | ToxiGen recall (95% CI) | ToxiGen F1 (95% CI) | "
            "HateCheck accuracy (95% CI) | HateCheck F1 (95% CI) | "
            "Civil identity FPR gap (95% CI) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for payload in payloads:
        tox = payload["toxigen_external"]["overall"]
        hate = payload["hatecheck_external"]["overall"]
        gap = payload["civil_comments_identity_fairness"]["fpr_gap_worst_minus_best"]
        lines.append(
            f"| `{payload['trial_id']}` | {metric(tox['recall'])} | "
            f"{metric(tox['f1'])} | {metric(hate['accuracy'])} | "
            f"{metric(hate['f1'])} | {metric(gap)} |"
        )

    families: dict[str, list[dict]] = {}
    for payload in payloads:
        families.setdefault(family_name(payload["trial_id"]), []).append(payload)
    lines.extend(["", "### What can be concluded at this checkpoint", ""])
    for name, family_payloads in families.items():
        selection_values = [
            item["selection_performance"]["macro_f1"]["value"]
            for item in family_payloads
        ]
        beaver_violent_values = [
            item["beavertails_external"]["per_category"]["harmful_violent"]["f1"][
                "value"
            ]
            for item in family_payloads
        ]
        lines.append(
            f"- **{name}:** selection macro F1 {mean_sd(selection_values)}; "
            f"BeaverTails harmful/violent F1 {mean_sd(beaver_violent_values)}."
        )

    adjacent_clear = all(
        payload["jigsaw_fairness"]["adjacent_benign_fpr"]["value"] == 0
        for payload in payloads
    )
    identity_intervals_cross_zero = all(
        payload["civil_comments_identity_fairness"]["fpr_gap_worst_minus_best"][
            "ci95"
        ][0]
        <= 0
        <= payload["civil_comments_identity_fairness"][
            "fpr_gap_worst_minus_best"
        ]["ci95"][1]
        for payload in payloads
    )
    lines.extend(
        [
            f"- **Direct violence supervision transfers:** completed trials produce "
            f"BeaverTails harmful/violent F1 from "
            f"{min(item['beavertails_external']['per_category']['harmful_violent']['f1']['value'] for item in payloads):.3f} "
            f"to {max(item['beavertails_external']['per_category']['harmful_violent']['f1']['value'] for item in payloads):.3f}. "
            "This is the clearest current mechanical signal from adding direct",
            "violence examples; it is not yet proof of final improvement.",
            f"- **Implicit hate remains weak:** ToxiGen F1 spans "
            f"{min(item['toxigen_external']['overall']['f1']['value'] for item in payloads):.3f}–"
            f"{max(item['toxigen_external']['overall']['f1']['value'] for item in payloads):.3f}, "
            f"with recall {min(item['toxigen_external']['overall']['recall']['value'] for item in payloads):.3f}–"
            f"{max(item['toxigen_external']['overall']['recall']['value'] for item in payloads):.3f}. "
            "High precision with low recall indicates domain/construct mismatch,",
            "not merely a threshold-free success.",
            f"- **Adjacent-benign behavior is stable so far:** "
            f"{'all completed trials have 0/60 flags' if adjacent_clear else 'at least one completed trial has a flag'}. "
            "Each 0/60 estimate still has a 6.02% Wilson upper bound.",
            f"- **Identity-gap claims are unresolved:** completed Civil Comments FPR "
            f"gap point estimates span "
            f"{min(item['civil_comments_identity_fairness']['fpr_gap_worst_minus_best']['value'] for item in payloads):.3f}–"
            f"{max(item['civil_comments_identity_fairness']['fpr_gap_worst_minus_best']['value'] for item in payloads):.3f}; "
            f"{'all intervals cross zero' if identity_intervals_cross_zero else 'not all intervals cross zero'}. "
            "Sparse identity intersections create wide intervals, so point-estimate",
            "rankings are not treated as demographic conclusions.",
            "- **No final winner exists yet:** capped-weighted BCE and focal-loss",
            "evaluations are absent from this snapshot. The preregistered selector",
            "must wait for 12/12 evaluations and uses model-selection data—not the",
            "frozen Jigsaw test.",
            "",
            "### Reproducibility and implemented safeguards",
            "",
            "- Exact 12-trial ledger: four locked losses × seeds 2025, 2026, 2027.",
            "- SHA-256 dataset manifest, pinned Hugging Face revisions, deterministic",
            "  Jigsaw splits, disjoint threshold-calibration/model-selection roles.",
            "- Per-category P/R/F1 and PR-AUC; Brier score, 15-bin ECE, threshold",
            "  sensitivity, fairness slices, bootstrap/Wilson/Newcombe intervals.",
            "- External evaluation on BeaverTails, ToxiGen, HateCheck, and original",
            "  Civil Comments identity columns; adjacent-benign clinical, news, and",
            "  reclaimed-language stress tests.",
            "- Blinded deterministic error-analysis sheet with two independent",
            "  annotation columns and Cohen's kappa; no fabricated annotations.",
            "- Generated model card and inference CLI with input hashing and a",
            "  high-impact-use warning. Raw harmful text and model weights remain",
            "  excluded from Git.",
            "",
            "Primary snapshot metrics and the generated interim report are in",
            "[`results/safety_v2_interim_2026-07-26/`](results/safety_v2_interim_2026-07-26/).",
            "Regenerate this entire section with:",
            "",
            "```bash",
            "python scripts/summarize_safety_v2_interim.py",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    snapshot_dir = Path(args.snapshot_dir)
    snapshot = load_json(snapshot_dir / "matrix_status.json")
    payloads = [
        load_json(path)
        for path in sorted(snapshot_dir.glob("*_seed*.json"))
    ]
    if not payloads:
        raise ValueError(f"No completed metric snapshots found in {snapshot_dir}")
    completed_ids = {
        row["trial_id"] for row in snapshot["trials"] if row["status"] == "complete"
    }
    payload_ids = {payload["trial_id"] for payload in payloads}
    if payload_ids != completed_ids:
        raise ValueError(
            "Snapshot metrics must exactly match completed trials: "
            f"metrics={sorted(payload_ids)}, ledger={sorted(completed_ids)}"
        )
    v1 = load_json(Path(args.v1_metrics))
    report = build_report(snapshot, payloads, v1)
    (snapshot_dir / "interim_report.md").write_text(report, encoding="utf-8")

    readme_path = Path(args.readme)
    readme = readme_path.read_text(encoding="utf-8")
    if START not in readme or END not in readme:
        raise ValueError("README safety-result markers are missing")
    before, remainder = readme.split(START, 1)
    _, after = remainder.split(END, 1)
    readme_path.write_text(
        before + START + "\n" + report + END + after,
        encoding="utf-8",
    )
    print(
        f"Wrote {snapshot_dir / 'interim_report.md'} and updated {readme_path} "
        f"from {len(payloads)}/{snapshot['n_trials_planned']} completed evaluations"
    )


if __name__ == "__main__":
    main()
