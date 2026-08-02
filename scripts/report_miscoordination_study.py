#!/usr/bin/env python3
"""Generate the compact miscoordination v1 Markdown report from metrics.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

OUTCOMES = (
    "global_success",
    "any_miscoordination",
    "redundant_work",
    "direct_contradiction",
    "silent_undo",
    "communication_breakdown",
)


def rate(cell: dict) -> str:
    return (
        f"{100 * cell['value']:.1f}% "
        f"[{100 * cell['ci95'][0]:.1f}%, {100 * cell['ci95'][1]:.1f}%]"
    )


def difference(cell: dict) -> str:
    return (
        f"{100 * cell['value']:+.1f} pp "
        f"[{100 * cell['ci95'][0]:+.1f}, {100 * cell['ci95'][1]:+.1f}]"
    )


def render(payload: dict) -> str:
    manifest = payload["manifest"]
    analysis = payload["analysis"]
    isolated = analysis["by_condition"]["isolated"]
    ledger = analysis["by_condition"]["shared_ledger"]
    paired = analysis["paired"]
    lines = [
        "# Multi-agent miscoordination v1",
        "",
        f"- Episodes: {manifest['n_episodes']} ({paired['n_matched_pairs']} matched pairs)",
        f"- API calls: {manifest['n_api_calls']}",
        f"- Model: `{manifest['model']}`",
        f"- Measured API cost: ${manifest['cost_usd']:.4f}",
        f"- API-error episodes: {manifest['api_errors']}",
        f"- Bootstrap replicates: {analysis['n_bootstrap']}",
        "",
        "Rates are episode-level with deterministic bootstrap 95% intervals.",
        "Differences are paired shared-ledger minus isolated estimates.",
        "",
        "| Outcome | Isolated (95% CI) | Shared ledger (95% CI) | Paired difference (95% CI) |",
        "|---|---:|---:|---:|",
    ]
    for outcome in OUTCOMES:
        lines.append(
            f"| `{outcome}` | {rate(isolated[outcome])} | {rate(ledger[outcome])} | "
            f"{difference(paired[outcome])} |"
        )
    lines.extend(
        [
            "",
            "The taxonomy is mechanically derived from the shared-state event log;",
            "no language-model judge assigns failure labels. Categories may co-occur.",
            "Interpretation is limited to this controlled deployment task and model.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(metrics_path: Path, output_path: Path) -> None:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render(payload), encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="results/miscoordination_v1/metrics.json")
    parser.add_argument("--output", default="results/miscoordination_v1/report.md")
    args = parser.parse_args()
    write_report(Path(args.metrics), Path(args.output))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
