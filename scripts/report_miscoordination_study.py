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
        f"[boot {100 * cell['ci95'][0]:.1f}%, {100 * cell['ci95'][1]:.1f}%; "
        f"Wilson {100 * cell['wilson_ci95'][0]:.1f}%, "
        f"{100 * cell['wilson_ci95'][1]:.1f}%]"
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
        "Wilson rate intervals are included as a post-run boundary sensitivity",
        "because ordinary bootstrap intervals collapse at observed rates of 0% or 100%.",
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
            "## Coordination overhead",
            "",
            "Means use episode-bootstrap 95% intervals; differences are paired.",
            "",
            "| Metric | Isolated mean (95% CI) | Shared-ledger mean (95% CI) | Paired difference (95% CI) |",
            "|---|---:|---:|---:|",
        ]
    )
    for outcome in ("api_calls", "input_tokens", "output_tokens", "cost_usd", "actions", "messages"):
        left, right, delta = isolated[outcome], ledger[outcome], paired[outcome]
        digits = 5 if outcome == "cost_usd" else 2
        lines.append(
            f"| `{outcome}` | {left['value']:.{digits}f} "
            f"[{left['ci95'][0]:.{digits}f}, {left['ci95'][1]:.{digits}f}] | "
            f"{right['value']:.{digits}f} "
            f"[{right['ci95'][0]:.{digits}f}, {right['ci95'][1]:.{digits}f}] | "
            f"{delta['value']:+.{digits}f} "
            f"[{delta['ci95'][0]:+.{digits}f}, {delta['ci95'][1]:+.{digits}f}] |"
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


def render_readme(payload: dict) -> str:
    analysis = payload["analysis"]
    isolated = analysis["by_condition"]["isolated"]
    ledger = analysis["by_condition"]["shared_ledger"]
    paired = analysis["paired"]
    report = render(payload).replace(
        "# Multi-agent miscoordination v1",
        "### Multi-agent miscoordination study — completed",
        1,
    ).replace("\n## Coordination overhead", "\n#### Coordination overhead").rstrip()
    conclusions = [
        "",
        "#### Interpretation",
        "",
        f"- The shared ledger changed any-miscoordination by "
        f"{difference(paired['any_miscoordination'])}; the observed mechanism was "
        f"redundant work ({rate(isolated['redundant_work'])} isolated versus "
        f"{rate(ledger['redundant_work'])} with the ledger).",
        f"- Final global success was {rate(isolated['global_success'])} in isolation "
        f"and {rate(ledger['global_success'])} with the ledger. The benchmark therefore "
        "shows an efficiency/process effect, not an outcome-accuracy improvement.",
        f"- The ledger reduced actions by {abs(paired['actions']['value']):.2f} per episode "
        f"(paired 95% CI {paired['actions']['ci95'][0]:+.2f} to "
        f"{paired['actions']['ci95'][1]:+.2f}) while adding "
        f"{paired['input_tokens']['value']:.0f} input tokens "
        f"({paired['input_tokens']['ci95'][0]:.0f} to "
        f"{paired['input_tokens']['ci95'][1]:.0f}) and "
        f"${paired['cost_usd']['value']:.5f} per episode.",
        f"- Silent undo and communication breakdown were not observed; their Wilson "
        f"upper bounds are {100 * ledger['silent_undo']['wilson_ci95'][1]:.1f}% per "
        "condition. The task was too easy to estimate severe-failure rates, so a harder "
        "follow-up would require a new preregistration rather than post-hoc task changes.",
        "",
        "Regenerate the metrics and this section from the local raw ledger with",
        "`scripts/analyze_miscoordination_results.py` and",
        "`scripts/update_miscoordination_readme.py`.",
    ]
    return report + "\n" + "\n".join(conclusions) + "\n"


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
