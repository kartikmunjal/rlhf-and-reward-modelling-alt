#!/usr/bin/env python3
"""Generate a pilot gate report directly from its metrics artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.metrics.read_text(encoding="utf-8"))
    metrics, gates, decisions = result["metrics"], result["gates"], result["gate_results"]
    lines = [
        f"# {result['study_id']} — feasibility result",
        "",
        f"**Status: {result['status'].upper()}**",
        "",
        f"Primary data: {metrics['n_problems']} disjoint problems and {metrics['n_completions']} sampled completions.",
        "",
        "| Gate | Observed | Threshold | Result |",
        "|---|---:|---:|---:|",
    ]
    rows = [
        ("Numeric parse rate", "numeric_parse_rate", "numeric_parse_rate_min", ">="),
        ("Numeric exact rate (minimum)", "numeric_exact_rate", "numeric_exact_rate_min", ">="),
    ]
    if "numeric_exact_rate_max" in gates:
        rows.append(("Numeric exact rate (maximum)", "numeric_exact_rate", "numeric_exact_rate_max", "<="))
    rows += [
        ("Truncation rate", "truncation_rate", "truncation_rate_max", "<="),
        ("Groups with reward contrast", "groups_with_reward_contrast", "groups_with_reward_contrast_min", ">="),
    ]
    for label, metric_key, gate_key, operator in rows:
        decision_key = metric_key
        if metric_key == "numeric_exact_rate" and gate_key.endswith("_min"):
            decision_key = "numeric_exact_rate_min" if "numeric_exact_rate_min" in decisions else "numeric_exact_rate"
        elif metric_key == "numeric_exact_rate" and gate_key.endswith("_max"):
            decision_key = "numeric_exact_rate_max"
        lines.append(
            f"| {label} | {metrics[metric_key]:.4f} | {operator} {gates[gate_key]:.4f} | "
            f"{'PASS' if decisions[decision_key] else 'FAIL'} |"
        )
    lines += [
        "",
        f"Reward mean: {metrics['reward_mean']:.4f}; reward variance: {metrics['reward_variance']:.6f}; "
        f"tagged exact rate: {metrics['tagged_exact_rate']:.4f}.",
        "",
        "A failed pilot is retained as a result and cannot be converted into a pass by changing its frozen thresholds.",
    ]
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
