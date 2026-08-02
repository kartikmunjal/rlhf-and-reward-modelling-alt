#!/usr/bin/env python3
"""Validate raw episodes and regenerate compact miscoordination results."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.miscoordination import (  # noqa: E402
    ROLE_PROMPTS,
    WORKER_SYSTEM,
    DeploymentState,
    SharedDeploymentEnvironment,
    bootstrap_study,
    classify_failures,
)
from scripts.report_miscoordination_study import write_report  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/miscoordination_v1.json")
    parser.add_argument("--episodes", default="results/miscoordination_v1/episodes.jsonl")
    parser.add_argument("--output-dir", default="results/miscoordination_v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    raw_path = Path(args.episodes)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    episodes = [
        json.loads(line)
        for line in raw_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_keys = {
        (pair_id, condition)
        for pair_id in range(config["matched_pairs"])
        for condition in config["conditions"]
    }
    observed_keys = {(row["pair_id"], row["condition"]) for row in episodes}
    if len(episodes) != config["episodes_total"] or observed_keys != expected_keys:
        raise ValueError(
            f"Episode ledger mismatch: rows={len(episodes)}, unique={len(observed_keys)}, "
            f"expected={len(expected_keys)}"
        )
    for row in episodes:
        environment = SharedDeploymentEnvironment(
            state=DeploymentState(**row["final_state"]),
            events=row["events"],
            messages=row["messages"],
        )
        recomputed = classify_failures(environment)
        recorded = {name: row[name] for name in recomputed}
        if recomputed != recorded or environment.state.global_success != row["global_success"]:
            raise ValueError(f"Mechanical label mismatch in pair {row['pair_id']} {row['condition']}")

    analysis = bootstrap_study(
        episodes, config["bootstrap_replicates"], config["bootstrap_seed"]
    )
    manifest = {
        "study": config["study"],
        "preregistration": "docs/miscoordination_v1_preregistered_plan.md",
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "episodes_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "model": config["model"],
        "temperature": config["temperature"],
        "worker_system_sha256": hashlib.sha256(WORKER_SYSTEM.encode()).hexdigest(),
        "role_prompts_sha256": hashlib.sha256(
            json.dumps(ROLE_PROMPTS, sort_keys=True).encode()
        ).hexdigest(),
        "n_episodes": len(episodes),
        "n_api_calls": sum(row["api_usage"]["calls"] for row in episodes),
        "input_tokens": sum(row["api_usage"]["input_tokens"] for row in episodes),
        "output_tokens": sum(row["api_usage"]["output_tokens"] for row in episodes),
        "cost_usd": sum(row["api_usage"]["cost_usd"] for row in episodes),
        "api_errors": sum(row["api_error"] is not None for row in episodes),
        "analysis_note": (
            "Preregistered bootstrap intervals are primary; Wilson intervals are a "
            "boundary-sensitivity analysis added after observing degenerate 0/1 bootstrap bounds."
        ),
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics.json"
    metrics_path.write_text(
        json.dumps({"manifest": manifest, "analysis": analysis}, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(metrics_path, output / "report.md")
    print(f"Validated {len(episodes)} episodes and wrote {metrics_path}")


if __name__ == "__main__":
    main()
