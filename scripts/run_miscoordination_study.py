#!/usr/bin/env python3
"""Run the preregistered API-bound multi-agent miscoordination study."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.miscoordination import (  # noqa: E402
    MiscoordinationWorker,
    ROLE_PROMPTS,
    SharedDeploymentEnvironment,
    WORKER_SYSTEM,
    bootstrap_study,
    classify_failures,
)
from scripts.report_miscoordination_study import write_report  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/miscoordination_v1.json")
    parser.add_argument("--output-dir", default="results/miscoordination_v1")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-pairs", type=int, help="Diagnostic pilot only")
    return parser.parse_args()


def cost(config: dict[str, Any], input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens * config["haiku_input_usd_per_million_tokens"]
        + output_tokens * config["haiku_output_usd_per_million_tokens"]
    ) / 1_000_000


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.max_pairs is not None and args.output_dir == "results/miscoordination_v1":
        raise SystemExit("Diagnostic --max-pairs requires a non-production --output-dir")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is required for the live study")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / "episodes.jsonl"
    existing: list[dict[str, Any]] = []
    if args.resume and raw_path.exists():
        existing = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]
    complete_keys = {(row["pair_id"], row["condition"]) for row in existing}
    cumulative_cost = sum(row["api_usage"]["cost_usd"] for row in existing)
    workers = {
        role: MiscoordinationWorker(
            role,
            config["model"],
            config["max_output_tokens_per_call"],
            config["temperature"],
        )
        for role in ("performance", "reliability")
    }
    pair_count = config["matched_pairs"]
    if args.max_pairs is not None:
        pair_count = min(pair_count, args.max_pairs)
    order_rng = random.Random(config["order_seed"])
    starts = ["performance", "reliability"] * ((config["matched_pairs"] + 1) // 2)
    order_rng.shuffle(starts)

    for pair_id in range(pair_count):
        first = starts[pair_id]
        second = "reliability" if first == "performance" else "performance"
        turn_order = [first, second, first, second]
        for condition in config["conditions"]:
            if (pair_id, condition) in complete_keys:
                continue
            env = SharedDeploymentEnvironment()
            usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
            api_error = None
            for turn, actor in enumerate(turn_order, start=1):
                if cumulative_cost >= config["max_api_cost_usd"]:
                    raise RuntimeError("Hard API cost ceiling reached before study completion")
                try:
                    payload, call_usage = workers[actor].act(
                        env.visible_context(condition), condition, turn
                    )
                    input_tokens = call_usage["input_tokens"]
                    output_tokens = call_usage["output_tokens"]
                    call_cost = cost(config, input_tokens, output_tokens)
                    usage["calls"] += 1
                    usage["input_tokens"] += input_tokens
                    usage["output_tokens"] += output_tokens
                    usage["cost_usd"] += call_cost
                    cumulative_cost += call_cost
                    env.apply_turn(actor, payload)
                except Exception as error:  # API failures are disclosed, never dropped
                    api_error = f"{type(error).__name__}: {error}"
                    break
            flags = classify_failures(env)
            episode = {
                "study": config["study"],
                "pair_id": pair_id,
                "condition": condition,
                "turn_order": turn_order,
                "global_success": env.state.global_success,
                **flags,
                "final_state": env.state.__dict__,
                "events": env.events,
                "messages": env.messages,
                "api_error": api_error,
                "api_usage": usage,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            with raw_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(episode, ensure_ascii=False) + "\n")
            existing.append(episode)
            print(
                f"pair={pair_id:02d} condition={condition} success={env.state.global_success} "
                f"miscoordination={flags['any_miscoordination']} cost=${cumulative_cost:.4f}"
            )

    expected = pair_count * len(config["conditions"])
    if len(existing) < expected:
        raise RuntimeError(f"Incomplete study: {len(existing)}/{expected} episodes")
    analysis = bootstrap_study(
        existing, config["bootstrap_replicates"], config["bootstrap_seed"]
    )
    manifest = {
        "study": config["study"],
        "preregistration": "docs/miscoordination_v1_preregistered_plan.md",
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "model": config["model"],
        "temperature": config["temperature"],
        "worker_system_sha256": hashlib.sha256(WORKER_SYSTEM.encode()).hexdigest(),
        "role_prompts_sha256": hashlib.sha256(
            json.dumps(ROLE_PROMPTS, sort_keys=True).encode()
        ).hexdigest(),
        "n_episodes": len(existing),
        "n_api_calls": sum(row["api_usage"]["calls"] for row in existing),
        "input_tokens": sum(row["api_usage"]["input_tokens"] for row in existing),
        "output_tokens": sum(row["api_usage"]["output_tokens"] for row in existing),
        "cost_usd": sum(row["api_usage"]["cost_usd"] for row in existing),
        "api_errors": sum(row["api_error"] is not None for row in existing),
    }
    metrics_path = output / "metrics.json"
    metrics_path.write_text(
        json.dumps({"manifest": manifest, "analysis": analysis}, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(metrics_path, output / "report.md")
    print(f"Wrote {metrics_path}; total cost=${manifest['cost_usd']:.4f}")


if __name__ == "__main__":
    main()
