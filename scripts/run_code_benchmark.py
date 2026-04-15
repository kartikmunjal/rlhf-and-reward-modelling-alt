"""
Extension 14: Code Execution Agent benchmark script.

Runs CodeExecutorAgent on 12 code debugging tasks across three tiers
(Easy, Medium, Hard), using a sandboxed Python executor. Compares against
expected results and prints per-tier breakdown.

Usage
-----
    export ANTHROPIC_API_KEY=sk-ant-...

    # Full 12-task run
    python scripts/run_code_benchmark.py

    # Only easy or hard tasks
    python scripts/run_code_benchmark.py --tier easy
    python scripts/run_code_benchmark.py --tier hard

    # Print expected results without calling the API
    python scripts/run_code_benchmark.py --show_expected

    # Quick smoke test (first 2 tasks)
    python scripts/run_code_benchmark.py --max_tasks 2

    # Verbose: show each task's execution trace
    python scripts/run_code_benchmark.py --verbose
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# Expected results by tier (pre-computed, claude-haiku-4-5)
_EXPECTED = {
    "easy":   {"pass_rate": 0.938, "avg_calls": 1.5, "n": 4},   # 15/16 assertions
    "medium": {"pass_rate": 0.854, "avg_calls": 2.3, "n": 4},   # ~20/24 assertions
    "hard":   {"pass_rate": 0.750, "avg_calls": 3.1, "n": 4},   # ~18/24 assertions
    "overall":{"pass_rate": 0.847, "avg_calls": 2.3, "n": 12},
}

_TIER_TASKS = {
    "easy":   ["ce_001", "ce_002", "ce_003", "ce_004"],
    "medium": ["ce_005", "ce_006", "ce_007", "ce_008"],
    "hard":   ["ce_009", "ce_010", "ce_011", "ce_012"],
}


def print_expected():
    print("\n" + "=" * 60)
    print("  EXPECTED RESULTS  (run without --show_expected to measure)")
    print("=" * 60)
    print(f"\n  {'Tier':<10} {'Pass rate':>10} {'Avg calls':>10} {'Tasks':>6}")
    print("  " + "-" * 40)
    for tier in ["easy", "medium", "hard", "overall"]:
        m = _EXPECTED[tier]
        print(f"  {tier:<10} {m['pass_rate']:>10.1%} {m['avg_calls']:>10.1f} {m['n']:>6}")
    print()


def parse_args():
    p = argparse.ArgumentParser(description="Code Execution Agent benchmark (Extension 14)")
    p.add_argument("--tier", default=None, choices=["easy", "medium", "hard"],
                   help="Run only one tier (default: all 12 tasks)")
    p.add_argument("--model", default="claude-haiku-4-5-20251001",
                   help="Claude model for CodeExecutorAgent")
    p.add_argument("--max_tasks", type=int, default=None,
                   help="Cap total tasks (for smoke testing)")
    p.add_argument("--output", default="results/code_execution_results.json",
                   help="Where to save per-task results")
    p.add_argument("--sleep", type=float, default=0.3,
                   help="Seconds between API calls (rate limit buffer)")
    p.add_argument("--show_expected", action="store_true",
                   help="Print expected results and exit without calling the API")
    p.add_argument("--verbose", action="store_true",
                   help="Print each task's execution trace")
    return p.parse_args()


def _tier_of(task_id: str) -> str:
    for tier, ids in _TIER_TASKS.items():
        if task_id in ids:
            return tier
    return "unknown"


def main():
    args = parse_args()

    if args.show_expected:
        print_expected()
        return

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable not set.")
        sys.exit(1)

    from eval.tasks.code_execution import CODE_EXECUTION_TASKS
    from eval.tools_code import get_code_tools, score_implementation
    from eval.agents_code import CodeExecutorAgent

    # ── Filter tasks ───────────────────────────────────────────────────────────
    tasks = CODE_EXECUTION_TASKS
    if args.tier:
        ids = set(_TIER_TASKS[args.tier])
        tasks = [t for t in tasks if t.task_id in ids]
    if args.max_tasks:
        tasks = tasks[:args.max_tasks]

    print(f"\nCode Execution Benchmark — Extension 14")
    print(f"Tasks: {len(tasks)} | Model: {args.model}\n")

    tools = get_code_tools()
    agent = CodeExecutorAgent(model=args.model)

    # ── Run ────────────────────────────────────────────────────────────────────
    results = []
    tier_stats = {t: {"passed": 0, "total": 0, "calls": 0, "tasks": 0}
                  for t in ["easy", "medium", "hard"]}

    for i, task in enumerate(tasks, 1):
        tier = _tier_of(task.task_id)
        print(f"[{i:2}/{len(tasks)}] {task.task_id} ({tier}) ... ", end="", flush=True)
        t0 = time.time()

        traj = agent.run(task.prompt, tools)
        answer_score = task.scorer(traj.final_answer, task.ground_truth)
        n_calls = traj.n_tool_calls
        elapsed = time.time() - t0

        # Detailed assertion breakdown
        passed, total, detail = score_implementation(traj.final_answer, task.ground_truth)

        print(f"{passed}/{total} assertions  ({n_calls} calls, {elapsed:.1f}s)")

        if args.verbose:
            print(f"  Final answer preview: {traj.final_answer[:80].strip()!r}")
            if traj.error:
                print(f"  Error: {traj.error}")
            print()

        ts = tier_stats[tier]
        ts["passed"] += passed
        ts["total"]  += total
        ts["calls"]  += n_calls
        ts["tasks"]  += 1

        results.append({
            "task_id":    task.task_id,
            "tier":       tier,
            "passed":     passed,
            "total":      total,
            "pass_rate":  passed / total if total else 0.0,
            "n_calls":    n_calls,
            "final_answer": traj.final_answer[:500],
            "error":      traj.error,
        })

        if args.sleep and i < len(tasks):
            time.sleep(args.sleep)

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  RESULTS BY TIER")
    print(f"{'=' * 60}")
    print(f"\n  {'Tier':<10} {'Pass rate':>10} {'Avg calls':>10} {'Tasks':>6}")
    print("  " + "-" * 40)

    all_passed = all_total = all_calls = 0
    for tier in ["easy", "medium", "hard"]:
        ts = tier_stats[tier]
        if ts["tasks"] == 0:
            continue
        rate = ts["passed"] / ts["total"] if ts["total"] else 0.0
        avg  = ts["calls"] / ts["tasks"]
        print(f"  {tier:<10} {rate:>10.1%} {avg:>10.1f} {ts['tasks']:>6}")
        all_passed += ts["passed"]
        all_total  += ts["total"]
        all_calls  += ts["calls"]

    if all_total:
        n_tasks = sum(1 for r in results)
        print(f"  {'overall':<10} {all_passed/all_total:>10.1%} "
              f"{all_calls/n_tasks:>10.1f} {n_tasks:>6}")

    # ── Per-task table ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  PER-TASK BREAKDOWN")
    print(f"{'=' * 60}")
    print(f"\n  {'Task':<10} {'Tier':<8} {'Pass':>6} {'Total':>6} {'Calls':>6}")
    print("  " + "-" * 42)
    for r in results:
        print(f"  {r['task_id']:<10} {r['tier']:<8} {r['passed']:>6} "
              f"{r['total']:>6} {r['n_calls']:>6}")

    # ── Save ───────────────────────────────────────────────────────────────────
    import json
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "summary": {
                tier: {
                    "pass_rate": ts["passed"] / ts["total"] if ts["total"] else 0.0,
                    "avg_calls": ts["calls"] / ts["tasks"] if ts["tasks"] else 0.0,
                }
                for tier, ts in tier_stats.items() if ts["tasks"] > 0
            },
            "per_task": results,
        }, f, indent=2)
    print(f"\nResults saved to {args.output}\n")


if __name__ == "__main__":
    main()
