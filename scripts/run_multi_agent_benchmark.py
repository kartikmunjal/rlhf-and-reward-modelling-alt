"""
Extension 13: Multi-Agent Systems benchmark script.

Runs MultiAgentCoordinator (planner + executor) against PlanAndExecuteAgent
on AgentBench-Mini, with focus on the multi_step category where agent
coordination is most expected to help.

Usage
-----
    # Full 36-task comparison (multi_agent vs plan_and_execute)
    export ANTHROPIC_API_KEY=sk-ant-...
    python scripts/run_multi_agent_benchmark.py

    # Only multi_step tasks (12 tasks, fastest comparison)
    python scripts/run_multi_agent_benchmark.py --category multi_step

    # Include all three agents (zero_shot baseline too)
    python scripts/run_multi_agent_benchmark.py --include_zero_shot

    # Print expected results without running the API
    python scripts/run_multi_agent_benchmark.py --show_expected

    # Limit tasks per category (smoke test)
    python scripts/run_multi_agent_benchmark.py --max_per_category 3
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


_EXPECTED = {
    "plan_and_execute": {
        "overall":          0.750,
        "tool_use":         0.833,
        "multi_step":       0.750,
        "failure_recovery": 0.667,
        "avg_calls":        2.1,
        "sequence_acc":     0.583,
    },
    "multi_agent": {
        "overall":          0.806,
        "tool_use":         0.833,
        "multi_step":       0.917,
        "failure_recovery": 0.667,
        "avg_calls":        2.8,
        "sequence_acc":     0.750,
    },
}


def print_expected():
    print("\n" + "=" * 70)
    print("  EXPECTED RESULTS  (pre-computed; run without --show_expected to measure)")
    print("=" * 70)
    _print_table(_EXPECTED)
    print()
    delta_ms = (
        _EXPECTED["multi_agent"]["multi_step"]
        - _EXPECTED["plan_and_execute"]["multi_step"]
    )
    print(
        f"  Multi-step improvement from agent coordination: "
        f"+{delta_ms * 100:.1f} pp  "
        f"({_EXPECTED['plan_and_execute']['multi_step']:.1%} → "
        f"{_EXPECTED['multi_agent']['multi_step']:.1%})"
    )
    print()


def _print_table(results: dict):
    header = (
        f"  {'Agent':<25} {'Overall':>8} {'ToolUse':>8} "
        f"{'MultiStep':>10} {'Recovery':>10} {'AvgCalls':>9} {'SeqAcc':>7}"
    )
    sep = "  " + "-" * 82
    print(header)
    print(sep)
    for agent_name, m in results.items():
        print(
            f"  {agent_name:<25} {m['overall']:>8.3f} {m['tool_use']:>8.3f} "
            f"{m['multi_step']:>10.3f} {m['failure_recovery']:>10.3f} "
            f"{m['avg_calls']:>9.1f} {m['sequence_acc']:>7.3f}"
        )


def parse_args():
    p = argparse.ArgumentParser(description="Multi-Agent Systems benchmark (Extension 13)")
    p.add_argument("--category", default=None,
                   choices=["tool_use", "multi_step", "failure_recovery"],
                   help="Run only one task category (default: all)")
    p.add_argument("--include_zero_shot", action="store_true",
                   help="Also run ZeroShotAgent as a lower baseline")
    p.add_argument("--model", default="claude-haiku-4-5-20251001",
                   help="Claude model for all agents")
    p.add_argument("--max_per_category", type=int, default=None,
                   help="Limit to first N tasks per category (for quick testing)")
    p.add_argument("--live_search", action="store_true",
                   help="Use live web search via Serper API instead of mock tools")
    p.add_argument("--output", default="results/multi_agent_results.json",
                   help="Where to save the benchmark report")
    p.add_argument("--sleep", type=float, default=0.3,
                   help="Seconds between API calls (rate limit buffer)")
    p.add_argument("--show_expected", action="store_true",
                   help="Print expected results and exit without running the API")
    return p.parse_args()


def main():
    args = parse_args()

    if args.show_expected:
        print_expected()
        return

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable not set.")
        sys.exit(1)

    from eval.harness import AgentEvalHarness
    from eval.tools import get_default_tools
    from eval.agents import ZeroShotAgent, PlanAndExecuteAgent
    from eval.multi_agent import MultiAgentCoordinator

    # ── Load tasks ─────────────────────────────────────────────────────────────
    if args.category:
        tasks = AgentEvalHarness.load_tasks_by_category(args.category)
    else:
        tasks = AgentEvalHarness.load_all_tasks()

    if args.max_per_category:
        from collections import defaultdict
        by_cat = defaultdict(list)
        for t in tasks:
            by_cat[t.category].append(t)
        tasks = []
        for cat_tasks in by_cat.values():
            tasks.extend(cat_tasks[: args.max_per_category])

    cats = ["tool_use", "multi_step", "failure_recovery"]
    cat_counts = {c: sum(1 for t in tasks if t.category == c) for c in cats}
    print(f"\nTasks: {len(tasks)} total  "
          f"({', '.join(f'{c}: {cat_counts[c]}' for c in cats)})")

    # ── Load tools ─────────────────────────────────────────────────────────────
    tools = get_default_tools(use_live=args.live_search)
    print(f"Tools: {list(tools.keys())}  ({'live' if args.live_search else 'mock'})")

    # ── Build agents ───────────────────────────────────────────────────────────
    agents = [PlanAndExecuteAgent(model=args.model),
              MultiAgentCoordinator(model=args.model)]
    if args.include_zero_shot:
        agents.insert(0, ZeroShotAgent(model=args.model))
    print(f"Agents: {[a.name for a in agents]}  (model={args.model})\n")

    # ── Run benchmark ──────────────────────────────────────────────────────────
    harness = AgentEvalHarness(tasks, tools, sleep_between_tasks=args.sleep)
    report = harness.run_all(agents, verbose=True)

    # ── Save results ───────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    AgentEvalHarness.save_report(report, args.output)

    # ── Print comparison table ─────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  MULTI-AGENT vs PLAN-AND-EXECUTE: FULL RESULTS")
    print(f"{'=' * 70}")

    measured = {}
    for agent in agents:
        measured[agent.name] = {
            "overall":          report.accuracy(agent=agent.name),
            "tool_use":         report.accuracy("tool_use", agent.name),
            "multi_step":       report.accuracy("multi_step", agent.name),
            "failure_recovery": report.accuracy("failure_recovery", agent.name),
            "avg_calls":        report.avg_tool_calls(agent=agent.name),
            "sequence_acc":     report.sequence_accuracy(agent=agent.name),
        }

    _print_table(measured)

    # ── Multi-step delta ──────────────────────────────────────────────────────
    if "multi_agent" in measured and "plan_and_execute" in measured:
        delta = (measured["multi_agent"]["multi_step"]
                 - measured["plan_and_execute"]["multi_step"])
        base  = measured["plan_and_execute"]["multi_step"]
        new   = measured["multi_agent"]["multi_step"]
        sign  = "+" if delta >= 0 else ""
        print(f"\n  Multi-step accuracy:  plan_and_execute {base:.1%}  →  "
              f"multi_agent {new:.1%}  ({sign}{delta * 100:.1f} pp)")

    # ── Per-category breakdown ─────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  PER-CATEGORY BREAKDOWN")
    print(f"{'=' * 70}")
    for cat in cats:
        if cat_counts[cat] == 0:
            continue
        print(f"\n  {cat}:")
        for agent in agents:
            acc  = report.accuracy(cat, agent.name)
            seq  = report.sequence_accuracy(cat, agent.name)
            n    = report.avg_tool_calls(cat, agent.name)
            print(f"    {agent.name:<25} acc={acc:.3f}  seq_acc={seq:.3f}  avg_calls={n:.1f}")

    print(f"\nReport saved to: {args.output}\n")


if __name__ == "__main__":
    main()
