"""
CLI entry point: run AgentBench-Mini.

Usage
-----
    # Full benchmark: all tasks, all three agent configurations
    export ANTHROPIC_API_KEY=sk-ant-...
    python eval/run_benchmark.py

    # Only one category
    python eval/run_benchmark.py --category tool_use

    # Only one agent
    python eval/run_benchmark.py --agents react

    # Quick smoke test (first 3 tasks per category)
    python eval/run_benchmark.py --max_per_category 3

    # Use GPT-2 class model instead of Haiku (cheaper)
    python eval/run_benchmark.py --model claude-haiku-4-5-20251001

    # Use live web search (requires SERPER_API_KEY)
    python eval/run_benchmark.py --live_search
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def parse_args():
    p = argparse.ArgumentParser(description="Run AgentBench-Mini")
    p.add_argument("--category", default=None,
                   choices=["tool_use", "multi_step", "failure_recovery"],
                   help="Run only one task category (default: all)")
    p.add_argument("--agents", nargs="+",
                   default=["zero_shot", "react", "plan_and_execute"],
                   choices=["zero_shot", "react", "plan_and_execute"],
                   help="Which agent configs to evaluate")
    p.add_argument("--model", default="claude-haiku-4-5-20251001",
                   help="Claude model to use for all agents")
    p.add_argument("--max_per_category", type=int, default=None,
                   help="Limit to first N tasks per category (for quick testing)")
    p.add_argument("--live_search", action="store_true",
                   help="Use live web search via Serper API instead of mock tools")
    p.add_argument("--output", default="results/agentbench_results.json",
                   help="Where to save the benchmark report")
    p.add_argument("--sleep", type=float, default=0.5,
                   help="Seconds between API calls (rate limit buffer)")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable not set.")
        sys.exit(1)

    from eval.harness import AgentEvalHarness
    from eval.tools import get_default_tools
    from eval.agents import ZeroShotAgent, ReActAgent, PlanAndExecuteAgent

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

    print(f"Tasks: {len(tasks)} total  ({', '.join(f'{c}: {sum(1 for t in tasks if t.category==c)}' for c in ['tool_use','multi_step','failure_recovery'])})")

    # ── Load tools ─────────────────────────────────────────────────────────────
    tools = get_default_tools(use_live=args.live_search)
    print(f"Tools: {list(tools.keys())}  ({'live' if args.live_search else 'mock'})")

    # ── Build agents ───────────────────────────────────────────────────────────
    agent_map = {
        "zero_shot": ZeroShotAgent,
        "react": ReActAgent,
        "plan_and_execute": PlanAndExecuteAgent,
    }
    agents = [agent_map[name](model=args.model) for name in args.agents]
    print(f"Agents: {[a.name for a in agents]}  (model={args.model})\n")

    # ── Run benchmark ──────────────────────────────────────────────────────────
    harness = AgentEvalHarness(tasks, tools, sleep_between_tasks=args.sleep)
    report = harness.run_all(agents, verbose=True)

    # ── Save results ───────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    AgentEvalHarness.save_report(report, args.output)

    # ── Print category breakdown ───────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  PER-CATEGORY BREAKDOWN")
    print(f"{'='*70}")
    for cat in ["tool_use", "multi_step", "failure_recovery"]:
        print(f"\n  {cat}:")
        for agent in agents:
            acc = report.accuracy(cat, agent.name)
            seq = report.sequence_accuracy(cat, agent.name)
            n_calls = report.avg_tool_calls(cat, agent.name)
            print(f"    {agent.name:<25} acc={acc:.3f}  seq_acc={seq:.3f}  avg_calls={n_calls:.1f}")


if __name__ == "__main__":
    main()
