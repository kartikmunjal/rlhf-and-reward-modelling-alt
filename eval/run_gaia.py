"""
CLI entry point: run GAIA benchmark against agent configurations.

Usage
-----
    export ANTHROPIC_API_KEY=sk-ant-...

    # Full GAIA-Mini run (30 tasks × 3 agents)
    python eval/run_gaia.py

    # Level 1 only (quick)
    python eval/run_gaia.py --levels 1

    # Level 1 and 2, react agent only
    python eval/run_gaia.py --levels 1 2 --agents react

    # Use HuggingFace GAIA (165 tasks, requires dataset approval)
    python eval/run_gaia.py --use_hf

    # Smoke test: 2 tasks per level, 1 agent
    python eval/run_gaia.py --max_per_level 2 --agents react

    # Live web search (requires SERPER_API_KEY)
    python eval/run_gaia.py --live_search
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def parse_args():
    p = argparse.ArgumentParser(description="Run GAIA benchmark")
    p.add_argument("--levels", nargs="+", type=int, choices=[1, 2, 3], default=[1, 2, 3],
                   help="GAIA difficulty levels to evaluate (default: 1 2 3)")
    p.add_argument("--agents", nargs="+",
                   default=["zero_shot", "react", "plan_and_execute"],
                   choices=["zero_shot", "react", "plan_and_execute"],
                   help="Agent configurations to evaluate")
    p.add_argument("--model", default="claude-haiku-4-5-20251001",
                   help="Claude model for agents")
    p.add_argument("--max_per_level", type=int, default=None,
                   help="Limit tasks per level for quick testing")
    p.add_argument("--use_hf", action="store_true",
                   help="Load from HuggingFace gaia-benchmark/GAIA (requires dataset approval)")
    p.add_argument("--live_search", action="store_true",
                   help="Use live web search via Serper API (requires SERPER_API_KEY)")
    p.add_argument("--output", default="results/gaia_results.json",
                   help="Where to save the benchmark report")
    p.add_argument("--sleep", type=float, default=0.5,
                   help="Seconds between API calls")
    return p.parse_args()


def run_agent_on_task(agent, task, tools, sleep: float = 0.5):
    """Run a single GAIA task and return a GAIAResult."""
    from eval.gaia import GAIAResult, gaia_exact_match, gaia_token_overlap
    from eval.tasks.base import AgentTrajectory

    try:
        # Build a prompt that includes the GAIA question
        prompt = task.question
        traj = agent.run(prompt, tools)
        predicted = str(traj.final_answer).strip()
        n_calls = traj.n_tool_calls
        error = traj.error
        traj_text = "\n".join(traj.reasoning_steps) if traj.reasoning_steps else ""
    except Exception as e:
        predicted = ""
        n_calls = 0
        error = str(e)
        traj_text = ""

    time.sleep(sleep)

    exact = gaia_exact_match(predicted, task.ground_truth)
    overlap = gaia_token_overlap(predicted, task.ground_truth)

    return GAIAResult(
        task_id=task.task_id,
        level=task.level,
        agent_name=agent.name,
        predicted_answer=predicted,
        ground_truth=task.ground_truth,
        exact_match=exact,
        token_overlap=overlap,
        n_tool_calls=n_calls,
        trajectory=traj_text,
        error=error,
    )


def main():
    args = parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable not set.")
        sys.exit(1)

    from eval.gaia import load_gaia_tasks, GAIAReport
    from eval.agents import ZeroShotAgent, ReActAgent, PlanAndExecuteAgent
    from eval.tools import get_default_tools
    from tqdm.auto import tqdm

    # ── Load tasks ─────────────────────────────────────────────────────────────
    tasks = load_gaia_tasks(
        use_hf=args.use_hf,
        levels=args.levels,
        max_per_level=args.max_per_level,
    )
    by_level = {}
    for t in tasks:
        by_level.setdefault(t.level, []).append(t)
    print(f"Tasks: {len(tasks)} total — " +
          ", ".join(f"L{l}: {len(ts)}" for l, ts in sorted(by_level.items())))

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
    all_results = []

    for agent in agents:
        print(f"\nRunning agent: {agent.name}")
        pbar = tqdm(tasks, desc=f"  {agent.name}")
        for task in pbar:
            result = run_agent_on_task(agent, task, tools, sleep=args.sleep)
            all_results.append(result)
            pbar.set_postfix({
                "L": task.level,
                "em": f"{result.exact_match:.2f}",
                "f1": f"{result.token_overlap:.2f}",
            })

        # Per-agent summary
        agent_results = [r for r in all_results if r.agent_name == agent.name]
        for level in sorted(set(r.level for r in agent_results)):
            lvl_results = [r for r in agent_results if r.level == level]
            avg = sum(r.score for r in lvl_results) / len(lvl_results)
            print(f"    Level {level}: {avg:.3f} ({len(lvl_results)} tasks)")

    # ── Aggregate and save ─────────────────────────────────────────────────────
    from eval.gaia import GAIAReport
    report = GAIAReport(all_results)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    results_data = {
        "summary": {
            agent_name: {
                f"level_{l}": report.accuracy(l, agent_name)
                for l in [1, 2, 3]
            } | {
                "overall": report.accuracy(agent=agent_name),
                "avg_tool_calls": report.avg_tool_calls(agent=agent_name),
            }
            for agent_name in args.agents
        },
        "results": [
            {
                "task_id": r.task_id,
                "level": r.level,
                "agent_name": r.agent_name,
                "predicted_answer": r.predicted_answer,
                "ground_truth": r.ground_truth,
                "exact_match": r.exact_match,
                "token_overlap": r.token_overlap,
                "score": r.score,
                "n_tool_calls": r.n_tool_calls,
                "error": r.error,
            }
            for r in all_results
        ],
        "config": {
            "model": args.model,
            "levels": args.levels,
            "use_hf": args.use_hf,
            "n_tasks": len(tasks),
            "timestamp": time.time(),
        },
    }

    with open(args.output, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved → {args.output}")

    # ── Print summary table ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  GAIA BENCHMARK RESULTS")
    print(f"{'='*70}")
    print(report.summary_table(agents=args.agents))

    # Frontier model reference
    print(f"\nReference (frontier models on full 165-task GAIA validation):")
    print(f"  GPT-4 + tools:          L1=0.380  L2=0.160  L3=0.070  Overall=0.203")
    print(f"  GPT-4 + code_interp:    L1=0.670  L2=0.340  L3=0.140  Overall=0.383")
    print(f"  Claude 3 Opus + tools:  L1=0.650  L2=0.280  L3=0.100  Overall=0.343")


if __name__ == "__main__":
    main()
