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
from pathlib import Path
from typing import Optional

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
    p.add_argument("--benchmark_mode", choices=["official", "live"], default="official",
                   help="Benchmark framing: official-style evaluation or live-web evaluation")
    p.add_argument("--attachment_root", default=None,
                   help="Optional local directory containing GAIA attachments referenced by file_name")
    p.add_argument("--resume", action="store_true",
                   help="Resume from an existing output JSON and skip completed agent/task pairs")
    p.add_argument("--strict_hf", action="store_true",
                   help="Fail if --use_hf is requested but the full HuggingFace dataset cannot be loaded")
    p.add_argument("--artifacts_dir", default="results/gaia_artifacts",
                   help="Directory to store per-task artifacts")
    p.add_argument("--output", default="results/gaia_results.json",
                   help="Where to save the benchmark report")
    p.add_argument("--sleep", type=float, default=0.5,
                   help="Seconds between API calls")
    return p.parse_args()


def run_agent_on_task(
    agent,
    task,
    tools,
    benchmark_mode: str = "official",
    attachment_path: Optional[str] = None,
    sleep: float = 0.5,
    task_source: str = "gaia_mini",
):
    """Run a single GAIA task and return a GAIAResult."""
    from eval.gaia import (
        GAIAResult,
        build_task_prompt,
        gaia_exact_match,
        gaia_token_overlap,
    )

    prompt = build_task_prompt(task, benchmark_mode=benchmark_mode, attachment_path=attachment_path)
    started = time.time()
    try:
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
    runtime_sec = time.time() - started

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
        benchmark_mode=benchmark_mode,
        task_source=task_source,
        attachment_name=task.file_name,
        attachment_available=bool(attachment_path),
        runtime_sec=runtime_sec,
    )


def _load_existing_results(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            data = json.load(f)
        return data.get("results", [])
    except Exception:
        return []


def _dict_to_result(item):
    from eval.gaia import GAIAResult

    return GAIAResult(
        task_id=item["task_id"],
        level=item["level"],
        agent_name=item["agent_name"],
        predicted_answer=item.get("predicted_answer", ""),
        ground_truth=item.get("ground_truth", ""),
        exact_match=item.get("exact_match", 0.0),
        token_overlap=item.get("token_overlap", 0.0),
        n_tool_calls=item.get("n_tool_calls", 0),
        trajectory=item.get("trajectory"),
        error=item.get("error"),
        benchmark_mode=item.get("benchmark_mode", "official"),
        task_source=item.get("task_source", "gaia_mini"),
        attachment_name=item.get("attachment_name"),
        attachment_available=item.get("attachment_available", False),
        runtime_sec=item.get("runtime_sec", 0.0),
    )


def _result_to_dict(r) -> dict:
    return {
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
        "benchmark_mode": r.benchmark_mode,
        "task_source": r.task_source,
        "attachment_name": r.attachment_name,
        "attachment_available": r.attachment_available,
        "runtime_sec": r.runtime_sec,
    }


def _write_report(args, tasks, all_results, task_source: str) -> None:
    from eval.gaia import GAIAReport

    report = GAIAReport(all_results)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    results_data = {
        "summary": {
            agent_name: {
                f"level_{l}": report.accuracy(l, agent_name, metric="exact")
                for l in [1, 2, 3]
            } | {
                "overall_exact": report.accuracy(agent=agent_name, metric="exact"),
                "overall_relaxed": report.accuracy(agent=agent_name, metric="score"),
                "avg_tool_calls": report.avg_tool_calls(agent=agent_name),
                "ci_exact": list(report.confidence_interval(agent=agent_name, metric="exact", seed=0)),
            }
            for agent_name in sorted({r.agent_name for r in all_results})
        },
        "results": [_result_to_dict(r) for r in all_results],
        "config": {
            "model": args.model,
            "levels": args.levels,
            "use_hf": args.use_hf,
            "benchmark_mode": args.benchmark_mode,
            "live_search": args.live_search,
            "attachment_root": args.attachment_root,
            "task_source": task_source,
            "n_tasks": len(tasks),
            "timestamp": time.time(),
        },
    }

    with open(args.output, "w") as f:
        json.dump(results_data, f, indent=2)


def _write_task_artifact(artifacts_dir: str, agent_name: str, result, attachment_path: Optional[str]) -> None:
    os.makedirs(artifacts_dir, exist_ok=True)
    path = Path(artifacts_dir) / f"{agent_name}__{result.task_id}.json"
    with open(path, "w") as f:
        json.dump(
            {
                "task_id": result.task_id,
                "agent_name": agent_name,
                "level": result.level,
                "benchmark_mode": result.benchmark_mode,
                "attachment_path": attachment_path,
                "predicted_answer": result.predicted_answer,
                "ground_truth": result.ground_truth,
                "exact_match": result.exact_match,
                "token_overlap": result.token_overlap,
                "n_tool_calls": result.n_tool_calls,
                "runtime_sec": result.runtime_sec,
                "error": result.error,
                "trajectory": result.trajectory,
            },
            f,
            indent=2,
        )


def main():
    args = parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable not set.")
        sys.exit(1)

    from eval.gaia import GAIAReport, load_gaia_tasks, resolve_attachment_path
    from eval.agents import ZeroShotAgent, ReActAgent, PlanAndExecuteAgent
    from eval.tools import get_default_tools
    from tqdm.auto import tqdm

    # ── Load tasks ─────────────────────────────────────────────────────────────
    tasks = load_gaia_tasks(
        use_hf=args.use_hf,
        levels=args.levels,
        max_per_level=args.max_per_level,
    )
    task_source = "huggingface" if args.use_hf and len(tasks) > 30 else "gaia_mini"
    if args.use_hf and args.strict_hf and task_source != "huggingface":
        print("ERROR: --strict_hf requested but the full HuggingFace GAIA dataset was not loaded.")
        sys.exit(1)

    by_level = {}
    for t in tasks:
        by_level.setdefault(t.level, []).append(t)
    print(f"Tasks: {len(tasks)} total — " +
          ", ".join(f"L{l}: {len(ts)}" for l, ts in sorted(by_level.items())))

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
    completed = set()
    if args.resume:
        existing = _load_existing_results(args.output)
        for item in existing:
            completed.add((item["agent_name"], item["task_id"]))
            all_results.append(_dict_to_result(item))
        print(f"Resume: loaded {len(existing)} existing task results from {args.output}")

    for agent in agents:
        print(f"\nRunning agent: {agent.name}")
        pbar = tqdm(tasks, desc=f"  {agent.name}")
        for task in pbar:
            if (agent.name, task.task_id) in completed:
                continue

            attachment_path = resolve_attachment_path(task, args.attachment_root)
            attachments = {Path(attachment_path).name: attachment_path} if attachment_path else None
            tools = get_default_tools(use_live=args.live_search, attachments=attachments)

            result = run_agent_on_task(
                agent,
                task,
                tools,
                benchmark_mode=args.benchmark_mode,
                attachment_path=attachment_path,
                sleep=args.sleep,
                task_source=task_source,
            )
            all_results.append(result)
            _write_task_artifact(args.artifacts_dir, agent.name, result, attachment_path)
            _write_report(args, tasks, all_results, task_source)
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
    report = GAIAReport(all_results)
    _write_report(args, tasks, all_results, task_source)
    print(f"\nResults saved → {args.output}")

    # ── Print summary table ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  GAIA BENCHMARK RESULTS")
    print(f"{'='*70}")
    print("Official exact-match view:")
    print(report.summary_table(agents=args.agents, metric="exact"))
    print("\nRelaxed view (L1 exact, L2/L3 token-overlap):")
    print(report.summary_table(agents=args.agents, metric="score"))

    # Frontier model reference
    print(f"\nReference (frontier models on full 165-task GAIA validation):")
    print(f"  GPT-4 + tools:          L1=0.380  L2=0.160  L3=0.070  Overall=0.203")
    print(f"  GPT-4 + code_interp:    L1=0.670  L2=0.340  L3=0.140  Overall=0.383")
    print(f"  Claude 3 Opus + tools:  L1=0.650  L2=0.280  L3=0.100  Overall=0.343")


if __name__ == "__main__":
    main()
