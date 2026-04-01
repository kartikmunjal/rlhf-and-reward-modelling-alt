"""
AgentEvalHarness — runs any agent against the full AgentBench-Mini task set.

Design goals:
  - Swap in different agents without changing any task or scoring code
  - Capture the full trajectory (tool calls + reasoning) for process evaluation
  - Aggregate results into a BenchmarkReport with per-category and per-agent stats
  - Fail gracefully: one task error does not abort the whole run
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

from tqdm.auto import tqdm

from eval.tasks.base import (
    AgentTrajectory,
    BenchmarkReport,
    EvalResult,
    EvalTask,
)
from eval.tools import Tool


class AgentEvalHarness:
    """Runs one or more agents against a task set and collects results.

    Parameters
    ----------
    tasks:
        List of EvalTask objects (from tool_use, multi_step, failure_recovery).
    tools:
        Dict mapping tool name → Tool callable.
    sleep_between_tasks:
        Seconds to sleep between API calls (avoids rate limiting).
    """

    def __init__(
        self,
        tasks: List[EvalTask],
        tools: Dict[str, Tool],
        sleep_between_tasks: float = 0.5,
    ):
        self.tasks = tasks
        self.tools = tools
        self.sleep_between_tasks = sleep_between_tasks

    # ── Single task ────────────────────────────────────────────────────────────

    def run_task(self, agent, task: EvalTask) -> EvalResult:
        """Run a single task with a single agent; return scored EvalResult."""
        try:
            trajectory = agent.run(task.prompt, self.tools)
        except Exception as e:
            trajectory = AgentTrajectory(
                task_id=task.task_id,
                agent_name=agent.name,
                prompt=task.prompt,
                final_answer="",
                error=str(e),
            )

        trajectory.task_id = task.task_id

        # Score final answer
        try:
            answer_score = float(
                task.scorer(trajectory.final_answer, task.ground_truth)
            )
        except Exception:
            answer_score = 0.0

        # Score tool call sequence (if scorer provided)
        if task.sequence_scorer is not None:
            try:
                sequence_score = float(task.sequence_scorer(trajectory.tool_calls))
            except Exception:
                sequence_score = 0.0
        else:
            sequence_score = 1.0  # no sequence requirement → always correct

        return EvalResult(
            task_id=task.task_id,
            category=task.category,
            agent_name=agent.name,
            answer_score=answer_score,
            sequence_score=sequence_score,
            trajectory=trajectory,
            ground_truth=task.ground_truth,
            predicted_answer=str(trajectory.final_answer),
        )

    # ── One agent, all tasks ───────────────────────────────────────────────────

    def run_agent(self, agent, verbose: bool = True) -> List[EvalResult]:
        """Run all tasks for one agent."""
        results = []
        pbar = tqdm(self.tasks, desc=f"  {agent.name}", leave=False)
        for task in pbar:
            result = self.run_task(agent, task)
            results.append(result)
            pbar.set_postfix({
                "ans": f"{result.answer_score:.2f}",
                "seq": f"{result.sequence_score:.2f}",
                "tools": result.trajectory.n_tool_calls,
            })
            time.sleep(self.sleep_between_tasks)

        if verbose:
            acc = sum(r.answer_score for r in results) / len(results)
            print(f"  {agent.name:<25} accuracy={acc:.3f}  n={len(results)}")

        return results

    # ── Multiple agents, all tasks ─────────────────────────────────────────────

    def run_all(self, agents: list, verbose: bool = True) -> BenchmarkReport:
        """Run all agents against all tasks; return a BenchmarkReport.

        Parameters
        ----------
        agents:
            List of agent instances (ZeroShotAgent, ReActAgent, PlanAndExecuteAgent, …)
        """
        all_results: List[EvalResult] = []

        print(f"\nRunning AgentBench-Mini: {len(self.tasks)} tasks × {len(agents)} agents")
        print(f"  Categories: {sorted({t.category for t in self.tasks})}")
        print()

        for agent in agents:
            results = self.run_agent(agent, verbose=verbose)
            all_results.extend(results)

        report = BenchmarkReport(all_results)

        if verbose:
            print(f"\n{'='*70}")
            print("  BENCHMARK RESULTS")
            print(f"{'='*70}")
            print(report.summary_table())

        return report

    # ── Convenience: load all standard tasks ──────────────────────────────────

    @staticmethod
    def load_all_tasks() -> List[EvalTask]:
        """Return the complete AgentBench-Mini task set (all three categories)."""
        from eval.tasks.tool_use import TOOL_USE_TASKS
        from eval.tasks.multi_step import MULTI_STEP_TASKS
        from eval.tasks.failure_recovery import FAILURE_RECOVERY_TASKS
        return TOOL_USE_TASKS + MULTI_STEP_TASKS + FAILURE_RECOVERY_TASKS

    @staticmethod
    def load_tasks_by_category(category: str) -> List[EvalTask]:
        """Return tasks for a single category."""
        return [t for t in AgentEvalHarness.load_all_tasks() if t.category == category]

    # ── Save / load results ───────────────────────────────────────────────────

    @staticmethod
    def save_report(report: BenchmarkReport, path: str) -> None:
        report.to_json(path)
        print(f"Report saved to {path}")

    @staticmethod
    def load_report(path: str) -> dict:
        with open(path) as f:
            return json.load(f)
