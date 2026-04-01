"""
Base data structures for AgentBench-Mini.

EvalTask       — a single benchmark task with prompt, ground truth, and scorers
AgentTrajectory — the full record of an agent run (tool calls + final answer)
EvalResult     — outcome of running one task (scores + trajectory)
BenchmarkReport — aggregate results across all tasks
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


# ── Task ──────────────────────────────────────────────────────────────────────

@dataclass
class EvalTask:
    """A single benchmark task.

    Parameters
    ----------
    task_id:
        Unique identifier (e.g., "tool_use_001").
    category:
        One of "tool_use", "multi_step", "failure_recovery".
    prompt:
        The question or instruction given to the agent.
    ground_truth:
        The correct answer (string, number, list, etc.).
    scorer:
        Callable(agent_answer, ground_truth) → float in [0, 1].
        For exact match this is 0 or 1; for fuzzy match it can be continuous.
    sequence_scorer:
        Optional callable(tool_calls: List[ToolCall]) → float in [0, 1].
        Measures whether the agent's tool-call sequence was correct,
        independent of the final answer.  Analogous to PRM vs ORM.
    expected_tool_sequence:
        Optional list of expected tool names in order.  Used by default
        sequence scorers.
    max_tool_calls:
        Hard cap on tool calls the agent may make for this task.
    notes:
        Human-readable description of what makes this task hard.
    """

    task_id: str
    category: str  # "tool_use" | "multi_step" | "failure_recovery"
    prompt: str
    ground_truth: Any
    scorer: Callable[[Any, Any], float]
    sequence_scorer: Optional[Callable[[List], float]] = None
    expected_tool_sequence: Optional[List[str]] = None
    max_tool_calls: int = 5
    notes: str = ""


# ── Tool call record ──────────────────────────────────────────────────────────

@dataclass
class ToolCall:
    tool_name: str
    arguments: Dict[str, Any]
    result: Any          # what the tool returned
    step: int            # which reasoning step this occurred in


# ── Trajectory ────────────────────────────────────────────────────────────────

@dataclass
class AgentTrajectory:
    """Full record of one agent run.

    Captures everything needed for both answer scoring (outcome) and
    tool-call sequence scoring (process) — the same ORM vs PRM distinction
    from the reward modeling section, applied to agent evaluation.
    """

    task_id: str
    agent_name: str
    prompt: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    reasoning_steps: List[str] = field(default_factory=list)  # scratchpad
    final_answer: str = ""
    error: Optional[str] = None
    total_tokens_used: int = 0

    @property
    def n_tool_calls(self) -> int:
        return len(self.tool_calls)

    @property
    def tool_names_used(self) -> List[str]:
        return [tc.tool_name for tc in self.tool_calls]

    def to_dict(self) -> Dict:
        return {
            "task_id": self.task_id,
            "agent_name": self.agent_name,
            "prompt": self.prompt,
            "tool_calls": [
                {
                    "tool": tc.tool_name,
                    "args": tc.arguments,
                    "result_preview": str(tc.result)[:200],
                    "step": tc.step,
                }
                for tc in self.tool_calls
            ],
            "reasoning_steps": self.reasoning_steps,
            "final_answer": self.final_answer,
            "error": self.error,
            "total_tokens_used": self.total_tokens_used,
            "n_tool_calls": self.n_tool_calls,
        }


# ── Eval result ───────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    task_id: str
    category: str
    agent_name: str
    answer_score: float         # 0.0–1.0 from task.scorer
    sequence_score: float       # 0.0–1.0 from task.sequence_scorer (or 1.0 if N/A)
    trajectory: AgentTrajectory
    ground_truth: Any
    predicted_answer: str

    @property
    def combined_score(self) -> float:
        """Equally-weighted average of answer and sequence scores."""
        return (self.answer_score + self.sequence_score) / 2


# ── Benchmark report ──────────────────────────────────────────────────────────

@dataclass
class BenchmarkReport:
    """Aggregate benchmark results across all tasks and agent configurations."""

    results: List[EvalResult]

    def accuracy(self, category: Optional[str] = None, agent: Optional[str] = None) -> float:
        """Mean answer score, optionally filtered by category and/or agent."""
        subset = self._filter(category, agent)
        if not subset:
            return 0.0
        return sum(r.answer_score for r in subset) / len(subset)

    def sequence_accuracy(self, category: Optional[str] = None, agent: Optional[str] = None) -> float:
        subset = self._filter(category, agent)
        if not subset:
            return 0.0
        return sum(r.sequence_score for r in subset) / len(subset)

    def avg_tool_calls(self, category: Optional[str] = None, agent: Optional[str] = None) -> float:
        """Average number of tool calls per task (efficiency metric)."""
        subset = self._filter(category, agent)
        if not subset:
            return 0.0
        return sum(r.trajectory.n_tool_calls for r in subset) / len(subset)

    def failure_recovery_rate(self, agent: Optional[str] = None) -> float:
        """For failure_recovery tasks: fraction handled gracefully (score > 0.5)."""
        subset = self._filter("failure_recovery", agent)
        if not subset:
            return 0.0
        return sum(1 for r in subset if r.answer_score > 0.5) / len(subset)

    def summary_table(self) -> str:
        """Markdown table: agents × (overall, per-category, efficiency, recovery)."""
        agents = sorted({r.agent_name for r in self.results})
        categories = ["tool_use", "multi_step", "failure_recovery"]

        header = f"| {'Agent':<25} | {'Overall':>8} | {'ToolUse':>8} | {'MultiStep':>10} | {'Recovery':>10} | {'Avg calls':>10} |"
        sep = "|" + "-" * 27 + "|" + ("-" * 10 + "|") * 5
        rows = [header, sep]

        for agent in agents:
            overall = self.accuracy(agent=agent)
            by_cat = [self.accuracy(cat, agent) for cat in categories]
            calls = self.avg_tool_calls(agent=agent)
            row = (
                f"| {agent:<25} | {overall:>8.3f} | "
                f"{by_cat[0]:>8.3f} | {by_cat[1]:>10.3f} | "
                f"{by_cat[2]:>10.3f} | {calls:>10.2f} |"
            )
            rows.append(row)

        return "\n".join(rows)

    def to_json(self, path: str) -> None:
        data = {
            "summary": {
                agent: {
                    "overall": self.accuracy(agent=agent),
                    "tool_use": self.accuracy("tool_use", agent),
                    "multi_step": self.accuracy("multi_step", agent),
                    "failure_recovery": self.failure_recovery_rate(agent),
                    "avg_tool_calls": self.avg_tool_calls(agent=agent),
                    "sequence_accuracy": self.sequence_accuracy(agent=agent),
                }
                for agent in {r.agent_name for r in self.results}
            },
            "per_task": [
                {
                    "task_id": r.task_id,
                    "category": r.category,
                    "agent": r.agent_name,
                    "answer_score": r.answer_score,
                    "sequence_score": r.sequence_score,
                    "n_tool_calls": r.trajectory.n_tool_calls,
                    "predicted": r.predicted_answer,
                    "ground_truth": str(r.ground_truth),
                }
                for r in self.results
            ],
        }
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _filter(self, category=None, agent=None) -> List[EvalResult]:
        r = self.results
        if category:
            r = [x for x in r if x.category == category]
        if agent:
            r = [x for x in r if x.agent_name == agent]
        return r
