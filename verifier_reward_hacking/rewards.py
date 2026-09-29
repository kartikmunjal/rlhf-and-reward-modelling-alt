"""Frozen verifier definitions and sandboxed reward computation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from .evaluation import evaluate_completion
from .sandbox import SandboxConfig
from .schema import CodeTask, TestCase


def v2_cases(task: CodeTask, audit: dict) -> tuple[TestCase, ...]:
    selected = set(audit["selected_v2_input_ids"])
    additions = tuple(case for case in task.hidden_tests if case.input_id in selected)
    if len(additions) != len(selected):
        raise RuntimeError(f"{task.task_id}: V2 case IDs do not match task")
    return task.visible_tests + additions


def verifier_cases(task: CodeTask, condition: str, audit: dict) -> tuple[TestCase, ...]:
    if condition == "C1":
        return task.visible_tests
    if condition == "C2":
        return v2_cases(task, audit)
    if condition == "C3":
        return task.property_tests
    if condition == "C4":
        return task.visible_tests + task.property_tests
    raise ValueError(f"No training reward for condition {condition}")


def completion_text(value) -> str:  # TRL versions may return text or chat messages
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value and isinstance(value[-1], dict):
        return str(value[-1].get("content", ""))
    return str(value)


class SandboxedVerifierReward:
    def __init__(self, tasks: dict[str, CodeTask], audits: dict[str, dict], condition: str, sandbox: SandboxConfig, workers: int = 4):
        self.tasks = tasks
        self.audits = audits
        self.condition = condition
        self.sandbox = sandbox
        self.workers = workers

    def _score(self, completion, task_id: str) -> float:
        task = self.tasks[str(task_id)]
        cases = verifier_cases(task, self.condition, self.audits[task.task_id])
        result = evaluate_completion(completion_text(completion), task, cases, self.sandbox)
        return float(result["passed"])

    def __call__(self, completions, task_id, **kwargs):
        del kwargs
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return list(pool.map(self._score, completions, [str(value) for value in task_id]))
