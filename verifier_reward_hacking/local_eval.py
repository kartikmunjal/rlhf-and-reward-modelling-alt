"""Trusted local evaluation for references and AST mutants only.

Never call this module on model-generated code. Untrusted code must go through
the container boundary in :mod:`verifier_reward_hacking.sandbox`.
"""

from __future__ import annotations

import json
import signal
from contextlib import contextmanager
from typing import Any, Iterable

from .schema import TestCase


@contextmanager
def _time_limit(seconds: float | None):
    if seconds is None or not hasattr(signal, "setitimer"):
        yield
        return
    def expired(_signum, _frame):
        raise TimeoutError("trusted mutant execution timed out")
    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def evaluate_trusted(
    source: str,
    entry_point: str,
    cases: Iterable[TestCase],
    timeout_seconds: float | None = None,
) -> list[dict[str, Any]]:
    namespace: dict[str, Any] = {"__builtins__": __builtins__}
    exec(compile(source, "<trusted-task-source>", "exec"), namespace, namespace)
    function = namespace[entry_point]
    rows = []
    for case in cases:
        try:
            with _time_limit(timeout_seconds):
                value = function(*case.args, **case.kwargs)
            actual = json.loads(json.dumps(value, allow_nan=False))
            expected = json.loads(json.dumps(case.expected, allow_nan=False))
            passed = actual == expected and type(actual) is type(expected)
            rows.append({"passed": passed, "actual": actual, "error": None})
        except Exception as exc:  # trusted mutations are expected to fail
            rows.append({"passed": False, "actual": None, "error": type(exc).__name__})
    return rows


def pass_all(source: str, entry_point: str, cases: Iterable[TestCase]) -> bool:
    rows = evaluate_trusted(source, entry_point, cases)
    return bool(rows) and all(row["passed"] for row in rows)
