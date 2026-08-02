"""Deterministic arithmetic tasks and metrics for the PPO/GRPO study."""

from __future__ import annotations

import re
from dataclasses import dataclass
from random import Random


ANSWER_RE = re.compile(r"<answer>\s*([+-]?\d+)\s*</answer>", re.IGNORECASE)
ID_RE = re.compile(r"Problem ID:\s*(\d+)")


@dataclass(frozen=True)
class ArithmeticProblem:
    problem_id: int
    prompt: str
    answer: int


def generate_problems(seed: int, count: int, *, id_offset: int = 0) -> list[ArithmeticProblem]:
    """Generate version-1, multi-step integer word problems."""
    rng = Random(seed)
    problems: list[ArithmeticProblem] = []
    for index in range(count):
        start = rng.randint(8, 60)
        added = rng.randint(3, 35)
        removed = rng.randint(2, min(25, start + added - 1))
        bundles = rng.randint(2, 6)
        answer = (start + added - removed) * bundles
        problem_id = id_offset + index
        prompt = (
            f"Problem ID: {problem_id}\n"
            f"A store starts with {start} notebooks, receives {added} more, and sells {removed}. "
            f"It then prepares {bundles} identical shipments, each containing that remaining number. "
            "How many notebooks are shipped in total? Show brief reasoning and finish with exactly "
            "<answer>integer</answer>."
        )
        problems.append(ArithmeticProblem(problem_id, prompt, answer))
    return problems


def extract_answer(text: str) -> int | None:
    match = ANSWER_RE.search(text)
    return int(match.group(1)) if match else None


def score_completion(text: str, answer: int, config: dict) -> float:
    parsed = extract_answer(text)
    if parsed == answer:
        score = float(config["exact_answer"])
    elif parsed is not None:
        score = float(config["valid_answer_tag"])
    else:
        score = float(config["invalid_format"])
    # Tokenization-independent whitespace count keeps the verifier auditable.
    extra = max(0, len(text.split()) - 48)
    return score + extra * float(config["length_penalty_per_token_after_48"])


def answer_map(problems: list[ArithmeticProblem]) -> dict[int, int]:
    return {problem.problem_id: problem.answer for problem in problems}
