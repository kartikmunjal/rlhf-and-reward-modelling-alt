"""Versioned arithmetic task, parsing, and shaped reward for the v2 pilot."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from random import Random


TAGGED_RE = re.compile(r"<answer>\s*([+-]?\d+)\s*</answer>", re.IGNORECASE)
INTEGER_RE = re.compile(r"(?<![\w.])[+-]?\d+(?![\w.])")


@dataclass(frozen=True)
class ArithmeticProblemV2:
    problem_id: int
    prompt: str
    answer: int
    intermediate: int
    operands: tuple[int, int, int, int]

    def to_dict(self) -> dict:
        value = asdict(self)
        value["operands"] = list(self.operands)
        return value


def generate_problems_v2(seed: int, count: int, *, id_offset: int = 0) -> list[ArithmeticProblemV2]:
    """Generate unambiguous two-operation word problems with positive integers."""
    rng = Random(seed)
    problems = []
    for index in range(count):
        start = rng.randint(10, 70)
        received = rng.randint(4, 40)
        sold = rng.randint(2, min(30, start + received - 1))
        stores = rng.randint(2, 6)
        remaining = start + received - sold
        answer = remaining * stores
        problem_id = id_offset + index
        prompt = (
            f"Problem ID: {problem_id}\n"
            f"One warehouse has {start} notebooks, receives {received}, then sells {sold}. "
            f"There are {stores} warehouses with exactly that final inventory. "
            "Compute the total inventory across all warehouses. Give one short equation, then finish "
            "with <answer>integer</answer>."
        )
        problems.append(
            ArithmeticProblemV2(problem_id, prompt, answer, remaining, (start, received, sold, stores))
        )
    return problems


def sft_completion(problem: ArithmeticProblemV2) -> str:
    start, received, sold, stores = problem.operands
    return (
        f"Remaining per warehouse: {start} + {received} - {sold} = {problem.intermediate}. "
        f"Total: {problem.intermediate} * {stores} = {problem.answer}. "
        f"<answer>{problem.answer}</answer>"
    )


def parse_tagged_answer(text: str) -> int | None:
    match = TAGGED_RE.search(text)
    return int(match.group(1)) if match else None


def parse_final_integer(text: str) -> int | None:
    tagged = parse_tagged_answer(text)
    if tagged is not None:
        return tagged
    matches = INTEGER_RE.findall(text)
    return int(matches[-1]) if matches else None


def score_completion_v2(text: str, problem: ArithmeticProblemV2, reward: dict) -> tuple[float, dict]:
    tagged = parse_tagged_answer(text)
    numeric = parse_final_integer(text)
    if tagged == problem.answer:
        score = float(reward["tagged_exact"])
        outcome = "tagged_exact"
    elif numeric == problem.answer:
        score = float(reward["numeric_exact_without_tag"])
        outcome = "numeric_exact_without_tag"
    elif tagged is not None:
        score = float(reward["valid_tag_wrong"])
        outcome = "valid_tag_wrong"
    else:
        score = float(reward["invalid_or_wrong"])
        outcome = "invalid_or_wrong"
    intermediate_present = bool(
        re.search(rf"(?<!\d){re.escape(str(problem.intermediate))}(?!\d)", text)
    )
    if intermediate_present and numeric != problem.answer:
        score += float(reward["correct_intermediate"])
    token_count = len(text.split())
    excess = max(0, token_count - int(reward["length_penalty_after_tokens"]))
    score += excess * float(reward["length_penalty_per_token"])
    return score, {
        "tagged_answer": tagged,
        "numeric_answer": numeric,
        "numeric_exact": numeric == problem.answer,
        "tagged_exact": tagged == problem.answer,
        "intermediate_present": intermediate_present,
        "outcome": outcome,
        "whitespace_tokens": token_count,
    }
