"""Harder, unambiguous arithmetic distribution for the separate v2b pilot."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from random import Random

from src.evaluation.arithmetic_rl_v2 import parse_final_integer, parse_tagged_answer


@dataclass(frozen=True)
class ArithmeticProblemV2B:
    problem_id: int
    prompt: str
    answer: int
    incoming: int
    remaining: int
    operands: tuple[int, int, int, int, int]

    def to_dict(self) -> dict:
        value = asdict(self)
        value["operands"] = list(self.operands)
        return value


def generate_problems_v2b(seed: int, count: int, *, id_offset: int = 0) -> list[ArithmeticProblemV2B]:
    rng = Random(seed)
    problems = []
    for index in range(count):
        initial = rng.randint(80, 300)
        crates = rng.randint(4, 15)
        per_crate = rng.randint(7, 25)
        damaged = rng.randint(5, 60)
        branches = rng.randint(2, 7)
        incoming = crates * per_crate
        remaining = initial + incoming - damaged
        answer = remaining * branches
        problem_id = id_offset + index
        prompt = (
            f"Problem ID: {problem_id}\n"
            f"A warehouse begins with {initial} devices. It receives {crates} crates containing "
            f"{per_crate} devices each, then discards {damaged} damaged devices. There are {branches} "
            "warehouses with exactly that final inventory. Compute the total inventory across all warehouses. "
            "Give short equations and finish with <answer>integer</answer>."
        )
        problems.append(
            ArithmeticProblemV2B(
                problem_id, prompt, answer, incoming, remaining, (initial, crates, per_crate, damaged, branches)
            )
        )
    return problems


def score_completion_v2b(text: str, problem: ArithmeticProblemV2B, reward: dict) -> tuple[float, dict]:
    tagged = parse_tagged_answer(text)
    numeric = parse_final_integer(text)
    if tagged == problem.answer:
        score, outcome = float(reward["tagged_exact"]), "tagged_exact"
    elif numeric == problem.answer:
        score, outcome = float(reward["numeric_exact_without_tag"]), "numeric_exact_without_tag"
    elif tagged is not None:
        score, outcome = float(reward["valid_tag_wrong"]), "valid_tag_wrong"
    else:
        score, outcome = float(reward["invalid_or_wrong"]), "invalid_or_wrong"
    intermediate_hits = sum(
        bool(re.search(rf"(?<!\d){value}(?!\d)", text)) for value in (problem.incoming, problem.remaining)
    )
    if numeric != problem.answer:
        score += intermediate_hits * float(reward["correct_intermediate_each"])
    token_count = len(text.split())
    excess = max(0, token_count - int(reward["length_penalty_after_tokens"]))
    score += excess * float(reward["length_penalty_per_token"])
    return score, {
        "tagged_answer": tagged,
        "numeric_answer": numeric,
        "numeric_exact": numeric == problem.answer,
        "tagged_exact": tagged == problem.answer,
        "intermediate_hits": intermediate_hits,
        "outcome": outcome,
        "whitespace_tokens": token_count,
    }
