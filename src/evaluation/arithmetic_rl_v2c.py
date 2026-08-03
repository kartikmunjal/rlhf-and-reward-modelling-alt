"""Intermediate-difficulty operand ranges for the separate v2c pilot."""

from __future__ import annotations

from random import Random

from src.evaluation.arithmetic_rl_v2b import ArithmeticProblemV2B, score_completion_v2b


def generate_problems_v2c(seed: int, count: int, *, id_offset: int = 0) -> list[ArithmeticProblemV2B]:
    rng = Random(seed)
    problems = []
    for index in range(count):
        initial = rng.randint(40, 150)
        crates = rng.randint(3, 10)
        per_crate = rng.randint(5, 15)
        damaged = rng.randint(3, 30)
        branches = rng.randint(2, 5)
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


score_completion_v2c = score_completion_v2b
