"""Pinned HumanEval loader with independently generated oracle-labeled cases."""

from __future__ import annotations

import gzip
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Callable

from .local_eval import evaluate_trusted
from .schema import CodeTask, TestCase, canonical_json


ARCHIVE_SHA256 = "b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef"
REVISION = "6d43fb980f9fee3c892a914eda09951f772ad10d"
SELECTION_SEED = 20260928


def _strings(rng: random.Random, alphabet: str = "abcXYZ09", maximum: int = 12):
    return "".join(rng.choice(alphabet) for _ in range(rng.randrange(maximum + 1)))


def _cases(entry: str, rng: random.Random) -> list[list[Any]]:
    planets = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"]
    number_words = "zero one two three four five six seven eight nine".split()
    if entry == "match_parens":
        return [[["", ""]], [["()", ""]], [["(", ")"]]] + [[["".join(rng.choice("()") for _ in range(rng.randrange(9))), "".join(rng.choice("()") for _ in range(rng.randrange(9)))] ] for _ in range(80)]
    if entry == "select_words":
        words = ["alpha", "sky", "queue", "rhythm", "code", "model", "test"]
        return [["", 0], ["sky", 3], ["alpha sky", 3]] + [[" ".join(rng.choice(words) for _ in range(rng.randrange(1, 8))), rng.randrange(0, 7)] for _ in range(80)]
    if entry == "choose_num":
        return [[1, 1], [1, 2], [3, 8]] + [[rng.randrange(1, 60), rng.randrange(1, 60)] for _ in range(80)]
    if entry == "bf":
        return [["Mercury", "Venus"], ["Earth", "Mars"], ["Pluto", "Earth"]] + [[rng.choice(planets), rng.choice(planets)] for _ in range(80)]
    if entry in {"decode_shift", "decode_cyclic", "encrypt"}:
        return [[""], ["a"], ["abc"]] + [[_strings(rng, "abcdefghijklmnopqrstuvwxyz", 25)] for _ in range(80)]
    if entry == "sort_numbers":
        return [[""], ["one"], ["three one"]] + [[" ".join(rng.choice(number_words) for _ in range(rng.randrange(0, 12)))] for _ in range(80)]
    if entry == "unique_digits":
        return [[[]], [[1]], [[15, 22]]] + [[[rng.randrange(1, 10000) for _ in range(rng.randrange(0, 12))]] for _ in range(80)]
    if entry in {"get_positive", "incr_list", "rolling_max", "sum_squares"}:
        return [[[]], [[0]], [[-1, 2, 3]]] + [[[rng.randrange(-20, 21) for _ in range(rng.randrange(0, 15))]] for _ in range(80)]
    if entry == "check_dict_case":
        keys = ["a", "b", "A", "B", "Mixed"]
        return [[{}], [{"a": 1}], [{"A": 1}]] + [[{rng.choice(keys): rng.randrange(5) for _ in range(rng.randrange(1, 6))}] for _ in range(80)]
    if entry in {"triples_sum_to_zero", "next_smallest", "can_arrange"}:
        rows = [[[]], [[1]], [[1, -1, 0]]]
        for _ in range(80):
            values = [rng.randrange(-12, 13) for _ in range(rng.randrange(0, 12))]
            if entry == "can_arrange":
                values = list(dict.fromkeys(values))
            rows.append([values])
        return rows
    if entry == "search":
        return [[[1]], [[2]], [[1, 1, 2]]] + [[[rng.randrange(1, 9) for _ in range(rng.randrange(1, 18))]] for _ in range(80)]
    if entry in {"fizz_buzz", "fib4", "get_max_triples"}:
        return [[0], [1], [5]] + [[rng.randrange(0 if entry != "get_max_triples" else 1, 80)] for _ in range(80)]
    if entry == "same_chars":
        return [["", ""], ["a", "a"], ["a", "b"]] + [[_strings(rng, "abcde", 15), _strings(rng, "abcde", 15)] for _ in range(80)]
    if entry == "double_the_difference":
        return [[[]], [[1]], [[-1, 3, 4]]] + [[[rng.choice([rng.randrange(-9, 10), rng.random() * 5]) for _ in range(rng.randrange(0, 12))]] for _ in range(80)]
    if entry == "encode":
        return [[""], ["a"], ["Test"]] + [[_strings(rng, "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ", 25)] for _ in range(80)]
    if entry == "concatenate":
        return [[[]], [[""]], [["a", "b"]]] + [[[ _strings(rng, "abc", 6) for _ in range(rng.randrange(0, 10))]] for _ in range(80)]
    if entry == "is_sorted":
        return [[[]], [[1]], [[1, 2, 2]]] + [[[rng.randrange(0, 10) for _ in range(rng.randrange(0, 12))]] for _ in range(80)]
    if entry == "longest":
        return [[[]], [[""]], [["a", "bb"]]] + [[[_strings(rng, "abc", 10) for _ in range(rng.randrange(0, 10))]] for _ in range(80)]
    if entry == "add":
        return [[[1]], [[2, 4]], [[4, 2, 6, 7]]] + [[[rng.randrange(-20, 21) for _ in range(rng.randrange(1, 15))]] for _ in range(80)]
    if entry == "simplify":
        fractions = [f"{a}/{b}" for a in range(1, 11) for b in range(1, 8)]
        return [["1/1", "1/1"], ["1/2", "2/1"], ["1/3", "2/1"]] + [[rng.choice(fractions), rng.choice(fractions)] for _ in range(80)]
    if entry == "count_upper":
        return [[""], ["A"], ["aA"]] + [[_strings(rng, "aeiouAEIOUbcDF", 25)] for _ in range(80)]
    if entry == "greatest_common_divisor":
        return [[1, 1], [2, 3], [12, 18]] + [[rng.randrange(1, 500), rng.randrange(1, 500)] for _ in range(80)]
    if entry == "has_close_elements":
        rows = [[[1.0, 2.0], 0.5], [[1.0, 1.1], 0.2], [[], 0.5]]
        for _ in range(80):
            rows.append([[round(rng.uniform(-10, 10), 2) for _ in range(rng.randrange(0, 10))], round(rng.uniform(0.01, 3), 2)])
        return rows
    if entry == "do_algebra":
        ops = ["+", "-", "*", "//", "**"]
        rows = [[['+'], [1, 2]], [['*'], [2, 3]], [['//'], [5, 2]]]
        while len(rows) < 90:
            count = rng.randrange(1, 5)
            chosen = [rng.choice(ops) for _ in range(count)]
            operands = [rng.randrange(1, 5) for _ in range(count + 1)]
            if chosen.count("**") <= 1:
                rows.append([chosen, operands])
        return rows
    if entry == "by_length":
        return [[[]], [[1]], [[1, 10, -1]]] + [[[rng.randrange(-5, 16) for _ in range(rng.randrange(0, 15))]] for _ in range(80)]
    if entry == "rescale_to_unit":
        rows = [[[0.0, 1.0]], [[-1.0, 1.0]], [[1.0, 2.0, 3.0]]]
        while len(rows) < 90:
            values = sorted({round(rng.uniform(-20, 20), 2) for _ in range(rng.randrange(2, 10))})
            if len(values) >= 2:
                rows.append([values])
        return rows
    if entry == "parse_music":
        notes = ["o", "o|", ".|"]
        return [["o"], ["o|"], [".| o"]] + [[" ".join(rng.choice(notes) for _ in range(rng.randrange(1, 20)))] for _ in range(80)]
    if entry == "count_nums":
        return [[[]], [[0]], [[-1, 11, -11]]] + [[[rng.randrange(-9999, 10000) for _ in range(rng.randrange(0, 15))]] for _ in range(80)]
    if entry == "sorted_list_sum":
        return [[[]], [["a"]], [["aa", "b"]]] + [[[_strings(rng, "abc", 9) for _ in range(rng.randrange(0, 12))]] for _ in range(80)]
    if entry == "truncate_number":
        return [[1.0], [1.5], [3.25]] + [[round(rng.uniform(0.01, 100), 4)] for _ in range(80)]
    if entry == "change_base":
        return [[1, 2], [8, 2], [8, 3]] + [[rng.randrange(1, 10000), rng.randrange(2, 10)] for _ in range(80)]
    raise KeyError(f"No independent-case generator for {entry}")


def _unique_cases(arguments: list[list[Any]]) -> list[list[Any]]:
    output, seen = [], set()
    for args in arguments:
        key = canonical_json(args)
        if key not in seen:
            seen.add(key)
            output.append(args)
    if len(output) < 43:
        raise ValueError(f"Only {len(output)} independent inputs")
    return output[:43]


def load_selected_tasks(archive: str | Path) -> list[CodeTask]:
    archive = Path(archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("HumanEval archive SHA-256 mismatch")
    with gzip.open(archive, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    selected = sorted(
        rows,
        key=lambda row: hashlib.sha256(f"{SELECTION_SEED}:{row['task_id']}".encode()).hexdigest(),
    )[:40]
    tasks = []
    for row in selected:
        rng = random.Random(f"{SELECTION_SEED}:{row['task_id']}:independent")
        args_rows = _unique_cases(_cases(row["entry_point"], rng))
        source = row["prompt"] + row["canonical_solution"]
        provisional = tuple(TestCase(args=args, partition=f"domain_{index % 4}") for index, args in enumerate(args_rows))
        evaluated = evaluate_trusted(source, row["entry_point"], provisional)
        if any(item["error"] for item in evaluated):
            raise RuntimeError(f"Canonical solution failed generated domain: {row['task_id']}")
        cases = tuple(
            TestCase(args=case.args, expected=item["actual"], partition=case.partition)
            for case, item in zip(provisional, evaluated)
        )
        tasks.append(CodeTask(
            task_id=row["task_id"], source="openai_humaneval_mit",
            family=f"humaneval_{row['task_id'].split('/')[-1]}", prompt=row["prompt"],
            entry_point=row["entry_point"], reference_solution=source,
            visible_tests=cases[:3], hidden_tests=cases[3:23], property_tests=cases[23:43],
            metadata={"upstream_revision": REVISION, "archive_sha256": ARCHIVE_SHA256},
        ))
    return tasks
