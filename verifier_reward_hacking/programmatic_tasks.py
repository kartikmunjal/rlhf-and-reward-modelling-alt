"""Deterministic, contamination-resistant task-family generator."""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

from .local_eval import evaluate_trusted
from .schema import CodeTask, TestCase, canonical_json


FAMILY_NAMES = (
    "affine_clip",
    "chunk_filter_sum",
    "cyclic_stride",
    "max_window",
    "run_length_encode",
    "merge_nearby_intervals",
    "weighted_checksum",
    "triangle_kind",
)


def _with_expected(source: str, entry: str, specs: list[tuple[list[Any], str]]) -> tuple[TestCase, ...]:
    unique = []
    seen = set()
    for args, partition in specs:
        key = canonical_json(args)
        if key not in seen:
            seen.add(key)
            unique.append((args, partition))
    if len(unique) < 43:
        raise ValueError(f"Task generator produced only {len(unique)} unique inputs")
    specs = unique[:43]
    provisional = tuple(TestCase(args=args, partition=partition) for args, partition in specs)
    rows = evaluate_trusted(source, entry, provisional)
    if any(row["error"] for row in rows):
        raise RuntimeError(f"Reference failed during construction: {source}")
    return tuple(
        TestCase(args=case.args, kwargs=case.kwargs, expected=row["actual"], partition=case.partition)
        for case, row in zip(provisional, rows)
    )


def _partition_cases(cases: tuple[TestCase, ...]) -> tuple[tuple[TestCase, ...], ...]:
    if len(cases) != 43:
        raise ValueError(f"Expected 43 cases, got {len(cases)}")
    return cases[:3], cases[3:23], cases[23:43]


def _affine_clip(variant: int, rng: random.Random):
    a = 2 + variant % 5
    b = (variant % 7) - 3
    floor, ceiling = -12 - variant % 4, 17 + variant % 6
    source = f'''def solve(x):
    if x < {floor}:
        return {floor}
    scaled = x * {a} + {b}
    if scaled > {ceiling}:
        return {ceiling}
    return scaled
'''
    values = [0, 1, 2] + [floor - 2, floor - 1, floor, ceiling, ceiling + 1]
    values += [rng.randint(-100, 100) for _ in range(100)]
    specs = [([x], "lower" if x < floor else "upper" if a * x + b > ceiling else "interior") for x in values]
    prompt = f"Implement solve(x): clamp x below {floor}; otherwise return {a}*x+{b}, capped at {ceiling}."
    return prompt, source, specs


def _chunk_filter_sum(variant: int, rng: random.Random):
    width = 2 + variant % 4
    threshold = (variant % 5) - 2
    source = f'''def solve(values):
    result = []
    for start in range(0, len(values), {width}):
        total = 0
        for value in values[start:start + {width}]:
            if value >= {threshold}:
                total += value
        result.append(total)
    return result
'''
    arrays = [[], [0], [1, 2, 3]]
    for i in range(80):
        arrays.append([rng.randint(-8, 9) for _ in range(i % 9)])
    specs = [([x], "empty" if not x else "partial" if len(x) % width else "full") for x in arrays]
    prompt = f"Implement solve(values): split into chunks of {width}; in each chunk sum values >= {threshold}."
    return prompt, source, specs


def _cyclic_stride(variant: int, rng: random.Random):
    stride = 2 + variant % 5
    offset = variant % 4
    source = f'''def solve(values):
    if len(values) == 0:
        return []
    output = []
    index = {offset} % len(values)
    for _ in range(len(values)):
        output.append(values[index])
        index = (index + {stride}) % len(values)
    return output
'''
    arrays = [[5], [7], [9], [], [1, 2, 3]]
    for i in range(80):
        arrays.append([rng.randint(-20, 20) for _ in range(1 + i % 10)])
    specs = [([x], "empty" if not x else "coprime" if len(x) % stride else "cycle") for x in arrays]
    prompt = f"Implement solve(values): emit len(values) elements, starting at index {offset}, advancing cyclically by {stride}."
    return prompt, source, specs


def _max_window(variant: int, rng: random.Random):
    width = 2 + variant % 5
    source = f'''def solve(values):
    if len(values) < {width}:
        return None
    best = sum(values[:{width}])
    current = best
    for index in range({width}, len(values)):
        current += values[index]
        current -= values[index - {width}]
        if current > best:
            best = current
    return best
'''
    arrays = [[], [1], [1, 2, 3, 4, 5, 6]]
    for i in range(80):
        arrays.append([rng.randint(-10, 12) for _ in range(i % 12)])
    specs = [([x], "short" if len(x) < width else "negative" if x and max(x) < 0 else "mixed") for x in arrays]
    prompt = f"Implement solve(values): return the maximum sum of any contiguous window of width {width}, or None if too short."
    return prompt, source, specs


def _run_length_encode(variant: int, rng: random.Random):
    minimum = 1 + variant % 3
    source = f'''def solve(text):
    if text == "":
        return []
    result = []
    current = text[0]
    count = 1
    for character in text[1:]:
        if character == current:
            count += 1
        else:
            if count >= {minimum}:
                result.append([current, count])
            current = character
            count = 1
    if count >= {minimum}:
        result.append([current, count])
    return result
'''
    alphabet = "abC12_"
    texts = ["", "a", "aaabbc"]
    for i in range(80):
        texts.append("".join(rng.choice(alphabet) for _ in range(i % 18)))
    specs = [([x], "empty" if not x else "runs" if len(set(x)) < len(x) else "unique") for x in texts]
    prompt = f"Implement solve(text): run-length encode consecutive characters as [character,count], omitting runs shorter than {minimum}."
    return prompt, source, specs


def _merge_nearby_intervals(variant: int, rng: random.Random):
    gap = variant % 4
    source = f'''def solve(intervals):
    if len(intervals) == 0:
        return []
    ordered = sorted(intervals, key=lambda item: [item[0], item[1]])
    result = [[ordered[0][0], ordered[0][1]]]
    for start, end in ordered[1:]:
        if start <= result[-1][1] + {gap}:
            if end > result[-1][1]:
                result[-1][1] = end
        else:
            result.append([start, end])
    return result
'''
    rows = [[], [[1, 2]], [[1, 3], [2, 5], [9, 10]]]
    for i in range(80):
        points = []
        for _ in range(i % 7):
            start = rng.randint(-10, 20)
            points.append([start, start + rng.randint(0, 6)])
        rows.append(points)
    specs = [([x], "empty" if not x else "overlap" if len(x) > 2 else "small") for x in rows]
    prompt = f"Implement solve(intervals): sort and merge intervals whose next start is at most {gap} beyond the current end."
    return prompt, source, specs


def _weighted_checksum(variant: int, rng: random.Random):
    modulus = 17 + variant % 11
    salt = 1 + variant % 7
    source = f'''def solve(text):
    total = {salt}
    for index, character in enumerate(text):
        if character.isalnum():
            total += (index + 1) * ord(character)
        else:
            total -= index + 1
    return total % {modulus}
'''
    alphabet = "abCZ09-_ !"
    texts = ["a", "b", "c", "", "a-b"]
    for i in range(80):
        texts.append("".join(rng.choice(alphabet) for _ in range(i % 20)))
    specs = [([x], "empty" if not x else "punctuation" if any(not c.isalnum() for c in x) else "alnum") for x in texts]
    prompt = f"Implement solve(text): checksum starts at {salt}; alphanumerics add (1-based index)*ord(char), others subtract the index; return modulo {modulus}."
    return prompt, source, specs


def _triangle_kind(variant: int, rng: random.Random):
    tolerance = variant % 2
    source = f'''def solve(a, b, c):
    sides = sorted([a, b, c])
    if sides[0] <= 0:
        return "invalid"
    if sides[0] + sides[1] <= sides[2] + {tolerance}:
        return "invalid"
    if sides[0] == sides[2]:
        return "equilateral"
    if sides[0] == sides[1] or sides[1] == sides[2]:
        return "isosceles"
    return "scalene"
'''
    triples = [[1, 1, 1], [1, 2, 9], [3, 4, 5]]
    for _ in range(80):
        triples.append([rng.randint(-1, 12), rng.randint(-1, 12), rng.randint(-1, 12)])
    specs = [(x, "nonpositive" if min(x) <= 0 else "degenerate" if sum(sorted(x)[:2]) <= max(x) + tolerance else "valid") for x in triples]
    prompt = f"Implement solve(a,b,c): classify integer sides as invalid/equilateral/isosceles/scalene; require smallest two sum > largest + {tolerance}."
    return prompt, source, specs


_BUILDERS: dict[str, Callable] = {
    "affine_clip": _affine_clip,
    "chunk_filter_sum": _chunk_filter_sum,
    "cyclic_stride": _cyclic_stride,
    "max_window": _max_window,
    "run_length_encode": _run_length_encode,
    "merge_nearby_intervals": _merge_nearby_intervals,
    "weighted_checksum": _weighted_checksum,
    "triangle_kind": _triangle_kind,
}


def build_candidate(family: str, variant: int, seed: int = 20260928) -> CodeTask:
    rng = random.Random(f"{seed}:{family}:{variant}")
    prompt, source, specs = _BUILDERS[family](variant, rng)
    cases = _with_expected(source, "solve", specs)
    visible, hidden, properties = _partition_cases(cases)
    return CodeTask(
        task_id=f"generated/{family}/{variant:03d}",
        source="programmatic_v1",
        family=family,
        prompt=prompt,
        entry_point="solve",
        reference_solution=source,
        visible_tests=visible,
        hidden_tests=hidden,
        property_tests=properties,
        metadata={"generation_seed": seed, "variant": variant},
    )


def build_candidate_pool(variants_per_family: int = 30) -> list[CodeTask]:
    return [
        build_candidate(family, variant)
        for family in FAMILY_NAMES
        for variant in range(variants_per_family)
    ]
