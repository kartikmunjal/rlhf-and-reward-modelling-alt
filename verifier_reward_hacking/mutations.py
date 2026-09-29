"""Deterministic first-order AST mutations for verifier-strength audits."""

from __future__ import annotations

import ast
import copy
import hashlib
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class Mutant:
    mutant_id: str
    operator: str
    source: str


_BINOP_REPLACEMENTS = {
    ast.Add: ast.Sub,
    ast.Sub: ast.Add,
    ast.Mult: ast.FloorDiv,
    ast.FloorDiv: ast.Mult,
    ast.Mod: ast.FloorDiv,
}
_COMPARE_REPLACEMENTS = {
    ast.Lt: ast.LtE,
    ast.LtE: ast.Lt,
    ast.Gt: ast.GtE,
    ast.GtE: ast.Gt,
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
}


def _locations(tree: ast.AST) -> Iterator[tuple[str, int]]:
    for index, node in enumerate(ast.walk(tree)):
        if type(node) in _BINOP_REPLACEMENTS:
            yield "binop", index
        elif type(node) in _COMPARE_REPLACEMENTS:
            yield "compare", index
        elif isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
            yield "constant_plus_one", index
            yield "constant_minus_one", index
        elif isinstance(node, ast.BoolOp):
            yield "boolop", index


def _mutate_at(tree: ast.AST, operator: str, target_index: int) -> ast.AST:
    mutated = copy.deepcopy(tree)
    nodes = list(ast.walk(mutated))
    node = nodes[target_index]
    if operator == "binop":
        node.__class__ = _BINOP_REPLACEMENTS[type(node)]
    elif operator == "compare":
        node.__class__ = _COMPARE_REPLACEMENTS[type(node)]
    elif operator == "constant_plus_one":
        node.value += 1
    elif operator == "constant_minus_one":
        node.value -= 1
    elif operator == "boolop":
        node.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
    else:
        raise ValueError(operator)
    ast.fix_missing_locations(mutated)
    return mutated


def generate_mutants(source: str, maximum: int = 64) -> list[Mutant]:
    tree = ast.parse(source)
    seen = {source.strip()}
    mutants: list[Mutant] = []
    for operator, index in _locations(tree):
        try:
            candidate = ast.unparse(_mutate_at(tree, operator, index)).strip() + "\n"
            compile(candidate, "<mutant>", "exec")
        except (SyntaxError, ValueError, ZeroDivisionError):
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        digest = hashlib.sha256(candidate.encode("utf-8")).hexdigest()[:16]
        mutants.append(Mutant(f"{operator}:{index}:{digest}", operator, candidate))
        if len(mutants) >= maximum:
            break
    return mutants

