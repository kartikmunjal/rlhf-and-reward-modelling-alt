"""Typed task and verifier records with canonical serialization."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True)
class TestCase:
    args: list[Any] = field(default_factory=list)
    kwargs: dict[str, Any] = field(default_factory=dict)
    expected: Any = None
    partition: str = "default"

    @property
    def input_id(self) -> str:
        payload = {"args": self.args, "kwargs": self.kwargs}
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CodeTask:
    task_id: str
    source: str
    family: str
    prompt: str
    entry_point: str
    reference_solution: str
    visible_tests: tuple[TestCase, ...]
    hidden_tests: tuple[TestCase, ...]
    property_tests: tuple[TestCase, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.visible_tests) != 3:
            raise ValueError(f"{self.task_id}: V0 must contain exactly three tests")
        if len(self.hidden_tests) < 20:
            raise ValueError(f"{self.task_id}: V1 must contain at least 20 tests")
        if len(self.property_tests) != 20:
            raise ValueError(f"{self.task_id}: V3 must contain exactly 20 tests")
        all_hidden = [case.input_id for case in self.hidden_tests]
        if len(all_hidden) != len(set(all_hidden)):
            raise ValueError(f"{self.task_id}: duplicate V1 inputs")

    @property
    def task_hash(self) -> str:
        return hashlib.sha256(canonical_json(self.to_dict()).encode("utf-8")).hexdigest()

    def to_dict(self, include_reference: bool = True) -> dict[str, Any]:
        payload = asdict(self)
        if not include_reference:
            payload.pop("reference_solution")
            for key in ("hidden_tests", "property_tests"):
                payload.pop(key)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CodeTask":
        row = dict(payload)
        for key in ("visible_tests", "hidden_tests", "property_tests"):
            row[key] = tuple(TestCase(**case) for case in row[key])
        return cls(**row)

