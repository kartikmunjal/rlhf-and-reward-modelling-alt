"""Strict schemas and validation for pointwise and pairwise judge outputs."""

from __future__ import annotations

from llm_judge_summeval.data import AXES


def pointwise_schema() -> dict:
    item = {
        "type": "object",
        # Anthropic structured outputs do not support numeric range keywords;
        # validate the preregistered 1--5 range locally in validate_output.
        "properties": {"score": {"type": "integer"}, "rationale": {"type": "string"}},
        "required": ["score", "rationale"], "additionalProperties": False,
    }
    return {"type": "object", "properties": {axis: item for axis in AXES}, "required": list(AXES), "additionalProperties": False}


def pairwise_schema() -> dict:
    item = {
        "type": "object",
        "properties": {"winner": {"type": "string", "enum": ["A", "B", "tie"]}, "rationale": {"type": "string"}},
        "required": ["winner", "rationale"], "additionalProperties": False,
    }
    return {"type": "object", "properties": {axis: item for axis in AXES}, "required": list(AXES), "additionalProperties": False}


def validate_output(value: object, kind: str) -> dict:
    if not isinstance(value, dict) or set(value) != set(AXES):
        raise ValueError("Output must contain exactly the four registered axes")
    for axis in AXES:
        item = value[axis]
        if not isinstance(item, dict) or not isinstance(item.get("rationale"), str) or not item["rationale"].strip():
            raise ValueError(f"Invalid rationale for {axis}")
        if kind == "pointwise":
            score = item.get("score")
            if isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 5:
                raise ValueError(f"Invalid pointwise score for {axis}")
        elif kind == "pairwise":
            if item.get("winner") not in {"A", "B", "tie"}:
                raise ValueError(f"Invalid pairwise winner for {axis}")
        else:
            raise ValueError(f"Unknown judge kind: {kind}")
    return value
