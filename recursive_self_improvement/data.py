"""Deterministic prompt-level partitions with fail-closed leakage checks."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable


NORMALIZATION_VERSION = "nfkc_whitespace_lower_v1"


def normalize_prompt(text: str) -> str:
    import unicodedata
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def prompt_id(prompt: str) -> str:
    return hashlib.sha256(normalize_prompt(prompt).encode("utf-8")).hexdigest()


def stable_rank(seed: int, prompt: str) -> str:
    return hashlib.sha256(f"{seed}:{prompt_id(prompt)}".encode()).hexdigest()


def partition_pairs(rows: Iterable[dict], config: dict) -> dict[str, list[dict]]:
    """Deduplicate by normalized prompt and allocate immutable disjoint pools."""
    unique = {}
    for row in rows:
        prompt = row["prompt"]
        key = prompt_id(prompt)
        candidate = {**row, "prompt_id": key}
        if key in unique:
            if (unique[key].get("chosen"), unique[key].get("rejected")) != (
                candidate.get("chosen"), candidate.get("rejected")
            ):
                raise ValueError(f"Conflicting duplicate prompt: {key}")
            continue
        unique[key] = candidate
    ordered = sorted(unique.values(), key=lambda row: stable_rank(config["seed"], row["prompt"]))
    counts = config["data"]
    names = (
        ("sft_train", counts["sft_train_pairs"]),
        ("reward_train", counts["reward_train_pairs"]),
        ("reward_validation", counts["reward_validation_pairs"]),
        ("improvement", counts["improvement_prompts"]),
        ("independent_eval", counts["independent_eval_prompts"]),
    )
    required = sum(count for _, count in names)
    if len(ordered) < required:
        raise ValueError(f"Need {required} unique prompts, found {len(ordered)}")
    output, cursor = {}, 0
    for name, count in names:
        output[name] = ordered[cursor:cursor + count]
        cursor += count
    assert_disjoint(output)
    return output


def partition_allocations(rows: Iterable[dict], *, seed: int, allocations: list[tuple[str, int]]) -> dict[str, list[dict]]:
    """General split-aware allocator used by the feasibility amendment."""
    unique = {}
    for row in rows:
        key = prompt_id(row["prompt"])
        candidate = {**row, "prompt_id": key}
        if key in unique:
            if (unique[key].get("chosen"), unique[key].get("rejected")) != (
                candidate.get("chosen"), candidate.get("rejected")
            ):
                raise ValueError(f"Conflicting duplicate prompt: {key}")
            continue
        unique[key] = candidate
    ordered = sorted(unique.values(), key=lambda row: stable_rank(seed, row["prompt"]))
    required = sum(count for _, count in allocations)
    if len(ordered) < required:
        raise ValueError(f"Need {required} unique prompts, found {len(ordered)}")
    output, cursor = {}, 0
    for name, count in allocations:
        output[name] = ordered[cursor:cursor + count]
        cursor += count
    assert_disjoint(output)
    return output


def assert_disjoint(partitions: dict[str, list[dict]]) -> None:
    owner = {}
    for name, rows in partitions.items():
        for row in rows:
            key = row["prompt_id"]
            if key in owner:
                raise ValueError(f"Prompt {key} appears in {owner[key]} and {name}")
            owner[key] = name


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
