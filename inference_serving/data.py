"""Locked partitions and prompt construction for inference-serving v1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def stable_rank(seed: int, namespace: str, item_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{namespace}\0{item_id}".encode()).hexdigest()


def article_id(row: dict) -> str:
    for key in ("article_id", "id", "story_id"):
        if row.get(key):
            return str(row[key])
    raise KeyError("Article row lacks a stable ID")


def partition_final_articles(rows: list[dict], config: dict) -> tuple[list[dict], list[dict]]:
    expected = config["data"]["pilot_articles"] + config["data"]["heldout_articles"]
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} final articles, found {len(rows)}")
    ids = [article_id(row) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate final-evaluation article IDs")
    ranked = sorted(rows, key=lambda row: stable_rank(config["seed"], "serving-final", article_id(row)))
    cut = config["data"]["pilot_articles"]
    return ranked[:cut], ranked[cut:]


def prompt_text(source: str, config: dict) -> str:
    return config["generation"]["prompt_template"].format(source=source)


def prepare_partitions(config: dict, source_path: Path, output_dir: Path) -> dict:
    rows = read_jsonl(source_path)
    pilot, heldout = partition_final_articles(rows, config)
    write_jsonl(output_dir / "pilot_articles.jsonl", pilot)
    write_jsonl(output_dir / "heldout_articles.jsonl", heldout)
    manifest = {
        "study_id": config["study_id"],
        "partition_method": config["data"]["partition_method"],
        "pilot": {"count": len(pilot), "ids": [article_id(row) for row in pilot]},
        "heldout": {"count": len(heldout), "ids": [article_id(row) for row in heldout]},
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "partition_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def prepare_calibration(config: dict, train_parquet: Path, output_dir: Path) -> dict:
    import pandas as pd
    count = config["stage2"]["calibration_articles"]
    frame = pd.read_parquet(train_parquet, columns=["article_id", "source", "reference"])
    frame["rank"] = frame["article_id"].astype(str).map(
        lambda value: stable_rank(config["seed"], "gptq-calibration", value)
    )
    rows = frame.sort_values("rank").head(count).drop(columns=["rank"]).to_dict("records")
    if len(rows) != count:
        raise ValueError(f"Expected {count} calibration rows, found {len(rows)}")
    write_jsonl(output_dir / "calibration_articles.jsonl", rows)
    return {"count": len(rows), "ids": [article_id(row) for row in rows]}


def verify_tokenizer_identity(tokenizers: dict[str, object]) -> dict:
    names = list(tokenizers)
    if not names:
        raise ValueError("No tokenizers supplied")
    reference = tokenizers[names[0]]
    reference_vocab = reference.get_vocab()
    reference_special = reference.special_tokens_map
    for name in names[1:]:
        candidate = tokenizers[name]
        if candidate.get_vocab() != reference_vocab:
            raise ValueError(f"Tokenizer vocabulary mismatch: {names[0]} vs {name}")
        if candidate.special_tokens_map != reference_special:
            raise ValueError(f"Tokenizer special-token mismatch: {names[0]} vs {name}")
    return {"tokenizers": names, "vocab_size": len(reference_vocab), "identical": True}
