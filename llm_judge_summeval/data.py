"""SummEval preparation and partition-specific access controls."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable


AXES = ("coherence", "consistency", "fluency", "relevance")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def story_id(row: dict) -> str:
    return Path(row["filepath"]).stem


def summary_id(row: dict) -> str:
    """SummEval's `id` is article-level; model_id makes a summary unique."""
    return f"{story_id(row)}:{row['model_id']}"


def deterministic_split(article_ids: Iterable[str], salt: str, dev_count: int) -> tuple[list[str], list[str]]:
    unique = sorted(set(article_ids))
    ranked = sorted(unique, key=lambda value: hashlib.sha256(f"{salt}\0{value}".encode()).hexdigest())
    return sorted(ranked[:dev_count]), sorted(ranked[dev_count:])


def select_balanced_pairs(summary_ids: list[str], article_id: str, count: int, salt: str) -> list[tuple[str, str]]:
    ids = sorted(summary_ids)
    candidates = [(left, right) for index, left in enumerate(ids) for right in ids[index + 1 :]]
    participation: Counter[str] = Counter()
    selected: list[tuple[str, str]] = []
    while len(selected) < count:
        remaining = [pair for pair in candidates if pair not in selected]
        if not remaining:
            raise ValueError("Requested more unique pairs than available")

        def key(pair: tuple[str, str]) -> tuple:
            left, right = pair
            digest = hashlib.sha256(f"{salt}\0{article_id}\0{left}\0{right}".encode()).hexdigest()
            return (max(participation[left], participation[right]), participation[left] + participation[right], digest)

        chosen = min(remaining, key=key)
        selected.append(chosen)
        participation.update(chosen)
    return selected


def _expert_means(row: dict) -> dict[str, float]:
    annotations = row["expert_annotations"]
    if len(annotations) != 3 or any(set(item) != set(AXES) for item in annotations):
        raise ValueError(f"Unexpected expert annotation schema for {row['id']}")
    return {axis: mean(float(item[axis]) for item in annotations) for axis in AXES}


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def prepare_dataset(annotation_path: Path, cnndm_path: Path, output_dir: Path, config: dict) -> dict:
    import pyarrow.parquet as pq

    annotations = [json.loads(line) for line in annotation_path.read_text(encoding="utf-8").splitlines()]
    expected_articles = config["dataset"]["expected_articles"]
    expected_per_article = config["dataset"]["expected_summaries_per_article"]
    if len(annotations) != expected_articles * expected_per_article:
        raise ValueError("Unexpected SummEval annotation row count")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in annotations:
        grouped[story_id(row)].append(row)
    if len(grouped) != expected_articles or set(map(len, grouped.values())) != {expected_per_article}:
        raise ValueError("Expected exactly 100 articles with 16 summaries each")

    table = pq.read_table(cnndm_path, columns=["id", "article", "highlights"])
    sources = {row["id"]: row for row in table.to_pylist() if row["id"] in grouped}
    if set(sources) != set(grouped):
        raise ValueError(f"Missing or duplicate CNN/DailyMail sources: {sorted(set(grouped) - set(sources))}")

    dev_ids, heldout_ids = deterministic_split(
        grouped, config["dataset"]["split_salt"], config["dataset"]["dev_articles"]
    )
    dev_set, heldout_set = set(dev_ids), set(heldout_ids)
    dev_rows, heldout_inputs, heldout_labels = [], [], []
    pair_rows = []
    for article_id in sorted(grouped):
        source = sources[article_id]
        summary_rows = sorted(grouped[article_id], key=summary_id)
        for row in summary_rows:
            common = {
                "article_id": article_id,
                "summary_id": summary_id(row),
                "model_id": row["model_id"],
                "source": source["article"],
                "reference": source["highlights"],
                "summary": row["decoded"],
            }
            labels = {"article_id": article_id, "summary_id": summary_id(row), "expert_mean": _expert_means(row)}
            if article_id in dev_set:
                dev_rows.append(common | labels)
            elif article_id in heldout_set:
                heldout_inputs.append(common)
                heldout_labels.append(labels)
            else:
                raise AssertionError("Article absent from split")
        if article_id in heldout_set:
            pairs = select_balanced_pairs(
                [summary_id(row) for row in summary_rows], article_id,
                config["pairwise"]["pairs_per_heldout_article"], config["dataset"]["split_salt"],
            )
            for pair_index, (left, right) in enumerate(pairs):
                pair_rows.append({
                    "pair_id": f"{article_id}:{pair_index}", "article_id": article_id,
                    "summary_a_id": left, "summary_b_id": right,
                })

    paths = {
        "dev": output_dir / "dev_with_labels.jsonl",
        "heldout_inputs": output_dir / "heldout_inputs.jsonl",
        "heldout_labels": output_dir / "heldout_labels.sealed.jsonl",
        "heldout_pairs": output_dir / "heldout_pairs.jsonl",
    }
    for key, rows in (("dev", dev_rows), ("heldout_inputs", heldout_inputs),
                      ("heldout_labels", heldout_labels), ("heldout_pairs", pair_rows)):
        _write_jsonl(paths[key], rows)
    manifest = {
        "study_id": config["study_id"],
        "status": "prepared_no_judge_calls",
        "sources": {
            "summeval_annotations": {
                "path": str(annotation_path), "sha256": file_sha256(annotation_path),
                "url": "https://storage.googleapis.com/sfr-summarization-repo-research/model_annotations.aligned.jsonl",
                "official_repository_commit": "81b59ad53d63cb6009764240853c91235a44e238",
            },
            "cnndm_test": {
                "path": str(cnndm_path), "sha256": file_sha256(cnndm_path),
                "dataset": "abisee/cnn_dailymail", "dataset_version": "3.0.0", "split": "test",
                "lfs_sha256": "04e322d2634a96dba76bf9a6294fbbe48e0b36abeae43f13d86ba2c3bebffe4e",
            },
        },
        "counts": {
            "articles": len(grouped), "summaries": len(annotations),
            "dev_articles": len(dev_ids), "dev_summaries": len(dev_rows),
            "heldout_articles": len(heldout_ids), "heldout_summaries": len(heldout_inputs),
            "heldout_pairs": len(pair_rows),
        },
        "dev_article_ids": dev_ids,
        "heldout_article_ids": heldout_ids,
        "outputs": {key: {"path": str(path), "sha256": file_sha256(path)} for key, path in paths.items()},
    }
    if manifest["counts"]["heldout_pairs"] != config["pairwise"]["expected_unique_pairs"]:
        raise ValueError("Pair count violates preregistration")
    (output_dir / "data_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def load_prompt_development_rows(processed_dir: Path) -> list[dict]:
    """Only development rows are accessible to prompt-iteration code."""
    return [json.loads(line) for line in (processed_dir / "dev_with_labels.jsonl").read_text(encoding="utf-8").splitlines()]


def load_heldout_inputs(processed_dir: Path) -> list[dict]:
    """Judge runners receive text and identifiers, never held-out labels."""
    rows = [json.loads(line) for line in (processed_dir / "heldout_inputs.jsonl").read_text(encoding="utf-8").splitlines()]
    if any("expert_mean" in row for row in rows):
        raise ValueError("Held-out input leakage detected")
    return rows


def load_heldout_labels_for_analysis(processed_dir: Path, *, prompt_manifest_verified: bool) -> list[dict]:
    """Analysis must explicitly prove the prompt was frozen before labels load."""
    if not prompt_manifest_verified:
        raise PermissionError("Held-out labels are sealed until the prompt manifest is verified")
    return [json.loads(line) for line in (processed_dir / "heldout_labels.sealed.jsonl").read_text(encoding="utf-8").splitlines()]
