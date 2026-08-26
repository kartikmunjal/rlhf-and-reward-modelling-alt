"""Leakage-safe CNN/DailyMail preparation and summarization datasets."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


SPACE = re.compile(r"\s+")


def normalized_text(text: str) -> str:
    return SPACE.sub(" ", text).strip().lower()


def text_sha256(text: str) -> str:
    return hashlib.sha256(normalized_text(text).encode("utf-8")).hexdigest()


def ranked(rows, salt: str) -> list[dict]:
    return sorted(rows, key=lambda row: hashlib.sha256(f"{salt}\0{row['id']}".encode()).hexdigest())


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(output_dir: Path, config: dict, summeval_manifest_path: Path) -> dict:
    from datasets import load_dataset

    output_dir.mkdir(parents=True, exist_ok=True)
    summeval = json.loads(summeval_manifest_path.read_text(encoding="utf-8"))
    excluded_ids = set(summeval["dev_article_ids"]) | set(summeval["heldout_article_ids"])
    dataset = load_dataset(config["data"]["dataset"], config["data"]["version"])
    expected = {"train": 287113, "validation": 13368, "test": 11490}
    observed = {split: len(dataset[split]) for split in expected}
    if observed != expected:
        raise ValueError(f"Unexpected CNN/DailyMail 3.0.0 split sizes: {observed}")
    test_by_id = {row["id"]: row for row in dataset["test"]}
    missing = excluded_ids - set(test_by_id)
    if missing:
        raise ValueError(f"SummEval IDs absent from CNN/DailyMail test: {len(missing)}")
    excluded_hashes = {text_sha256(test_by_id[item]["article"]) for item in excluded_ids}

    def contaminated(row: dict) -> bool:
        return row["id"] in excluded_ids or text_sha256(row["article"]) in excluded_hashes

    train = dataset["train"].filter(lambda row: not contaminated(row), desc="Excluding SummEval contamination")
    validation = [row for row in dataset["validation"] if not contaminated(row)]
    test = [row for row in dataset["test"] if not contaminated(row)]
    ordered_val = ranked(validation, config["data"]["split_salt"] + ":validation")
    pref_n, loss_n = config["data"]["preference_articles"], config["data"]["sft_eval_articles"]
    preference, sft_eval = ordered_val[:pref_n], ordered_val[pref_n:pref_n + loss_n]
    final_eval = ranked(test, config["data"]["split_salt"] + ":test")[:config["data"]["final_eval_articles"]]

    train_path = output_dir / "train.parquet"
    train.to_parquet(str(train_path))
    paths = {
        "preference": output_dir / "preference_articles.jsonl",
        "sft_eval": output_dir / "sft_eval_articles.jsonl",
        "final_eval": output_dir / "final_eval_articles.jsonl",
    }
    for key, rows in (("preference", preference), ("sft_eval", sft_eval), ("final_eval", final_eval)):
        write_jsonl(paths[key], ({"article_id": row["id"], "source": row["article"], "reference": row["highlights"]} for row in rows))

    ids = {
        "train": set(train["id"]), "preference": {row["id"] for row in preference},
        "sft_eval": {row["id"] for row in sft_eval}, "final_eval": {row["id"] for row in final_eval},
    }
    for left, left_ids in ids.items():
        for right, right_ids in ids.items():
            if left < right and left_ids & right_ids:
                raise ValueError(f"Partition overlap: {left}/{right}")
    if any(item in excluded_ids for values in ids.values() for item in values):
        raise ValueError("SummEval ID contamination remained after filtering")
    train_hashes = {text_sha256(row["article"]) for row in train}
    if train_hashes & excluded_hashes:
        raise ValueError("SummEval normalized-text contamination remained in training")

    manifest = {
        "study_id": config["study_id"], "status": "prepared_before_training",
        "dataset": {"name": config["data"]["dataset"], "version": config["data"]["version"],
                    "raw_split_counts": observed},
        "counts": {name: len(values) for name, values in ids.items()},
        "summ_eval_exclusion": {"ids": len(excluded_ids), "normalized_source_hashes": len(excluded_hashes),
                                   "id_overlap_after_filter": 0, "hash_overlap_after_filter": 0},
        "partition_ids": {name: sorted(values) for name, values in ids.items() if name != "train"},
        "outputs": {"train": {"path": str(train_path), "sha256": file_sha256(train_path)},
                    **{key: {"path": str(path), "sha256": file_sha256(path)} for key, path in paths.items()}},
        "source_manifest": {"path": str(summeval_manifest_path), "sha256": file_sha256(summeval_manifest_path)},
    }
    (output_dir / "data_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


class SummarizationSFTDataset:
    """Lazy causal-LM examples with loss restricted to reference tokens."""

    def __init__(self, parquet_path: Path, tokenizer, config: dict):
        from datasets import load_dataset
        self.rows = load_dataset("parquet", data_files=str(parquet_path), split="train")
        self.tokenizer, self.config = tokenizer, config

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import torch
        row = self.rows[index]
        prefix = self.tokenizer.encode("Article:\n", add_special_tokens=False)
        source = self.tokenizer.encode(row["article"], add_special_tokens=False)[:self.config["model"]["source_tokens"]]
        suffix = self.tokenizer.encode("\n\nSummary:\n", add_special_tokens=False)
        prompt = prefix + source + suffix
        room = self.config["model"]["context_tokens"] - len(prompt) - 1
        target = self.tokenizer.encode(row["highlights"], add_special_tokens=False)[:max(0, min(room, self.config["model"]["new_summary_tokens"]))]
        input_ids = prompt + target + [self.tokenizer.eos_token_id]
        labels = [-100] * len(prompt) + target + [self.tokenizer.eos_token_id]
        return {"input_ids": torch.tensor(input_ids), "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
                "labels": torch.tensor(labels)}


def causal_lm_collator(tokenizer):
    def collate(rows):
        import torch
        width = max(len(row["input_ids"]) for row in rows)
        result = {"input_ids": [], "attention_mask": [], "labels": []}
        for row in rows:
            pad = width - len(row["input_ids"])
            result["input_ids"].append(torch.cat([row["input_ids"], torch.full((pad,), tokenizer.pad_token_id)]))
            result["attention_mask"].append(torch.cat([row["attention_mask"], torch.zeros(pad, dtype=torch.long)]))
            result["labels"].append(torch.cat([row["labels"], torch.full((pad,), -100)]))
        return {key: torch.stack(value) for key, value in result.items()}
    return collate
