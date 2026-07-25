#!/usr/bin/env python3
"""Download, normalize, pin, and hash every external v2 dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.safety.v2_data import (  # noqa: E402
    dataframe_sha256,
    normalize_beavertails,
    normalize_civil_comments,
    normalize_hatecheck,
    normalize_toxigen,
)

DATASETS = {
    "beavertails": "PKU-Alignment/BeaverTails",
    "toxigen": "toxigen/toxigen-data",
    "hatecheck": "Paul/hatecheck",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--civil-comments-csv",
        required=True,
        help="Original Jigsaw unintended-bias train.csv with identity columns",
    )
    parser.add_argument("--output-dir", default="data/processed/safety_v2")
    parser.add_argument(
        "--revision-lock",
        help="Existing data_manifest.json; when supplied, exact HF revisions are reused",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def revision_map(lock_path: str | None) -> dict[str, str]:
    if lock_path:
        lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
        return {
            name: str(lock["sources"][name]["revision"])
            for name in DATASETS
        }
    api = HfApi()
    return {name: api.dataset_info(repo).sha for name, repo in DATASETS.items()}


def save_frame(frame: pd.DataFrame, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = frame.copy()
    if "labels" in serializable:
        serializable["labels"] = [value.tolist() for value in serializable["labels"]]
    serializable.to_parquet(path, index=False)
    return {
        "path": str(path),
        "rows": int(len(frame)),
        "sha256": file_sha256(path),
        "semantic_sha256": dataframe_sha256(frame),
    }


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    revisions = revision_map(args.revision_lock)

    beaver_train = normalize_beavertails(
        load_dataset(
            DATASETS["beavertails"],
            split="330k_train",
            revision=revisions["beavertails"],
        ),
        "330k_train",
    )
    beaver_test = normalize_beavertails(
        load_dataset(
            DATASETS["beavertails"],
            split="330k_test",
            revision=revisions["beavertails"],
        ),
        "330k_test",
    )
    toxigen_train = normalize_toxigen(
        load_dataset(
            DATASETS["toxigen"],
            "annotated",
            split="train",
            revision=revisions["toxigen"],
        ),
        "train",
    )
    toxigen_test = normalize_toxigen(
        load_dataset(
            DATASETS["toxigen"],
            "annotated",
            split="test",
            revision=revisions["toxigen"],
        ),
        "test",
    )
    hatecheck = normalize_hatecheck(
        load_dataset(
            DATASETS["hatecheck"],
            split="test",
            revision=revisions["hatecheck"],
        )
    )
    civil_path = Path(args.civil_comments_csv)
    civil = normalize_civil_comments(pd.read_csv(civil_path), "external_test")

    normalized_files = {
        "beavertails_train": save_frame(beaver_train, output / "beavertails_train.parquet"),
        "beavertails_test": save_frame(beaver_test, output / "beavertails_test.parquet"),
        "toxigen_train": save_frame(toxigen_train, output / "toxigen_train.parquet"),
        "toxigen_test": save_frame(toxigen_test, output / "toxigen_test.parquet"),
        "hatecheck": save_frame(hatecheck, output / "hatecheck.parquet"),
        "civil_comments": save_frame(civil, output / "civil_comments.parquet"),
    }
    manifest = {
        "manifest_version": 1,
        "preregistration": "docs/safety_v2_preregistered_plan.md",
        "sources": {
            name: {"repository": DATASETS[name], "revision": revisions[name]}
            for name in DATASETS
        },
        "civil_comments_source": {
            "path": str(civil_path),
            "sha256": file_sha256(civil_path),
        },
        "normalized_files": normalized_files,
    }
    manifest_path = output / "data_manifest.json"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(f"Wrote locked data manifest: {manifest_path}")


if __name__ == "__main__":
    main()
