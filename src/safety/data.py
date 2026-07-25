"""Deterministic data loading for safety-classifier experiments."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .taxonomy import JIGSAW_COLUMNS, map_jigsaw_labels


def stable_partition(identifier: str, seed: int = 2025) -> str:
    """Assign an id to train/calibration/test without row-order leakage."""

    digest = hashlib.sha256(f"{seed}:{identifier}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    return "train" if bucket < 80 else ("calibration" if bucket < 90 else "test")


def load_jigsaw_csv(path: str | Path, seed: int = 2025) -> pd.DataFrame:
    """Load Kaggle ``train.csv`` and attach targets plus a stable split."""

    frame = pd.read_csv(path)
    required = {"id", "comment_text", *JIGSAW_COLUMNS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Jigsaw CSV is missing columns: {sorted(missing)}")
    if frame["id"].duplicated().any():
        raise ValueError("Jigsaw ids must be unique")
    output = frame.loc[:, ["id", "comment_text", *JIGSAW_COLUMNS]].copy()
    output["comment_text"] = output["comment_text"].fillna("").astype(str)
    output["split"] = [stable_partition(str(value), seed) for value in output["id"]]
    output["labels"] = [map_jigsaw_labels(row) for row in output.to_dict("records")]
    return output


def load_adjacent_benign(path: str | Path) -> pd.DataFrame:
    """Load the frozen, all-negative fairness stress set."""

    frame = pd.read_csv(path)
    required = {"example_id", "slice", "text", "source_type"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Adjacent-benign CSV is missing columns: {sorted(missing)}")
    if frame["example_id"].duplicated().any():
        raise ValueError("Adjacent-benign example ids must be unique")
    if frame["text"].isna().any() or (frame["text"].str.strip() == "").any():
        raise ValueError("Adjacent-benign texts must be non-empty")
    return frame


@dataclass
class EncodedTextDataset(Dataset):
    texts: list[str]
    labels: np.ndarray
    tokenizer: object
    max_length: int = 256

    def __post_init__(self) -> None:
        if len(self.texts) != len(self.labels):
            raise ValueError("texts and labels must have equal length")

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[index],
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        encoded["labels"] = torch.as_tensor(self.labels[index], dtype=torch.float32)
        return encoded


def label_matrix(values: Iterable[np.ndarray]) -> np.ndarray:
    matrix = np.stack(list(values)).astype(np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != 3:
        raise ValueError(f"Expected an N x 3 label matrix, got {matrix.shape}")
    return matrix
