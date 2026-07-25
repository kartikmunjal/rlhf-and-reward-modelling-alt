from pathlib import Path

import pandas as pd
import pytest
import torch

from src.safety.data import MultiLabelDataCollator, load_adjacent_benign, load_jigsaw_csv


def test_jigsaw_loader_validates_and_maps(tmp_path: Path):
    path = tmp_path / "train.csv"
    pd.DataFrame(
        [
            {
                "id": "a",
                "comment_text": "example",
                "toxic": 0,
                "severe_toxic": 0,
                "obscene": 1,
                "threat": 0,
                "insult": 0,
                "identity_hate": 0,
            }
        ]
    ).to_csv(path, index=False)
    frame = load_jigsaw_csv(path)
    assert frame.iloc[0]["labels"].tolist() == [0, 1, 0]


def test_adjacent_benign_has_frozen_shape_and_slices():
    frame = load_adjacent_benign("data/adjacent_benign_v1.csv")
    assert len(frame) == 60
    assert frame.groupby("slice").size().to_dict() == {
        "clinical_medical": 20,
        "news_reporting_violence": 20,
        "reclaimed_language": 20,
    }


def test_adjacent_loader_rejects_duplicate_ids(tmp_path: Path):
    path = tmp_path / "bad.csv"
    pd.DataFrame(
        [
            {"example_id": "a", "slice": "x", "text": "one", "source_type": "test"},
            {"example_id": "a", "slice": "x", "text": "two", "source_type": "test"},
        ]
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="unique"):
        load_adjacent_benign(path)


def test_multilabel_collator_keeps_labels_out_of_tokenizer_padding():
    class RecordingTokenCollator:
        def __init__(self):
            self.features = None

        def __call__(self, features):
            self.features = features
            return {"input_ids": torch.tensor([[1, 2], [3, 0]])}

    collator = MultiLabelDataCollator.__new__(MultiLabelDataCollator)
    collator.token_collator = RecordingTokenCollator()
    batch = collator(
        [
            {"input_ids": [1, 2], "labels": torch.tensor([1.0, 0.0, 1.0])},
            {"input_ids": [3], "labels": torch.tensor([0.0, 1.0, 0.0])},
        ]
    )
    assert all("labels" not in feature for feature in collator.token_collator.features)
    assert batch["labels"].shape == (2, 3)
    assert batch["labels"].dtype == torch.float32
