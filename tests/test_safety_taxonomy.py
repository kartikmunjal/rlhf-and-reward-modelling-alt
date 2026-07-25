import numpy as np
import pytest

from src.safety.data import stable_partition
from src.safety.taxonomy import map_jigsaw_labels


def row(**overrides):
    values = {
        "toxic": 0,
        "severe_toxic": 0,
        "obscene": 0,
        "threat": 0,
        "insult": 0,
        "identity_hate": 0,
    }
    values.update(overrides)
    return values


@pytest.mark.parametrize("source", ["toxic", "severe_toxic", "insult", "identity_hate"])
def test_hate_harassment_is_union_of_locked_sources(source):
    assert np.array_equal(map_jigsaw_labels(row(**{source: 1})), [1, 0, 0])


def test_multilabel_mapping_preserves_overlap():
    assert np.array_equal(
        map_jigsaw_labels(row(identity_hate=1, obscene=1, threat=1)), [1, 1, 1]
    )


def test_mapping_rejects_incomplete_rows():
    with pytest.raises(ValueError, match="Missing Jigsaw columns"):
        map_jigsaw_labels({"toxic": 1})


def test_split_is_deterministic_and_identifier_based():
    first = [stable_partition(str(i), seed=2025) for i in range(100)]
    second = [stable_partition(str(i), seed=2025) for i in reversed(range(100))]
    assert first == list(reversed(second))
    assert set(first) == {"train", "calibration", "test"}
