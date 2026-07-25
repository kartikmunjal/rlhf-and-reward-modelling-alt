"""Locked mapping from Jigsaw labels to the job-description taxonomy."""

from __future__ import annotations

from typing import Mapping

import numpy as np

JIGSAW_COLUMNS = (
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
)
TARGET_LABELS = ("hate_harassment", "sexualized", "harmful_violent")
TAXONOMY_VERSION = "jigsaw-jd-v1"


def map_jigsaw_labels(row: Mapping[str, int | float]) -> np.ndarray:
    """Map one Jigsaw row to the preregistered three-label taxonomy.

    hate_harassment := toxic OR severe_toxic OR insult OR identity_hate
    sexualized      := obscene
    harmful_violent := threat
    """

    missing = set(JIGSAW_COLUMNS).difference(row)
    if missing:
        raise ValueError(f"Missing Jigsaw columns: {sorted(missing)}")
    return np.asarray(
        [
            any(float(row[name]) > 0 for name in ("toxic", "severe_toxic", "insult", "identity_hate")),
            float(row["obscene"]) > 0,
            float(row["threat"]) > 0,
        ],
        dtype=np.float32,
    )
