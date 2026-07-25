"""Locked adapters for the safety-classifier v2 external datasets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Literal, Mapping, overload

import numpy as np
import pandas as pd

BEAVER_TARGET_MAP = {
    "hate_harassment": (
        "hate_speech,offensive_language",
        "discrimination,stereotype,injustice",
    ),
    "sexualized": ("sexually_explicit,adult_content",),
    "harmful_violent": ("violence,aiding_and_abetting,incitement",),
}

IDENTITY_COLUMNS = (
    "male",
    "female",
    "transgender",
    "other_gender",
    "heterosexual",
    "homosexual_gay_or_lesbian",
    "bisexual",
    "other_sexual_orientation",
    "christian",
    "jewish",
    "muslim",
    "hindu",
    "buddhist",
    "atheist",
    "other_religion",
    "black",
    "white",
    "asian",
    "latino",
    "other_race_or_ethnicity",
    "physical_disability",
    "intellectual_or_learning_disability",
    "psychiatric_or_mental_illness",
    "other_disability",
)


def content_id(*parts: str) -> str:
    joined = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def v2_calibration_role(identifier: str) -> str:
    """Split v1 calibration IDs into threshold and model-selection roles."""

    digest = hashlib.sha256(f"safety-v2-role:{identifier}".encode()).digest()
    return "threshold_calibration" if digest[0] < 128 else "model_selection"


def _category_value(categories: Mapping[str, object], name: str) -> bool:
    if name not in categories:
        raise ValueError(f"BeaverTails category missing from schema: {name}")
    return bool(categories[name])


def map_beavertails_categories(categories: Mapping[str, object]) -> np.ndarray:
    return np.asarray(
        [
            any(_category_value(categories, key) for key in BEAVER_TARGET_MAP["hate_harassment"]),
            any(_category_value(categories, key) for key in BEAVER_TARGET_MAP["sexualized"]),
            any(_category_value(categories, key) for key in BEAVER_TARGET_MAP["harmful_violent"]),
        ],
        dtype=np.float32,
    )


@overload
def normalize_beavertails(
    records: Iterable[Mapping[str, object]],
    source_split: str,
    *,
    return_audit: Literal[False] = False,
) -> pd.DataFrame: ...


@overload
def normalize_beavertails(
    records: Iterable[Mapping[str, object]],
    source_split: str,
    *,
    return_audit: Literal[True],
) -> tuple[pd.DataFrame, dict[str, object]]: ...


def normalize_beavertails(
    records: Iterable[Mapping[str, object]],
    source_split: str,
    *,
    return_audit: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, object]]:
    """Aggregate repeated annotation rows into one strictly-majority-labeled pair.

    BeaverTails 330k contains repeated prompt-response rows representing
    annotations. A pair is excluded when any mapped target receives exactly
    half positive votes; otherwise each target is assigned by strict majority.
    """

    annotations: dict[str, dict[str, object]] = {}
    raw_rows = 0
    for record in records:
        raw_rows += 1
        prompt = str(record.get("prompt", "")).strip()
        response = str(record.get("response", "")).strip()
        categories = record.get("category")
        if not prompt or not response or not isinstance(categories, Mapping):
            raise ValueError("BeaverTails requires non-empty prompt/response and category dict")
        identifier = content_id(prompt, response)
        if identifier not in annotations:
            annotations[identifier] = {
                "id": identifier,
                "text": f"[PROMPT] {prompt}\n[RESPONSE] {response}",
                "source": "beavertails",
                "source_split": source_split,
                "votes": [],
            }
        annotations[identifier]["votes"].append(map_beavertails_categories(categories))

    annotation_counts = [len(annotation["votes"]) for annotation in annotations.values()]
    rows = []
    disagreement_pairs = np.zeros(3, dtype=np.int64)
    tie_pairs = np.zeros(3, dtype=np.int64)
    excluded_tie_pairs = 0
    for annotation in annotations.values():
        votes = np.stack(annotation.pop("votes"))
        positive_votes = votes.sum(axis=0)
        annotation_count = len(votes)
        disagreement_pairs += ((positive_votes > 0) & (positive_votes < annotation_count))
        ties = positive_votes * 2 == annotation_count
        tie_pairs += ties
        if ties.any():
            excluded_tie_pairs += 1
            continue
        annotation["labels"] = (positive_votes * 2 > annotation_count).astype(np.float32)
        annotation["annotation_count"] = annotation_count
        rows.append(annotation)

    frame = pd.DataFrame(
        rows,
        columns=[
            "id",
            "text",
            "source",
            "source_split",
            "labels",
            "annotation_count",
        ],
    )
    if frame["id"].duplicated().any():
        raise AssertionError("BeaverTails aggregation did not produce unique pairs")
    audit = {
        "policy": "strict_majority_per_target_exclude_pair_on_any_target_tie",
        "raw_annotation_rows": raw_rows,
        "unique_prompt_response_pairs": len(annotations),
        "retained_pairs": len(frame),
        "excluded_tie_pairs": excluded_tie_pairs,
        "pairs_with_target_disagreement": {
            name: int(disagreement_pairs[index])
            for index, name in enumerate(BEAVER_TARGET_MAP)
        },
        "pairs_with_target_tie": {
            name: int(tie_pairs[index])
            for index, name in enumerate(BEAVER_TARGET_MAP)
        },
        "annotation_count_distribution": {
            str(count): int(frequency)
            for count, frequency in sorted(
                pd.Series(
                    annotation_counts, dtype=np.int64
                ).value_counts().items()
            )
        },
    }
    return (frame, audit) if return_audit else frame


def normalize_toxigen(records: Iterable[Mapping[str, object]], source_split: str) -> pd.DataFrame:
    rows = []
    for record in records:
        text = str(record.get("text", "")).strip()
        group = str(record.get("target_group", "")).strip()
        toxicity = float(record.get("toxicity_human", np.nan))
        if not text or not group or not np.isfinite(toxicity):
            raise ValueError("ToxiGen annotated rows require text, target_group, toxicity_human")
        rows.append(
            {
                "id": content_id(text, group),
                "text": text,
                "target_group": group,
                "gold": int(toxicity >= 3.0),
                "toxicity_human": toxicity,
                "source_split": source_split,
            }
        )
    frame = pd.DataFrame(rows)
    if frame["id"].duplicated().any():
        raise ValueError("Duplicate ToxiGen text/group rows")
    return frame


def normalize_hatecheck(records: Iterable[Mapping[str, object]]) -> pd.DataFrame:
    rows = []
    for record in records:
        text = str(record.get("test_case", record.get("text", ""))).strip()
        label = str(record.get("label_gold", record.get("label", ""))).strip().lower()
        functionality = str(record.get("functionality", "")).strip()
        if label in {"hateful", "hate", "1", "true"}:
            gold = 1
        elif label in {"non-hateful", "non_hateful", "not hateful", "0", "false"}:
            gold = 0
        else:
            raise ValueError(f"Unknown HateCheck gold label: {label!r}")
        if not text or not functionality:
            raise ValueError("HateCheck requires text and functionality")
        rows.append(
            {
                "id": str(record.get("case_id", content_id(text, functionality))),
                "text": text,
                "gold": gold,
                "functionality": functionality,
                "target_identity": str(record.get("target_ident", "")),
            }
        )
    frame = pd.DataFrame(rows)
    if frame["id"].duplicated().any():
        raise ValueError("Duplicate HateCheck case ids")
    return frame


def normalize_civil_comments(frame: pd.DataFrame, source_split: str = "test") -> pd.DataFrame:
    required = {
        "id",
        "comment_text",
        "severe_toxicity",
        "obscene",
        "threat",
        "insult",
        "identity_attack",
        "sexual_explicit",
        *IDENTITY_COLUMNS,
    }
    missing = required.difference(frame.columns)
    toxicity_column = "target" if "target" in frame.columns else "toxicity"
    if missing or toxicity_column not in frame.columns:
        raise ValueError(
            "Civil Comments must be the original unintended-bias export; "
            f"missing columns: {sorted(missing | ({'target_or_toxicity'} if toxicity_column not in frame.columns else set()))}"
        )
    output = pd.DataFrame(
        {
            "id": frame["id"].astype(str),
            "text": frame["comment_text"].fillna("").astype(str),
            "hate_harassment": pd.concat(
                [
                    frame[toxicity_column],
                    frame["severe_toxicity"],
                    frame["insult"],
                    frame["identity_attack"],
                ],
                axis=1,
            ).max(axis=1),
            "sexualized": frame["sexual_explicit"].astype(float),
            "harmful_violent": frame["threat"].astype(float),
            "source_split": source_split,
        }
    )
    for name in IDENTITY_COLUMNS:
        output[name] = frame[name].astype(float)
    if output["id"].duplicated().any():
        raise ValueError("Civil Comments ids must be unique")
    return output


def combine_v2_training(jigsaw_train: pd.DataFrame, beaver_train: pd.DataFrame) -> pd.DataFrame:
    jigsaw = pd.DataFrame(
        {
            "id": "jigsaw:" + jigsaw_train["id"].astype(str),
            "text": jigsaw_train["comment_text"],
            "labels": jigsaw_train["labels"],
            "source": "jigsaw",
        }
    )
    beaver = pd.DataFrame(
        {
            "id": "beavertails:" + beaver_train["id"].astype(str),
            "text": beaver_train["text"],
            "labels": beaver_train["labels"],
            "source": "beavertails",
        }
    )
    combined = pd.concat([jigsaw, beaver], ignore_index=True)
    if combined["id"].duplicated().any():
        raise ValueError("Combined v2 training ids must be unique")
    return combined


def dataframe_sha256(frame: pd.DataFrame) -> str:
    """Hash a normalized frame deterministically without relying on parquet bytes."""

    digest = hashlib.sha256()
    for row in frame.sort_values("id").to_dict("records"):
        normalized = {}
        for key, value in row.items():
            if isinstance(value, np.ndarray):
                normalized[key] = value.tolist()
            elif pd.isna(value) if not isinstance(value, (list, dict)) else False:
                normalized[key] = None
            else:
                normalized[key] = value
        digest.update(json.dumps(normalized, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def verify_manifested_file(path: str | Path, manifest: Mapping[str, object], key: str) -> None:
    expected = str(manifest["normalized_files"][key]["sha256"])
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != expected:
        raise ValueError(f"Manifest hash mismatch for {key}: {digest} != {expected}")
