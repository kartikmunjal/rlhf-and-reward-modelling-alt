import numpy as np
import pandas as pd
import pytest

from src.safety.v2_data import (
    IDENTITY_COLUMNS,
    combine_v2_training,
    map_beavertails_categories,
    normalize_beavertails,
    normalize_civil_comments,
    normalize_hatecheck,
    normalize_toxigen,
    v2_calibration_role,
)


def beaver_categories(**active):
    values = {
        "hate_speech,offensive_language": False,
        "discrimination,stereotype,injustice": False,
        "sexually_explicit,adult_content": False,
        "violence,aiding_and_abetting,incitement": False,
    }
    values.update(active)
    return values


def test_locked_beavertails_mapping_preserves_multilabel_overlap():
    mapped = map_beavertails_categories(
        beaver_categories(
            **{
                "discrimination,stereotype,injustice": True,
                "sexually_explicit,adult_content": True,
                "violence,aiding_and_abetting,incitement": True,
            }
        )
    )
    assert mapped.tolist() == [1, 1, 1]


def test_beavertails_mapping_rejects_schema_drift():
    with pytest.raises(ValueError, match="category missing"):
        map_beavertails_categories({})


def test_beavertails_normalization_combines_prompt_and_response():
    frame = normalize_beavertails(
        [
            {
                "prompt": "prompt",
                "response": "response",
                "category": beaver_categories(),
            }
        ],
        "330k_train",
    )
    assert frame.iloc[0]["text"] == "[PROMPT] prompt\n[RESPONSE] response"
    assert frame.iloc[0]["source_split"] == "330k_train"
    assert frame.iloc[0]["annotation_count"] == 1


def test_beavertails_uses_strict_majority_and_reports_disagreement():
    negative = {
        "prompt": "same prompt",
        "response": "same response",
        "category": beaver_categories(),
    }
    positive = {
        **negative,
        "category": beaver_categories(
            **{"violence,aiding_and_abetting,incitement": True}
        ),
    }
    frame, audit = normalize_beavertails(
        [positive, positive, negative],
        "330k_train",
        return_audit=True,
    )
    assert len(frame) == 1
    assert frame.iloc[0]["labels"].tolist() == [0, 0, 1]
    assert frame.iloc[0]["annotation_count"] == 3
    assert audit["raw_annotation_rows"] == 3
    assert audit["unique_prompt_response_pairs"] == 1
    assert audit["pairs_with_target_disagreement"]["harmful_violent"] == 1


def test_beavertails_excludes_entire_pair_on_any_target_tie():
    base = {
        "prompt": "same prompt",
        "response": "same response",
    }
    frame, audit = normalize_beavertails(
        [
            {**base, "category": beaver_categories()},
            {
                **base,
                "category": beaver_categories(
                    **{"hate_speech,offensive_language": True}
                ),
            },
        ],
        "330k_train",
        return_audit=True,
    )
    assert frame.empty
    assert audit["retained_pairs"] == 0
    assert audit["excluded_tie_pairs"] == 1
    assert audit["pairs_with_target_tie"]["hate_harassment"] == 1


def test_toxigen_human_cutoff_is_locked_at_three():
    frame = normalize_toxigen(
        [
            {"text": "a", "target_group": "x", "toxicity_human": 2.99},
            {"text": "b", "target_group": "x", "toxicity_human": 3.0},
        ],
        "test",
    )
    assert frame["gold"].tolist() == [0, 1]


def test_hatecheck_label_adapter():
    frame = normalize_hatecheck(
        [
            {"case_id": "1", "test_case": "x", "label_gold": "hateful", "functionality": "f1"},
            {
                "case_id": "2",
                "test_case": "y",
                "label_gold": "non-hateful",
                "functionality": "f2",
            },
        ]
    )
    assert frame["gold"].tolist() == [1, 0]


def test_civil_comments_requires_original_identity_export():
    with pytest.raises(ValueError, match="original unintended-bias export"):
        normalize_civil_comments(pd.DataFrame({"id": ["1"]}))


def test_civil_comments_mapping_and_identity_columns():
    row = {
        "id": 1,
        "comment_text": "text",
        "target": 0.1,
        "severe_toxicity": 0.2,
        "obscene": 0.3,
        "threat": 0.7,
        "insult": 0.6,
        "identity_attack": 0.4,
        "sexual_explicit": 0.8,
        **{name: 0.0 for name in IDENTITY_COLUMNS},
    }
    row["female"] = 0.9
    frame = normalize_civil_comments(pd.DataFrame([row]))
    assert frame.iloc[0]["hate_harassment"] == 0.6
    assert frame.iloc[0]["sexualized"] == 0.8
    assert frame.iloc[0]["harmful_violent"] == 0.7
    assert frame.iloc[0]["female"] == 0.9


def test_combined_training_prefixes_source_ids():
    jigsaw = pd.DataFrame(
        {"id": ["1"], "comment_text": ["a"], "labels": [np.zeros(3)]}
    )
    beaver = pd.DataFrame(
        {"id": ["1"], "text": ["b"], "labels": [np.ones(3)]}
    )
    combined = combine_v2_training(jigsaw, beaver)
    assert combined["id"].tolist() == ["jigsaw:1", "beavertails:1"]


def test_v2_calibration_roles_are_deterministic_and_disjoint():
    roles = [v2_calibration_role(str(index)) for index in range(100)]
    assert roles == [v2_calibration_role(str(index)) for index in range(100)]
    assert set(roles) == {"threshold_calibration", "model_selection"}
