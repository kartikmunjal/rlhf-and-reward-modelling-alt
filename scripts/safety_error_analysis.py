#!/usr/bin/env python3
"""Blind, export, and analyze deterministic safety-classifier errors."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_safety_classifier import write_text_artifact  # noqa: E402
from src.safety.data import label_matrix, load_jigsaw_csv  # noqa: E402
from src.safety.taxonomy import TARGET_LABELS  # noqa: E402

PHENOMENA = (
    "quotation",
    "negation",
    "counter_speech",
    "reclaimed_language",
    "identity_mention",
    "clinical_context",
    "news_context",
    "sarcasm_irony",
    "long_context_truncation",
    "ambiguous_gold_label",
    "other",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    export = subparsers.add_parser("export")
    export.add_argument("--predictions", required=True)
    export.add_argument("--jigsaw-csv", default="data/raw/jigsaw/train.csv")
    export.add_argument("--output", required=True)
    export.add_argument("--per-stratum", type=int, default=50)
    export.add_argument("--seed", type=int, default=2025)
    export.add_argument("--version", required=True)
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--annotations", required=True)
    analyze.add_argument("--output", required=True)
    analyze.add_argument("--n-trials", type=int, required=True)
    return parser.parse_args()


def blind_id(identifier: str, version: str) -> str:
    return hashlib.sha256(f"{version}:{identifier}".encode()).hexdigest()[:16]


def export_sheet(args: argparse.Namespace) -> None:
    predictions = np.load(args.predictions, allow_pickle=True)
    ids_key = "jigsaw_test_ids" if "jigsaw_test_ids" in predictions else "test_ids"
    prob_key = (
        "jigsaw_test_probabilities"
        if "jigsaw_test_probabilities" in predictions
        else "test_probabilities"
    )
    ids = [str(value) for value in predictions[ids_key]]
    probabilities = predictions[prob_key]
    thresholds = predictions["thresholds"]
    jigsaw = load_jigsaw_csv(args.jigsaw_csv, seed=2025)
    test = jigsaw[jigsaw["split"] == "test"]
    by_id = test.set_index(test["id"].astype(str))
    labels = label_matrix([by_id.loc[identifier, "labels"] for identifier in ids])
    texts = [by_id.loc[identifier, "comment_text"] for identifier in ids]
    prediction = probabilities >= thresholds
    rng = np.random.default_rng(args.seed)
    rows = []
    for column, category in enumerate(TARGET_LABELS):
        for error_type, mask in (
            ("false_positive", (labels[:, column] == 0) & prediction[:, column]),
            ("false_negative", (labels[:, column] == 1) & ~prediction[:, column]),
        ):
            candidates = np.flatnonzero(mask)
            selected = rng.choice(
                candidates, size=min(args.per_stratum, len(candidates)), replace=False
            )
            for index in selected:
                rows.append(
                    {
                        "blind_id": blind_id(ids[index], args.version),
                        "version": args.version,
                        "category": category,
                        "error_type": error_type,
                        "text": texts[index],
                        "primary_phenomenon_a": "",
                        "gold_label_valid_a": "",
                        "notes_a": "",
                        "primary_phenomenon_b": "",
                        "gold_label_valid_b": "",
                        "notes_b": "",
                    }
                )
    sheet = pd.DataFrame(rows).sort_values("blind_id")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(args.output, index=False, lineterminator="\n")
    codebook = {
        "version": args.version,
        "allowed_primary_phenomena": PHENOMENA,
        "gold_label_valid_values": ["yes", "no", "uncertain"],
        "instructions": (
            "Annotators work independently. Do not expose model score, threshold, trial, "
            "or the other annotator's columns before both columns are complete."
        ),
    }
    write_text_artifact(
        Path(args.output).with_suffix(".codebook.json"),
        json.dumps(codebook, indent=2) + "\n",
    )
    print(f"Wrote {len(sheet)} blinded errors")


def analyze_sheet(args: argparse.Namespace) -> None:
    frame = pd.read_csv(args.annotations).fillna("")
    required = {
        "blind_id",
        "primary_phenomenon_a",
        "gold_label_valid_a",
        "primary_phenomenon_b",
        "gold_label_valid_b",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Annotation sheet missing columns: {sorted(missing)}")
    for suffix in ("a", "b"):
        phenomena = set(frame[f"primary_phenomenon_{suffix}"])
        validity = set(frame[f"gold_label_valid_{suffix}"])
        if "" in phenomena or "" in validity:
            raise ValueError("Both independent annotation columns must be complete")
        if not phenomena.issubset(PHENOMENA):
            raise ValueError(f"Unknown phenomenon labels: {sorted(phenomena.difference(PHENOMENA))}")
        if not validity.issubset({"yes", "no", "uncertain"}):
            raise ValueError("gold_label_valid must be yes/no/uncertain")
    result = {
        "n_examples": int(len(frame)),
        "n_trials": int(args.n_trials),
        "phenomenon_exact_agreement": float(
            (
                frame["primary_phenomenon_a"]
                == frame["primary_phenomenon_b"]
            ).mean()
        ),
        "phenomenon_cohen_kappa": float(
            cohen_kappa_score(
                frame["primary_phenomenon_a"], frame["primary_phenomenon_b"]
            )
        ),
        "gold_validity_exact_agreement": float(
            (frame["gold_label_valid_a"] == frame["gold_label_valid_b"]).mean()
        ),
        "gold_validity_cohen_kappa": float(
            cohen_kappa_score(
                frame["gold_label_valid_a"], frame["gold_label_valid_b"]
            )
        ),
        "annotator_a_phenomenon_counts": frame[
            "primary_phenomenon_a"
        ].value_counts().sort_index().to_dict(),
        "annotator_b_phenomenon_counts": frame[
            "primary_phenomenon_b"
        ].value_counts().sort_index().to_dict(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_text_artifact(output, json.dumps(result, indent=2) + "\n")
    print(f"Wrote error analysis: {output}")


def main() -> None:
    args = parse_args()
    export_sheet(args) if args.command == "export" else analyze_sheet(args)


if __name__ == "__main__":
    main()
