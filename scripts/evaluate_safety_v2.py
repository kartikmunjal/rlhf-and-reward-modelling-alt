#!/usr/bin/env python3
"""Evaluate one v2 trial on every preregistered internal and external set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_safety_classifier import (  # noqa: E402
    load_safety_model,
    predict,
    write_text_artifact,
)
from src.safety.data import label_matrix, load_adjacent_benign, load_jigsaw_csv  # noqa: E402
from src.safety.metrics import (  # noqa: E402
    classification_report,
    fairness_report,
    select_thresholds,
)
from src.safety.taxonomy import TARGET_LABELS  # noqa: E402
from src.safety.v2_data import (  # noqa: E402
    IDENTITY_COLUMNS,
    v2_calibration_role,
    verify_manifested_file,
)
from src.safety.v2_metrics import (  # noqa: E402
    binary_diagnostic_report,
    calibration_report,
    identity_fairness_report,
    threshold_sensitivity,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--config", default="configs/safety_v2.yaml")
    parser.add_argument("--ledger", default="configs/safety_v2_trial_ledger.json")
    parser.add_argument("--jigsaw-csv", default="data/raw/jigsaw/train.csv")
    parser.add_argument("--data-dir", default="data/processed/safety_v2")
    parser.add_argument("--checkpoint-root", default="checkpoints/safety_v2")
    parser.add_argument("--output-root", default="results/safety_v2")
    parser.add_argument("--adjacent-benign", default="data/adjacent_benign_v1.csv")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def completed_trials(checkpoint_root: Path, ledger: dict) -> list[str]:
    return [
        trial["trial_id"]
        for trial in ledger["trials"]
        if (checkpoint_root / trial["trial_id"] / "run_manifest.json").exists()
    ]


def compact_markdown(payload: dict) -> str:
    perf = payload["jigsaw_performance"]
    calibration = payload["jigsaw_calibration"]
    lines = [
        f'# Safety v2 trial: {payload["trial_id"]}',
        "",
        f'- N_trials completed: {payload["n_trials_completed"]}',
        f'- N_trials planned: {payload["n_trials_planned"]}',
        f'- Full training run: `{payload["manifest"]["full_run"]}`',
        "",
        "| Category | F1 (95% CI) | PR-AUC (95% CI) | Brier (95% CI) | ECE (95% CI) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in TARGET_LABELS:
        f1 = perf["per_category"][name]["f1"]
        item = calibration["per_category"][name]
        cells = []
        for metric in (f1, item["pr_auc"], item["brier"], item["ece"]):
            cells.append(
                f'{metric["value"]:.3f} [{metric["ci95"][0]:.3f}, {metric["ci95"][1]:.3f}]'
            )
        lines.append(f"| {name} | {' | '.join(cells)} |")
    lines.extend(
        [
            "",
            "External results and identity slices are available in `metrics.json`.",
            "This trial is not selectable until all 12 preregistered trials finish.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    ledger = json.loads(Path(args.ledger).read_text(encoding="utf-8"))
    checkpoint_root = Path(args.checkpoint_root)
    checkpoint = checkpoint_root / args.trial_id
    manifest = json.loads((checkpoint / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest["trial"]["trial_id"] != args.trial_id:
        raise ValueError("Checkpoint trial id mismatch")
    completed = completed_trials(checkpoint_root, ledger)
    n_trials = len(completed)
    if n_trials == 0:
        raise ValueError("No completed v2 trials found")

    data_dir = Path(args.data_dir)
    data_manifest = json.loads((data_dir / "data_manifest.json").read_text(encoding="utf-8"))
    frames = {}
    for key in (
        "beavertails_test",
        "toxigen_train",
        "toxigen_test",
        "hatecheck",
        "civil_comments",
    ):
        path = data_dir / f"{key}.parquet"
        verify_manifested_file(path, data_manifest, key)
        frames[key] = pd.read_parquet(path)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = load_safety_model(checkpoint)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model.to(device)
    inference = {
        "max_length": int(config["max_length"]),
        "batch_size": int(config["train"]["eval_batch_size"]),
        "device": device,
    }

    jigsaw = load_jigsaw_csv(args.jigsaw_csv, seed=2025)
    calibration_all = jigsaw[jigsaw["split"] == "calibration"].copy()
    calibration_all["v2_role"] = [
        v2_calibration_role(str(identifier)) for identifier in calibration_all["id"]
    ]
    calibration = calibration_all[
        calibration_all["v2_role"] == "threshold_calibration"
    ]
    selection = calibration_all[calibration_all["v2_role"] == "model_selection"]
    test = jigsaw[jigsaw["split"] == "test"]
    adjacent = load_adjacent_benign(args.adjacent_benign)
    calibration_prob = predict(model, tokenizer, calibration["comment_text"], **inference)
    selection_prob = predict(model, tokenizer, selection["comment_text"], **inference)
    test_prob = predict(model, tokenizer, test["comment_text"], **inference)
    adjacent_prob = predict(model, tokenizer, adjacent["text"], **inference)

    evaluation = config["evaluation"]
    grid = np.arange(
        float(evaluation["threshold_grid_start"]),
        float(evaluation["threshold_grid_stop"])
        + float(evaluation["threshold_grid_step"]) / 2,
        float(evaluation["threshold_grid_step"]),
    )
    thresholds = select_thresholds(
        label_matrix(calibration["labels"]), calibration_prob, grid
    )
    metric_kwargs = {
        "n_bootstrap": int(evaluation["bootstrap_replicates"]),
        "seed": int(manifest["trial"]["seed"]),
        "n_trials": n_trials,
    }
    test_labels = label_matrix(test["labels"])
    selection_performance = classification_report(
        label_matrix(selection["labels"]),
        selection_prob,
        thresholds,
        **metric_kwargs,
    )
    selection_labels = label_matrix(selection["labels"])
    selection_calibration = calibration_report(
        selection_labels,
        selection_prob,
        n_bins=int(evaluation["calibration_bins"]),
        **metric_kwargs,
    )
    selection_fairness = fairness_report(
        selection_labels,
        selection_prob,
        adjacent_prob,
        adjacent["slice"].to_numpy(),
        thresholds,
        **metric_kwargs,
    )
    jigsaw_performance = classification_report(
        test_labels, test_prob, thresholds, **metric_kwargs
    )
    jigsaw_fairness = fairness_report(
        test_labels,
        test_prob,
        adjacent_prob,
        adjacent["slice"].to_numpy(),
        thresholds,
        **metric_kwargs,
    )
    jigsaw_calibration = calibration_report(
        test_labels,
        test_prob,
        n_bins=int(evaluation["calibration_bins"]),
        **metric_kwargs,
    )

    beaver = frames["beavertails_test"]
    beaver_labels = label_matrix(
        [np.asarray(value, dtype=np.float32) for value in beaver["labels"]]
    )
    beaver_prob = predict(model, tokenizer, beaver["text"], **inference)
    beaver_performance = classification_report(
        beaver_labels, beaver_prob, thresholds, **metric_kwargs
    )

    toxigen = pd.concat(
        [frames["toxigen_train"], frames["toxigen_test"]], ignore_index=True
    )
    toxigen_prob = predict(model, tokenizer, toxigen["text"], **inference)[:, 0]
    toxigen_slices = (
        toxigen["source_split"].astype(str) + ":" + toxigen["target_group"].astype(str)
    ).to_numpy()
    toxigen_report = binary_diagnostic_report(
        toxigen["gold"].to_numpy(),
        toxigen_prob,
        thresholds[0],
        toxigen_slices,
        **metric_kwargs,
    )

    hatecheck = frames["hatecheck"]
    hatecheck_prob = predict(model, tokenizer, hatecheck["text"], **inference)[:, 0]
    hatecheck_report = binary_diagnostic_report(
        hatecheck["gold"].to_numpy(),
        hatecheck_prob,
        thresholds[0],
        hatecheck["functionality"].astype(str).to_numpy(),
        **metric_kwargs,
    )

    civil = frames["civil_comments"]
    civil_prob = predict(model, tokenizer, civil["text"], **inference)
    civil_labels = civil[list(TARGET_LABELS)].to_numpy(dtype=float) >= float(
        evaluation["civil_label_cutoff"]
    )
    memberships = {
        name: civil[name].to_numpy(dtype=float)
        >= float(evaluation["identity_membership_cutoff"])
        for name in IDENTITY_COLUMNS
    }
    civil_fairness = identity_fairness_report(
        civil_labels, civil_prob, thresholds, memberships, **metric_kwargs
    )

    payload = {
        "family": ledger["family"],
        "trial_id": args.trial_id,
        "n_trials_completed": n_trials,
        "n_trials_planned": int(ledger["planned_n_trials"]),
        "completed_trial_ids": completed,
        "manifest": manifest,
        "thresholds": dict(zip(TARGET_LABELS, thresholds.tolist())),
        "selection_performance": selection_performance,
        "selection_calibration": selection_calibration,
        "selection_fairness": selection_fairness,
        "jigsaw_performance": jigsaw_performance,
        "jigsaw_calibration": jigsaw_calibration,
        "jigsaw_threshold_sensitivity": threshold_sensitivity(
            test_labels, test_prob, grid
        ),
        "jigsaw_fairness": jigsaw_fairness,
        "beavertails_external": beaver_performance,
        "toxigen_external": toxigen_report,
        "hatecheck_external": hatecheck_report,
        "civil_comments_identity_fairness": civil_fairness,
    }
    output = Path(args.output_root) / args.trial_id
    output.mkdir(parents=True, exist_ok=True)
    write_text_artifact(output / "metrics.json", json.dumps(payload, indent=2) + "\n")
    write_text_artifact(output / "report.md", compact_markdown(payload))
    np.savez_compressed(
        output / "predictions.npz",
        jigsaw_test_ids=test["id"].to_numpy(),
        jigsaw_test_probabilities=test_prob,
        model_selection_ids=selection["id"].to_numpy(),
        model_selection_probabilities=selection_prob,
        beavertails_ids=beaver["id"].to_numpy(),
        beavertails_probabilities=beaver_prob,
        toxigen_ids=toxigen["id"].to_numpy(),
        toxigen_probabilities=toxigen_prob,
        hatecheck_ids=hatecheck["id"].to_numpy(),
        hatecheck_probabilities=hatecheck_prob,
        civil_ids=civil["id"].to_numpy(),
        civil_probabilities=civil_prob,
        thresholds=thresholds,
    )
    print(f"Wrote v2 evaluation: {output}")


if __name__ == "__main__":
    main()
