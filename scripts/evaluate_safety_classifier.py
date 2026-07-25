#!/usr/bin/env python3
"""Evaluate category performance and adjacent-benign false positives."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from peft import AutoPeftModelForSequenceClassification
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.safety.data import (
    EncodedTextDataset,
    MultiLabelDataCollator,
    label_matrix,
    load_adjacent_benign,
    load_jigsaw_csv,
)
from src.safety.metrics import classification_report, fairness_report, select_thresholds
from src.safety.taxonomy import TARGET_LABELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/safety_classifier.yaml")
    parser.add_argument("--jigsaw-csv", default="data/raw/jigsaw/train.csv")
    parser.add_argument("--model-dir", default="checkpoints/safety_classifier_v1")
    parser.add_argument("--adjacent-benign", default="data/adjacent_benign_v1.csv")
    parser.add_argument("--output-dir", default="results/safety_classifier_v1")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def predict(model, tokenizer, texts, max_length, batch_size, device) -> np.ndarray:
    labels = np.zeros((len(texts), len(TARGET_LABELS)), dtype=np.float32)
    dataset = EncodedTextDataset(list(texts), labels, tokenizer, max_length)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=MultiLabelDataCollator(tokenizer),
    )
    batches = []
    model.eval()
    for batch in loader:
        batch.pop("labels")
        batch = {name: tensor.to(device) for name, tensor in batch.items()}
        batches.append(torch.sigmoid(model(**batch).logits).cpu().numpy())
    return np.concatenate(batches)


def metric_cell(metric: dict) -> str:
    low, high = metric["ci95"]
    return f'{metric["value"]:.3f} [{low:.3f}, {high:.3f}]'


def markdown_report(performance: dict, fairness: dict, manifest: dict) -> str:
    lines = [
        "# Safety classifier evaluation",
        "",
        f'- Experiment: `{manifest["experiment_name"]}`',
        f'- Full run: `{manifest["full_run"]}`',
        f'- N_trials: {performance["n_trials"]}',
        f'- Bootstrap replicates: {performance["n_bootstrap"]}',
        f'- Test examples: {performance["n_examples"]}',
        "",
        "All intervals are row-bootstrap 95% confidence intervals. Thresholds were",
        "selected once on the calibration split and then frozen for both evaluations.",
        "",
        "| Category | Support | Precision (95% CI) | Recall (95% CI) | F1 (95% CI) | Threshold |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in TARGET_LABELS:
        item = performance["per_category"][name]
        lines.append(
            f'| {name} | {item["support"]} | {metric_cell(item["precision"])} | '
            f'{metric_cell(item["recall"])} | {metric_cell(item["f1"])} | '
            f'{performance["thresholds"][name]:.2f} |'
        )
    lines.extend(
        [
            "",
            "## False-positive fairness stress test",
            "",
            "| Population | N | Any-label FPR (95% CI) |",
            "|---|---:|---:|",
            f'| Jigsaw all-negative test rows | {fairness["overall_test_fpr"]["n_examples"]} | '
            f'{metric_cell(fairness["overall_test_fpr"])} |',
            f'| Adjacent-benign stress set | {fairness["adjacent_benign_fpr"]["n_examples"]} | '
            f'{metric_cell(fairness["adjacent_benign_fpr"])} |',
        ]
    )
    for name, item in fairness["adjacent_benign_by_slice"].items():
        lines.append(f"| ↳ {name} | {item['n_examples']} | {metric_cell(item)} |")
    lines.extend(
        [
            "",
            f'Adjacent-benign FPR gap: {metric_cell(fairness["adjacent_benign_fpr_gap"])}.',
            "",
            "Interpretation: this curated set is a targeted failure-mode stress test, not a",
            "representative demographic sample. A positive gap means the classifier flags",
            "contextually benign adjacent-domain text more often than ordinary negative",
            "Jigsaw comments. No causal or population-fairness claim follows from it.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    evaluation = config["evaluation"]
    model_dir = Path(args.model_dir)
    manifest = json.loads((model_dir / "run_manifest.json").read_text())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_jigsaw_csv(args.jigsaw_csv, int(config["seed"]))
    calibration = frame[frame["split"] == "calibration"]
    test = frame[frame["split"] == "test"]
    benign = load_adjacent_benign(args.adjacent_benign)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoPeftModelForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model.to(device)
    kwargs = {
        "max_length": int(config["max_length"]),
        "batch_size": int(config["train"]["eval_batch_size"]),
        "device": device,
    }
    calibration_prob = predict(model, tokenizer, calibration["comment_text"], **kwargs)
    test_prob = predict(model, tokenizer, test["comment_text"], **kwargs)
    benign_prob = predict(model, tokenizer, benign["text"], **kwargs)

    start = float(evaluation["threshold_grid_start"])
    stop = float(evaluation["threshold_grid_stop"])
    step = float(evaluation["threshold_grid_step"])
    thresholds = select_thresholds(
        label_matrix(calibration["labels"]),
        calibration_prob,
        np.arange(start, stop + step / 2, step),
    )
    metric_kwargs = {
        "n_bootstrap": int(evaluation["bootstrap_replicates"]),
        "seed": int(config["seed"]),
        "n_trials": int(evaluation["n_trials"]),
    }
    performance = classification_report(
        label_matrix(test["labels"]), test_prob, thresholds, **metric_kwargs
    )
    fairness = fairness_report(
        label_matrix(test["labels"]),
        test_prob,
        benign_prob,
        benign["slice"].to_numpy(),
        thresholds,
        **metric_kwargs,
    )
    payload = {"manifest": manifest, "performance": performance, "fairness": fairness}
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    (output_dir / "report.md").write_text(markdown_report(performance, fairness, manifest))
    np.savez_compressed(
        output_dir / "predictions.npz",
        calibration_ids=calibration["id"].to_numpy(),
        calibration_probabilities=calibration_prob,
        test_ids=test["id"].to_numpy(),
        test_probabilities=test_prob,
        benign_ids=benign["example_id"].to_numpy(),
        benign_probabilities=benign_prob,
        thresholds=thresholds,
    )
    print(f"Wrote {output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
