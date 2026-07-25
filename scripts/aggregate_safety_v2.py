#!/usr/bin/env python3
"""Aggregate all 12 disclosed v2 trials and apply the locked selection rule."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_safety_classifier import write_text_artifact  # noqa: E402
from src.safety.data import label_matrix, load_jigsaw_csv  # noqa: E402
from src.safety.taxonomy import TARGET_LABELS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", default="configs/safety_v2_trial_ledger.json")
    parser.add_argument("--results-root", default="results/safety_v2")
    parser.add_argument("--jigsaw-csv", default="data/raw/jigsaw/train.csv")
    parser.add_argument(
        "--v1-predictions", default="results/safety_classifier_v1/predictions.npz"
    )
    parser.add_argument("--output-dir", default="results/safety_v2/aggregate")
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def trial_key(payload: dict) -> tuple:
    performance = payload["selection_performance"]
    macro = performance["macro_f1"]["value"]
    violent = performance["per_category"]["harmful_violent"]["f1"]["value"]
    negative_fpr = payload["selection_fairness"]["overall_test_fpr"]["value"]
    ece = np.mean(
        [
            payload["selection_calibration"]["per_category"][name]["ece"]["value"]
            for name in TARGET_LABELS
        ]
    )
    return (macro, violent, -negative_fpr, -ece)


def macro_f1(truth: np.ndarray, probability: np.ndarray, thresholds: np.ndarray) -> float:
    from sklearn.metrics import f1_score

    return float(
        np.mean(
            [
                f1_score(truth[:, col], probability[:, col] >= thresholds[col], zero_division=0)
                for col in range(truth.shape[1])
            ]
        )
    )


def paired_improvement(
    truth: np.ndarray,
    v1_probability: np.ndarray,
    v1_thresholds: np.ndarray,
    v2_probability: np.ndarray,
    v2_thresholds: np.ndarray,
    seed: int = 2025,
    n_bootstrap: int = 2000,
) -> dict:
    point = macro_f1(truth, v2_probability, v2_thresholds) - macro_f1(
        truth, v1_probability, v1_thresholds
    )
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(truth), len(truth))
        estimates.append(
            macro_f1(truth[indices], v2_probability[indices], v2_thresholds)
            - macro_f1(truth[indices], v1_probability[indices], v1_thresholds)
        )
    return {
        "value": float(point),
        "ci95": [float(x) for x in np.quantile(estimates, [0.025, 0.975])],
        "n_bootstrap": int(n_bootstrap),
        "paired": True,
    }


def main() -> None:
    args = parse_args()
    ledger = json.loads(Path(args.ledger).read_text(encoding="utf-8"))
    root = Path(args.results_root)
    payloads = {}
    missing = []
    for trial in ledger["trials"]:
        path = root / trial["trial_id"] / "metrics.json"
        if path.exists():
            payloads[trial["trial_id"]] = json.loads(path.read_text(encoding="utf-8"))
        else:
            missing.append(trial["trial_id"])
    if missing and not args.allow_incomplete:
        raise ValueError(f"All 12 trials are required before selection; missing: {missing}")

    selected_id = max(payloads, key=lambda name: trial_key(payloads[name]))
    selected = payloads[selected_id]
    by_loss = {}
    for loss in sorted({trial["loss"] for trial in ledger["trials"]}):
        ids = [
            trial["trial_id"]
            for trial in ledger["trials"]
            if trial["loss"] == loss and trial["trial_id"] in payloads
        ]
        macro_values = [
            payloads[trial_id]["selection_performance"]["macro_f1"]["value"]
            for trial_id in ids
        ]
        by_loss[loss] = {
            "trial_ids": ids,
            "n_trials": len(ids),
            "macro_f1_mean": float(np.mean(macro_values)),
            "macro_f1_std_across_seeds": float(np.std(macro_values, ddof=1))
            if len(macro_values) > 1
            else None,
            "macro_f1_min": float(np.min(macro_values)),
            "macro_f1_max": float(np.max(macro_values)),
        }

    jigsaw = load_jigsaw_csv(args.jigsaw_csv, seed=2025)
    test = jigsaw[jigsaw["split"] == "test"]
    truth_by_id = dict(zip(test["id"].astype(str), test["labels"]))
    v1 = np.load(args.v1_predictions, allow_pickle=True)
    v2 = np.load(root / selected_id / "predictions.npz", allow_pickle=True)
    v1_ids = [str(value) for value in v1["test_ids"]]
    v2_ids = [str(value) for value in v2["jigsaw_test_ids"]]
    if v1_ids != v2_ids:
        raise ValueError("V1 and selected v2 Jigsaw test IDs are not identically ordered")
    truth = label_matrix([truth_by_id[value] for value in v1_ids])
    improvement = paired_improvement(
        truth,
        v1["test_probabilities"],
        v1["thresholds"],
        v2["jigsaw_test_probabilities"],
        v2["thresholds"],
    )

    aggregate = {
        "family": ledger["family"],
        "n_trials": len(payloads),
        "n_trials_planned": int(ledger["planned_n_trials"]),
        "missing_trials": missing,
        "selected_trial_id": selected_id,
        "selection_key": {
            "model_selection_macro_f1": trial_key(selected)[0],
            "model_selection_harmful_violent_f1": trial_key(selected)[1],
            "negative_model_selection_fpr": trial_key(selected)[2],
            "negative_model_selection_mean_ece": trial_key(selected)[3],
        },
        "paired_macro_f1_improvement_vs_v1": improvement,
        "by_loss": by_loss,
        "all_trial_selection_keys": {
            name: list(trial_key(payload)) for name, payload in payloads.items()
        },
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_text_artifact(output / "aggregate_metrics.json", json.dumps(aggregate, indent=2) + "\n")
    lines = [
        "# Safety v2 aggregate",
        "",
        f'- N_trials: {aggregate["n_trials"]}/{aggregate["n_trials_planned"]}',
        f'- Selected trial: `{selected_id}`',
        f'- Paired macro-F1 improvement vs v1: '
        f'{improvement["value"]:.3f} [{improvement["ci95"][0]:.3f}, '
        f'{improvement["ci95"][1]:.3f}]',
        "",
        "Selection is valid only when N_trials is 12 and no trial is missing.",
        "",
    ]
    write_text_artifact(output / "aggregate_report.md", "\n".join(lines))
    print(f"Wrote aggregate results: {output}")


if __name__ == "__main__":
    main()
