#!/usr/bin/env python3
"""Run the locked 12-trial v2 matrix sequentially with resumable status."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", default="configs/safety_v2_trial_ledger.json")
    parser.add_argument("--jigsaw-csv", default="data/raw/jigsaw/train.csv")
    parser.add_argument("--data-dir", default="data/processed/safety_v2")
    parser.add_argument("--checkpoint-root", default="checkpoints/safety_v2")
    parser.add_argument("--results-root", default="results/safety_v2")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def write_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    ledger = json.loads(Path(args.ledger).read_text(encoding="utf-8"))
    status_path = Path(args.results_root) / "matrix_status.json"
    records = {
        trial["trial_id"]: {"trial_id": trial["trial_id"], "status": "pending"}
        for trial in ledger["trials"]
    }
    status = {
        "family": ledger["family"],
        "n_trials_planned": ledger["planned_n_trials"],
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "training",
        "trials": list(records.values()),
    }
    write_status(status_path, status)

    # Train the complete matrix before any final evaluation so every published
    # metric is stamped with the same completed N_trials=12.
    for trial in ledger["trials"]:
        trial_id = trial["trial_id"]
        checkpoint_manifest = Path(args.checkpoint_root) / trial_id / "run_manifest.json"
        record = records[trial_id]
        if args.resume and checkpoint_manifest.exists():
            record["status"] = "trained_existing"
            write_status(status_path, status)
            continue
        train_command = [
            sys.executable,
            "scripts/train_safety_v2.py",
            "--trial-id",
            trial_id,
            "--jigsaw-csv",
            args.jigsaw_csv,
            "--data-dir",
            args.data_dir,
            "--output-root",
            args.checkpoint_root,
        ]
        if args.cpu:
            train_command.append("--cpu")
        try:
            record["status"] = "training"
            write_status(status_path, status)
            subprocess.run(train_command, check=True)
            record["status"] = "trained"
        except subprocess.CalledProcessError as error:
            record["status"] = "training_failed"
            record["returncode"] = error.returncode
            write_status(status_path, status)
            raise
        write_status(status_path, status)

    status["phase"] = "evaluation"
    write_status(status_path, status)
    for trial in ledger["trials"]:
        trial_id = trial["trial_id"]
        record = records[trial_id]
        result_metrics = Path(args.results_root) / trial_id / "metrics.json"
        if args.resume and result_metrics.exists():
            existing = json.loads(result_metrics.read_text(encoding="utf-8"))
            if existing.get("n_trials_completed") == ledger["planned_n_trials"]:
                record["status"] = "complete_existing"
                write_status(status_path, status)
                continue
        evaluate_command = [
            sys.executable,
            "scripts/evaluate_safety_v2.py",
            "--trial-id",
            trial_id,
            "--jigsaw-csv",
            args.jigsaw_csv,
            "--data-dir",
            args.data_dir,
            "--checkpoint-root",
            args.checkpoint_root,
            "--output-root",
            args.results_root,
        ]
        if args.cpu:
            evaluate_command.append("--cpu")
        try:
            record["status"] = "evaluating"
            write_status(status_path, status)
            subprocess.run(evaluate_command, check=True)
            record["status"] = "complete"
        except subprocess.CalledProcessError as error:
            record["status"] = "evaluation_failed"
            record["returncode"] = error.returncode
            write_status(status_path, status)
            raise
        write_status(status_path, status)
    status["phase"] = "complete"
    status["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_status(status_path, status)
    print(f"Completed matrix; status: {status_path}")


if __name__ == "__main__":
    main()
