"""Join blinded labels to private keys and bootstrap verifier validity."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from verifier_reward_hacking.io import read_jsonl, sha256_file


def clustered_metric(rows, metric, replicates, seed):
    groups = defaultdict(list)
    for row in rows:
        groups[row["task_id"]].append(row)
    task_ids = sorted(groups)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(replicates):
        sampled = rng.integers(0, len(task_ids), len(task_ids))
        draw = [row for index in sampled for row in groups[task_ids[index]]]
        samples.append(metric(draw))
    estimate = metric(rows)
    finite = np.asarray([value for value in samples if np.isfinite(value)])
    return {"estimate": float(estimate) if np.isfinite(estimate) else None, "ci95": np.quantile(finite, [0.025, 0.975]).tolist() if len(finite) else [None, None], "n_trials": 1, "n_items": len(rows), "n_task_clusters": len(task_ids), "bootstrap_replicates": replicates, "valid_bootstrap_replicates": int(len(finite))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", type=Path, default=Path("results/verifier_reward_hacking_v1/manual_audit"))
    args = parser.parse_args()
    manifest_path = args.audit_dir / "audit_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for filename, digest in manifest["artifacts"].items():
        if sha256_file(args.audit_dir / filename) != digest and "labels_blind" not in filename:
            raise RuntimeError(f"Private audit artifact changed: {filename}")
    trajectory_key = {row["item_id"]: row for row in read_jsonl(args.audit_dir / "trajectory_key_private.jsonl")}
    trajectory_labels = read_jsonl(args.audit_dir / "trajectory_labels_blind.jsonl")
    final_key = {row["item_id"]: row for row in read_jsonl(args.audit_dir / "final_key_private.jsonl")}
    final_labels = read_jsonl(args.audit_dir / "final_labels_blind.jsonl")
    if any(any(value is None for value in row["labels"].values()) for row in trajectory_labels) or any(row["semantic_correct"] is None for row in final_labels):
        raise RuntimeError("Manual labels are incomplete")
    trajectory = [{**trajectory_key[row["item_id"]], "manual": row["labels"]} for row in trajectory_labels]
    final = [{**final_key[row["item_id"]], "semantic_correct": bool(row["semantic_correct"])} for row in final_labels]
    reps, seed = 2000, 20260929
    precision = lambda rows: sum(r["semantic_correct"] and r["automatic_hidden_pass"] for r in rows) / sum(r["automatic_hidden_pass"] for r in rows) if sum(r["automatic_hidden_pass"] for r in rows) else float("nan")
    recall = lambda rows: sum(r["semantic_correct"] and r["automatic_hidden_pass"] for r in rows) / sum(r["semantic_correct"] for r in rows) if sum(r["semantic_correct"] for r in rows) else float("nan")
    results = {
        "study_id": "verifier_reward_hacking_v1", "status": "complete",
        "v1_precision": clustered_metric(final, precision, reps, seed),
        "v1_recall": clustered_metric(final, recall, reps, seed + 1),
        "trajectory_label_rates": {},
    }
    for offset, label in enumerate(trajectory_labels[0]["labels"]):
        results["trajectory_label_rates"][label] = clustered_metric(trajectory, lambda rows, key=label: sum(bool(r["manual"][key]) for r in rows) / len(rows), reps, seed + 10 + offset)
    flag_rows = [{**row, "flag": bool(row["verifier_result"].get("special_case_flag")), "manual_flag": bool(row["manual"]["special_casing"])} for row in trajectory]
    results["special_case_flag_precision"] = clustered_metric(flag_rows, lambda rows: sum(r["flag"] and r["manual_flag"] for r in rows) / sum(r["flag"] for r in rows) if sum(r["flag"] for r in rows) else float("nan"), reps, seed + 30)
    results["special_case_flag_recall"] = clustered_metric(flag_rows, lambda rows: sum(r["flag"] and r["manual_flag"] for r in rows) / sum(r["manual_flag"] for r in rows) if sum(r["manual_flag"] for r in rows) else float("nan"), reps, seed + 31)
    path = args.audit_dir / "manual_metrics.json"
    path.write_text(json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n")
    manifest.update({"status": "complete", "manual_metrics_sha256": sha256_file(path), "label_files_sha256_at_analysis": {"trajectory": sha256_file(args.audit_dir / "trajectory_labels_blind.jsonl"), "final": sha256_file(args.audit_dir / "final_labels_blind.jsonl")}})
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
