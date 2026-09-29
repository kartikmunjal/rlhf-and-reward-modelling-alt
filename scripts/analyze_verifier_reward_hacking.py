"""Generate all confirmatory Extension 16 statistics from raw ledgers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from verifier_reward_hacking.io import read_jsonl, sha256_file
from verifier_reward_hacking.statistics import hierarchical_bootstrap_mean, hierarchical_paired_bootstrap, holm_adjust, paired_task_bootstrap, spearman_task_bootstrap, task_bootstrap_mean


METRICS = {"hidden_pass_at_1": lambda row: row["hidden"]["passed"], "exploit_rate": lambda row: row["exploit"], "visible_hidden_gap": lambda row: row["gap"], "visible_pass_at_1": lambda row: row["visible"]["passed"]}
SEEDS = (2025, 2026, 2027)
CONDITIONS = ("C1", "C2", "C3", "C4")


def load_eval(path: Path) -> dict[str, dict]:
    manifest = json.loads((path / "evaluation_manifest.json").read_text())
    ledger = path / "predictions_private.jsonl"
    if manifest["status"] != "complete" or sha256_file(ledger) != manifest["predictions_sha256"]:
        raise RuntimeError(f"Invalid evaluation artifact: {path}")
    rows = read_jsonl(ledger)
    return {row["task_id"]: row for row in rows}


def subset(rows: dict[str, dict], slice_name: str) -> dict[str, dict]:
    if slice_name == "generated":
        return {key: row for key, row in rows.items() if row["source"] == "programmatic_v1"}
    if slice_name == "public":
        return {key: row for key, row in rows.items() if row["source"] == "openai_humaneval_mit"}
    return rows


def array(rows, ids, metric):
    return np.array([float(METRICS[metric](rows[key])) for key in ids])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluations-root", type=Path, required=True)
    parser.add_argument("--public-index", type=Path, default=Path("data/processed/verifier_reward_hacking_v1/final/public_index.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/verifier_reward_hacking_v1"))
    args = parser.parse_args()
    study = json.loads((ROOT / "verifier_reward_hacking/study_config.json").read_text())
    reps, seed0 = study["evaluation"]["bootstrap_replicates"], study["evaluation"]["bootstrap_seed"]
    evaluations = {"C0": load_eval(args.evaluations_root / "C0" / "all_test")}
    for condition in CONDITIONS:
        for seed in SEEDS:
            evaluations[f"{condition}_{seed}"] = load_eval(args.evaluations_root / condition / f"seed{seed}" / "all_test")
    metrics = {"study_id": study["study_id"], "bootstrap_replicates": reps, "slices": {}, "contrasts": {}, "hypotheses": {}}
    for slice_index, slice_name in enumerate(("generated", "public", "pooled")):
        base = subset(evaluations["C0"], slice_name)
        ids = sorted(base)
        if not ids:
            raise RuntimeError(f"Empty {slice_name} slice")
        slice_metrics = {"task_count": len(ids), "conditions": {"C0": {}}}
        for metric_index, metric in enumerate(METRICS):
            slice_metrics["conditions"]["C0"][metric] = task_bootstrap_mean(array(base, ids, metric), reps, seed0 + slice_index * 100 + metric_index)
        for condition_index, condition in enumerate(CONDITIONS):
            slice_metrics["conditions"][condition] = {}
            condition_rows = [subset(evaluations[f"{condition}_{seed}"], slice_name) for seed in SEEDS]
            if any(set(rows) != set(ids) for rows in condition_rows):
                raise RuntimeError(f"Task mismatch for {slice_name}/{condition}")
            for metric_index, metric in enumerate(METRICS):
                values = np.stack([array(rows, ids, metric) for rows in condition_rows])
                result = hierarchical_bootstrap_mean(values, reps, seed0 + 1000 + slice_index * 100 + condition_index * 10 + metric_index)
                result["per_seed"] = {str(seed): float(values[index].mean()) for index, seed in enumerate(SEEDS)}
                slice_metrics["conditions"][condition][metric] = result
        metrics["slices"][slice_name] = slice_metrics

    # Confirmatory contrasts use the frozen pooled endpoint while all slices remain shown above.
    ids = sorted(evaluations["C0"])
    c0_gap = array(evaluations["C0"], ids, "visible_hidden_gap")
    c1_gap_by_seed = np.stack([array(evaluations[f"C1_{seed}"], ids, "visible_hidden_gap") for seed in SEEDS])
    h1 = paired_task_bootstrap(c1_gap_by_seed.mean(axis=0), c0_gap, reps, seed0 + 5000)
    metrics["contrasts"]["H1_C1_minus_C0_gap"] = h1
    metrics["hypotheses"]["H1_supported"] = h1["ci95"][0] > 0
    for metric_offset, metric in enumerate(("hidden_pass_at_1", "exploit_rate")):
        c1 = np.stack([array(evaluations[f"C1_{seed}"], ids, metric) for seed in SEEDS])
        names, raw_p = [], []
        for condition_offset, condition in enumerate(("C2", "C3", "C4")):
            hardened = np.stack([array(evaluations[f"{condition}_{seed}"], ids, metric) for seed in SEEDS])
            result = hierarchical_paired_bootstrap(hardened, c1, reps, seed0 + 6000 + metric_offset * 100 + condition_offset)
            name = f"{condition}_minus_C1_{metric}"
            metrics["contrasts"][name] = result
            names.append(name); raw_p.append(result["p_two_sided"])
        adjusted = holm_adjust(raw_p)
        for name, value in zip(names, adjusted):
            metrics["contrasts"][name]["p_holm"] = value
    hp = metrics["contrasts"]["C2_minus_C1_hidden_pass_at_1"]
    exploit = metrics["contrasts"]["C2_minus_C1_exploit_rate"]
    metrics["hypotheses"]["H2_primary_supported"] = hp["ci95"][0] > 0 and exploit["ci95"][1] < 0 and hp["p_holm"] < 0.05 and exploit["p_holm"] < 0.05
    public_index = {row["task_id"]: row for row in read_jsonl(args.public_index)}
    generated_ids = sorted(key for key in ids if evaluations["C0"][key]["source"] == "programmatic_v1")
    kill = [public_index[key]["v0_mutation_kill_rate"] for key in generated_ids]
    c1_exploit = np.stack([array(evaluations[f"C1_{seed}"], generated_ids, "exploit_rate") for seed in SEEDS]).mean(axis=0)
    h3 = spearman_task_bootstrap(kill, c1_exploit, reps, seed0 + 7000)
    metrics["contrasts"]["H3_mutation_kill_vs_C1_exploit"] = h3
    metrics["hypotheses"]["H3_supported"] = h3["ci95"][1] < 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    lines = ["# Programmatic-verifier reward-hacking results", "", "Generated by `scripts/analyze_verifier_reward_hacking.py`; no values are hand-edited.", "", "## Honest status", "", "Automated confirmatory analysis complete. Manual trajectory and blinded semantic audits remain separate required gates until their manifests are complete.", "", "## Hypothesis decisions", ""]
    lines.extend(f"- {name}: **{value}**" for name, value in metrics["hypotheses"].items())
    lines += ["", "## Primary contrasts", "", "| Contrast | Estimate | 95% CI | N_trials | N_tasks |", "|---|---:|---:|---:|---:|"]
    for name, row in metrics["contrasts"].items():
        lines.append(f"| {name} | {row['estimate']:.4f} | [{row['ci95'][0]:.4f}, {row['ci95'][1]:.4f}] | {row.get('n_trials', 1)} | {row.get('n_tasks_per_trial', row.get('n_tasks'))} |")
    lines += ["", "## Scope", "", "This is a three-seed, 0.5B-parameter, single-GPU controlled study. Public HumanEval results carry pretraining-contamination risk and are reported separately from generated tasks. The rootless container is an experimental isolation boundary, not a production-security certification. No frontier-model, vendor, or general-GRPO claim follows."]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(metrics_path)


if __name__ == "__main__":
    main()
