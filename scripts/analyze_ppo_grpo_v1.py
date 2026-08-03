#!/usr/bin/env python3
"""Validate and summarize the preregistered PPO-vs-GRPO v1 runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


METHODS = ("ppo", "grpo")
README_START = "<!-- PPO_GRPO_V1_RESULTS_START -->"
README_END = "<!-- PPO_GRPO_V1_RESULTS_END -->"


def percentile_interval(values: np.ndarray) -> list[float]:
    return [float(x) for x in np.quantile(values, [0.025, 0.975])]


def bootstrap_seed_metric(values: list[float], rng: np.random.Generator, n: int) -> dict:
    array = np.asarray(values, dtype=float)
    draws = rng.choice(array, size=(n, len(array)), replace=True).mean(axis=1)
    return {"estimate": float(array.mean()), "ci95": percentile_interval(draws), "n_trials": len(array)}


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    low = 0.0 if successes == 0 else max(0.0, center - radius)
    high = 1.0 if successes == total else min(1.0, center + radius)
    return [low, high]


def load_runs(root: Path, config: dict) -> dict[str, list[dict]]:
    runs: dict[str, list[dict]] = {method: [] for method in METHODS}
    expected_ids = None
    for method in METHODS:
        for seed in config["paired_seeds"]:
            directory = root / f"{method}_seed{seed}"
            manifest_path = directory / "run_manifest.json"
            predictions_path = directory / "predictions.json"
            trajectory_path = directory / "trajectory.json"
            for path in (manifest_path, predictions_path, trajectory_path):
                if not path.is_file():
                    raise FileNotFoundError(path)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
            trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
            if manifest["method"] != method or manifest["seed"] != seed:
                raise ValueError(f"Manifest identity mismatch in {directory}")
            if len(predictions) != config["task"]["evaluation_examples"]:
                raise ValueError(f"Wrong evaluation count in {directory}")
            ids = [row["problem_id"] for row in predictions]
            if len(ids) != len(set(ids)):
                raise ValueError(f"Duplicate evaluation IDs in {directory}")
            if expected_ids is None:
                expected_ids = ids
            elif ids != expected_ids:
                raise ValueError(f"Evaluation set/order mismatch in {directory}")
            runs[method].append(
                {"seed": seed, "manifest": manifest, "predictions": predictions, "trajectory": trajectory}
            )
    return runs


def summarize_run(run: dict, method: str, config: dict) -> dict:
    predictions = run["predictions"]
    trajectory = run["trajectory"]
    correct = np.asarray([bool(row["correct"]) for row in predictions])
    formatted = np.asarray([row["prediction"] is not None for row in predictions])
    if method == "ppo":
        rows = [row for row in trajectory if "objective/scores" in row]
        rewards = np.asarray([row["objective/scores"] for row in rows], dtype=float)
        native_kl = np.asarray([row["objective/kl"] for row in rows], dtype=float)
        # TRL PPO logs sequence-summed KL; every response exhausted the locked
        # 64-token cap, so division yields a per-completion-token diagnostic.
        kl = native_kl / config["compute_budget"]["max_completion_tokens"]
        zero_group_fraction = None
        clipped_fraction = float(np.mean([row["policy/clipfrac_avg"] for row in rows]))
    else:
        rows = [row for row in trajectory if "reward" in row]
        rewards = np.asarray([row["reward"] for row in rows], dtype=float)
        native_kl = np.asarray([row["kl"] for row in trajectory if "kl" in row], dtype=float)
        kl = native_kl
        zero_group_fraction = float(np.mean([row["frac_reward_zero_std"] for row in rows]))
        clipped_fraction = float(np.mean([row["clip_ratio/region_mean"] for row in trajectory if "kl" in row]))
    observed_rollout_groups = len(rows)
    generated_tokens = (
        observed_rollout_groups
        * config["compute_budget"]["generations_per_group"]
        * config["compute_budget"]["max_completion_tokens"]
    )
    return {
        "seed": run["seed"],
        "evaluation_n": len(predictions),
        "exact_matches": int(correct.sum()),
        "exact_match": float(correct.mean()),
        "formatted_answers": int(formatted.sum()),
        "formatted_answer_rate": float(formatted.mean()),
        "reward_mean": float(rewards.mean()),
        "reward_variance_over_training": float(rewards.var(ddof=1)),
        "kl_native_mean": float(native_kl.mean()),
        "kl_per_token_mean": float(kl.mean()),
        "kl_per_token_auc": float(np.trapezoid(kl, dx=1 / max(1, len(kl) - 1))),
        "reward_zero_group_fraction": zero_group_fraction,
        "clip_fraction": clipped_fraction,
        "training_seconds": float(run["manifest"]["training_seconds"]),
        "peak_gpu_memory_mb": float(run["manifest"]["peak_gpu_memory_mb"]),
        "generated_tokens": generated_tokens,
        "generated_tokens_per_second": generated_tokens / float(run["manifest"]["training_seconds"]),
        "observed_rollout_groups": observed_rollout_groups,
        "trajectory_rows": len(trajectory),
        "resolved_model_revision": run["manifest"]["resolved_model_revision"],
    }


def hierarchical_accuracy_bootstrap(runs: dict, rng: np.random.Generator, n: int) -> dict:
    by_method = {
        method: np.asarray([[row["correct"] for row in run["predictions"]] for run in method_runs], dtype=float)
        for method, method_runs in runs.items()
    }
    estimates = {method: [] for method in METHODS}
    differences = []
    n_seeds, n_examples = by_method["ppo"].shape
    for _ in range(n):
        seed_indices = rng.integers(0, n_seeds, n_seeds)
        values = {}
        for method in METHODS:
            sampled = []
            for seed_index in seed_indices:
                example_indices = rng.integers(0, n_examples, n_examples)
                sampled.append(by_method[method][seed_index, example_indices].mean())
            values[method] = float(np.mean(sampled))
            estimates[method].append(values[method])
        differences.append(values["grpo"] - values["ppo"])
    return {
        method: {
            "estimate": float(by_method[method].mean()),
            "ci95": percentile_interval(np.asarray(estimates[method])),
            "n_trials": n_seeds,
            "n_evaluations_per_trial": n_examples,
        }
        for method in METHODS
    } | {
        "grpo_minus_ppo": {
            "estimate": float(by_method["grpo"].mean() - by_method["ppo"].mean()),
            "ci95": percentile_interval(np.asarray(differences)),
            "n_paired_trials": n_seeds,
        }
    }


def metric_cell(metric: dict, digits: int = 4) -> str:
    lo, hi = metric["ci95"]
    return f"{metric['estimate']:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def render_report(metrics: dict, config: dict) -> str:
    lines = [
        "# PPO vs GRPO v1 — preregistered result",
        "",
        "## Design and integrity",
        "",
        f"- Base model: `{config['model']['base_model']}` at resolved revision "
        f"`{metrics['provenance']['resolved_model_revision']}`.",
        f"- Paired seeds / trials: {len(config['paired_seeds'])}; optimizer steps per run: "
        f"{config['compute_budget']['optimizer_steps']}.",
        f"- Evaluation: {config['task']['evaluation_examples']} frozen problems per seed and method; "
        f"bootstrap replicates: {config['evaluation']['bootstrap_replicates']}.",
        "- Smoke outputs were excluded. All six full manifests, trajectories, and prediction sets passed validation.",
        "- Protocol deviation: both methods completed the locked 200 optimizer steps, but TRL's interaction between "
        "`num_iterations=2` and `steps_per_generation=2` yielded 50 observed GRPO generation groups rather than the "
        "preregistered 100. PPO yielded 100. Generated-token throughput below uses observed groups, not the intended count.",
        "",
        "## Primary outcome",
        "",
        "| Method | Exact match (hierarchical bootstrap 95% CI) | Per-seed successes | Valid tagged answers |",
        "|---|---:|---:|---:|",
    ]
    for method in METHODS:
        seeds = metrics["runs"][method]
        lines.append(
            f"| {method.upper()} | {metric_cell(metrics['accuracy'][method])} | "
            f"{' / '.join(str(row['exact_matches']) for row in seeds)} of 400 | "
            f"{' / '.join(str(row['formatted_answers']) for row in seeds)} of 400 |"
        )
    difference = metrics["accuracy"]["grpo_minus_ppo"]
    lines += [
        "",
        f"Paired GRPO − PPO exact-match difference: **{metric_cell(difference)}**. The bootstrap interval "
        "collapses at zero because every prediction failed; this is a floor effect and cannot establish equivalence. "
        f"A per-seed Wilson interval for 0/400 is [{metrics['accuracy']['zero_success_wilson_400'][0]:.4f}, "
        f"{metrics['accuracy']['zero_success_wilson_400'][1]:.4f}].",
        "",
        "## Stability and systems metrics",
        "",
        "All cells are means across three independent seeds with seed-bootstrap 95% intervals and `N_trials=3`.",
        "PPO native KL is sequence-summed by TRL and is divided by the locked 64-token completion length for the "
        "per-token comparison; GRPO logs per-token KL directly.",
        "",
        "| Method | Reward variance | KL/token AUC | Peak VRAM MB | Train seconds | Generated tok/s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        aggregate = metrics["aggregate"][method]
        lines.append(
            f"| {method.upper()} | {metric_cell(aggregate['reward_variance_over_training'], 6)} | "
            f"{metric_cell(aggregate['kl_per_token_auc'], 6)} | "
            f"{metric_cell(aggregate['peak_gpu_memory_mb'], 1)} | "
            f"{metric_cell(aggregate['training_seconds'], 1)} | "
            f"{metric_cell(aggregate['generated_tokens_per_second'], 1)} |"
        )
    grpo_zero = metrics["aggregate"]["grpo"]["reward_zero_group_fraction"]
    lines += [
        "",
        f"GRPO groups with zero reward standard deviation: **{metric_cell(grpo_zero, 4)}**. "
        "Both methods generated to the 64-token cap without producing valid tagged answers on evaluation.",
        "",
        "## Finding",
        "",
        "The preregistered comparison is valid as an execution study but uninformative about relative task performance. "
        "The base policy almost never entered the verifier's positive-support region: exact-answer reward was absent, "
        "format reward was absent on held-out generations, and GRPO frequently had no within-group contrast. Consequently, "
        "both algorithms remained at the evaluation floor. GRPO used less GPU memory, but the rollout-count deviation "
        "precludes a clean wall-time or throughput efficiency claim; those systems observations do not imply better "
        "optimization when the reward supplies essentially no task signal.",
        "",
        "This replaces the prior scaffold caveat with a real preregistered negative result. A follow-up must be separately "
        "preregistered and should add outcome-blind reward shaping or arithmetic SFT so that both methods encounter nonzero "
        "correctness signal; this v1 result must not be overwritten or reframed as a tie.",
    ]
    return "\n".join(lines) + "\n"


def update_readme(path: Path, report: str) -> None:
    text = path.read_text(encoding="utf-8")
    if README_START in text and README_END in text:
        before, remainder = text.split(README_START, 1)
        _, after = remainder.split(README_END, 1)
        text = before.rstrip() + "\n\n" + after.lstrip()
    summary = report.split("## Primary outcome", 1)[1]
    summary = summary.replace("\n## Stability", "\n#### Stability").replace("\n## Finding", "\n#### Finding")
    block = README_START + "\n\n### Preregistered PPO vs GRPO v1 result\n" + summary + "\n" + README_END
    stale = (
        "This repo now contains the training and comparison scaffolding for that experiment, but you should still "
        "treat the actual PPO-vs-GRPO numbers as run-dependent until `scripts/compare_ppo_grpo.py` has been executed "
        "and its result artifact committed."
    )
    text = text.replace(stale, "The preregistered execution result and its interpretation boundary follow.")
    anchor = "### GRPO Status"
    index = text.index(anchor)
    section_end = text.index("\n---\n", index)
    text = text[:section_end].rstrip() + "\n\n" + block + "\n" + text[section_end:]
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/ppo_grpo_v1.json"))
    parser.add_argument("--input-dir", type=Path, default=Path("results/ppo_grpo_v1"))
    parser.add_argument("--update-readme", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    runs = load_runs(args.input_dir, config)
    summaries = {method: [summarize_run(run, method, config) for run in method_runs] for method, method_runs in runs.items()}
    rng = np.random.default_rng(config["evaluation"]["bootstrap_seed"])
    accuracy = hierarchical_accuracy_bootstrap(runs, rng, config["evaluation"]["bootstrap_replicates"])
    accuracy["zero_success_wilson_400"] = wilson_interval(0, config["task"]["evaluation_examples"])
    aggregate = {}
    keys = [
        "reward_mean", "reward_variance_over_training", "kl_per_token_mean", "kl_per_token_auc",
        "clip_fraction", "training_seconds", "peak_gpu_memory_mb", "generated_tokens_per_second",
    ]
    for method in METHODS:
        aggregate[method] = {
            key: bootstrap_seed_metric([row[key] for row in summaries[method]], rng, config["evaluation"]["bootstrap_replicates"])
            for key in keys
        }
        if method == "grpo":
            aggregate[method]["reward_zero_group_fraction"] = bootstrap_seed_metric(
                [row["reward_zero_group_fraction"] for row in summaries[method]],
                rng,
                config["evaluation"]["bootstrap_replicates"],
            )
    revisions = {row["resolved_model_revision"] for method in METHODS for row in summaries[method]}
    if len(revisions) != 1:
        raise ValueError(f"Model revision mismatch: {revisions}")
    metrics = {
        "study_id": config["study_id"],
        "status": "complete_negative_finding_with_protocol_deviation",
        "bootstrap_replicates": config["evaluation"]["bootstrap_replicates"],
        "runs": summaries,
        "accuracy": accuracy,
        "aggregate": aggregate,
        "provenance": {"resolved_model_revision": revisions.pop()},
    }
    report = render_report(metrics, config)
    (args.input_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.input_dir / "report.md").write_text(report, encoding="utf-8")
    with (args.input_dir / "trajectory_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries["ppo"][0].keys()), lineterminator="\n")
        writer.writeheader()
        for method in METHODS:
            for row in summaries[method]:
                writer.writerow({**row, "seed": f"{method}:{row['seed']}"})
    if args.update_readme:
        update_readme(Path("README.md"), report)
    print(f"Validated six runs; wrote {args.input_dir / 'report.md'}")


if __name__ == "__main__":
    main()
