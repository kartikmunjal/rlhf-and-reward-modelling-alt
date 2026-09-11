"""Generated analysis for checkpoint density, plateau, and self-reliance."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from recursive_self_improvement.curves import select_curve
from recursive_self_improvement.statistics import holm_adjust, paired_bootstrap, paired_sign_flip_test, wilson


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _indexed(rows, checkpoint):
    output = {}
    for row in rows:
        if row["checkpoint"] == checkpoint and row.get("valid", True):
            if row["prompt_id"] in output: raise ValueError("Duplicate checkpoint/prompt evaluation")
            output[row["prompt_id"]] = row
    return output


def _mean_interval(values, config, seed):
    return paired_bootstrap(
        values, np.zeros(len(values)),
        replicates=config["evaluation"]["bootstrap_replicates"], seed=seed,
    ) if values else None


def analyze(config, evaluations, compute_rows, stage3_rows, stage3_training_audit=None):
    checkpoints = config["scope"]["primary_family"]
    capability, valid_sets = {}, {}
    planned = config["data"]["independent_eval_prompts"]
    for index, checkpoint in enumerate(checkpoints):
        indexed = _indexed(evaluations, checkpoint); valid_sets[checkpoint] = indexed
        wins = [row["win_vs_sft"] for row in indexed.values()]
        capability[checkpoint] = {
            "coverage": wilson(len(wins), planned),
            "win_rate": paired_bootstrap(wins, np.zeros(len(wins)), replicates=config["evaluation"]["bootstrap_replicates"], seed=config["evaluation"]["bootstrap_seed"] + index) if wins else None,
        }
    rounds, gains = [], {}
    for round_index in range(1, config["stage1"]["rounds"] + 1):
        name = f"iterative_dpo_round_{round_index}"; rounds.append(name)
        previous = "sft" if round_index == 1 else f"iterative_dpo_round_{round_index - 1}"
        joint = sorted(set(valid_sets[name]) & set(valid_sets[previous]))
        gains[name] = paired_bootstrap(
            [valid_sets[name][key]["win_vs_sft"] for key in joint],
            [valid_sets[previous][key]["win_vs_sft"] for key in joint],
            replicates=config["evaluation"]["bootstrap_replicates"], seed=config["evaluation"]["bootstrap_seed"] + 100 + round_index,
        ) if joint else None
    consecutive, plateau_round = 0, None
    for name in rounds:
        includes_zero = gains[name] is not None and gains[name]["ci95"][0] <= 0 <= gains[name]["ci95"][1]
        consecutive = consecutive + 1 if includes_zero else 0
        if consecutive == 2 and plateau_round is None: plateau_round = int(name.rsplit("_", 1)[1])
    compute = {row["checkpoint"]: row for row in compute_rows}
    complete = [name for name in checkpoints if capability[name]["win_rate"] is not None and name in compute]
    density_curve = select_curve(
        [compute[name]["cumulative_non_padding_training_tokens"] for name in complete],
        [capability[name]["win_rate"]["estimate"] for name in complete],
    ) if len(complete) >= 4 else {"status": "incomplete"}
    round_curve = select_curve(
        list(range(1, len(rounds) + 1)),
        [capability[name]["win_rate"]["estimate"] for name in rounds],
    ) if all(capability[name]["win_rate"] for name in rounds) else {"status": "incomplete"}

    stage3 = defaultdict(lambda: defaultdict(dict))
    for row in stage3_rows:
        if row.get("valid", True):
            stage3[row["percent_self"]][row["round"]][row["prompt_id"]] = row
    stage3_rounds = {}
    for percent in config["stage3"]["label_mixture_percent_self"]:
        stage3_rounds[str(percent)] = {}
        for round_index in range(1, config["stage3"]["rounds_per_condition"] + 1):
            rows = list(stage3[percent][round_index].values())
            seed = config["evaluation"]["bootstrap_seed"] + 500 + percent + round_index
            stage3_rounds[str(percent)][str(round_index)] = {
                "external_evaluator_win_rate": _mean_interval([row["external_evaluator_win"] for row in rows], config, seed),
                "external_margin": _mean_interval([row["external_margin"] for row in rows], config, seed + 1000),
                "ensemble_disagreement": _mean_interval([row["ensemble_disagreement"] for row in rows], config, seed + 2000),
                "response_tokens": _mean_interval([row["response_tokens"] for row in rows], config, seed + 3000),
                "kl_from_stage3_start": _mean_interval([row["kl_from_stage3_start"] for row in rows], config, seed + 4000),
            }
    comparisons = {}
    final_round = config["stage3"]["rounds_per_condition"]
    baseline = stage3[0][final_round]
    raw_p_values = {}
    for offset, percent in enumerate(config["stage3"]["label_mixture_percent_self"]):
        if percent == 0: continue
        condition = stage3[percent][final_round]
        joint = sorted(set(baseline) & set(condition))
        left = [condition[key]["external_evaluator_win"] for key in joint]
        right = [baseline[key]["external_evaluator_win"] for key in joint]
        comparisons[str(percent)] = paired_bootstrap(
            left,
            [baseline[key]["external_evaluator_win"] for key in joint],
            replicates=config["evaluation"]["bootstrap_replicates"], seed=config["evaluation"]["bootstrap_seed"] + 300 + offset,
        ) if joint else None
        if joint:
            test = paired_sign_flip_test(
                left, right, replicates=config["evaluation"]["bootstrap_replicates"],
                seed=config["evaluation"]["bootstrap_seed"] + 400 + offset,
            )
            comparisons[str(percent)]["p_value_two_sided"] = test["p_value_two_sided"]
            raw_p_values[str(percent)] = test["p_value_two_sided"]
    adjusted = holm_adjust(raw_p_values) if raw_p_values else {}
    for percent, value in adjusted.items():
        comparisons[percent]["holm_adjusted_p_value"] = value

    length_shifts = {}
    for percent in config["stage3"]["label_mixture_percent_self"]:
        if percent == 0: continue
        condition = stage3[percent][final_round]
        joint = sorted(set(baseline) & set(condition))
        length_shifts[str(percent)] = paired_bootstrap(
            [condition[key]["response_tokens"] for key in joint],
            [baseline[key]["response_tokens"] for key in joint],
            replicates=config["evaluation"]["bootstrap_replicates"],
            seed=config["evaluation"]["bootstrap_seed"] + 700 + percent,
        ) if joint else None

    label_agreement = {}
    audit_groups = defaultdict(list)
    for row in stage3_training_audit or []:
        audit_groups[(str(row["percent_self"]), str(row["round"]), "all")].append(row["external_evaluator_agrees"])
        audit_groups[(str(row["percent_self"]), str(row["round"]), row["label_source"])].append(row["external_evaluator_agrees"])
    for (percent, round_index, source), values in sorted(audit_groups.items()):
        label_agreement.setdefault(percent, {}).setdefault(round_index, {})[source] = _mean_interval(
            values, config, config["evaluation"]["bootstrap_seed"] + 900 + int(percent) + int(round_index)
        )
    return {
        "study_id": config["study_id"], "capability": capability,
        "stage1_round_gains": gains, "stage1_plateau_round": plateau_round,
        "stage1_curve": round_curve, "stage2_density_curve": density_curve,
        "stage2_checkpoints_in_fit": complete,
        "stage2_compute": {name: compute[name] for name in complete},
        "stage3_round_metrics": stage3_rounds,
        "stage3_vs_zero_percent": comparisons, "stage3_round4_length_shift_vs_zero": length_shifts,
        "stage3_training_label_agreement": label_agreement,
        "training_seeds": len(config["scope"]["training_seeds"]),
    }


def write_results(metrics, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Recursive Self-Improvement Results", "", f"Study: `{metrics['study_id']}`", ""]
    if metrics["stage1_curve"].get("status") == "incomplete":
        lines.append("Status: incomplete; no empirical claim is made.")
    else:
        lines.extend([
            "## Stage 1: horizon and plateau", "",
            "| Checkpoint | Win rate vs SFT | 95% CI | N | Incremental gain | Gain 95% CI |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for checkpoint, entry in metrics["capability"].items():
            win = entry["win_rate"]
            if win is None: continue
            gain = metrics["stage1_round_gains"].get(checkpoint)
            gain_columns = "— | —" if gain is None else f"{gain['estimate']:.3f} | [{gain['ci95'][0]:.3f}, {gain['ci95'][1]:.3f}]"
            lines.append(f"| `{checkpoint}` | {win['estimate']:.3f} | [{win['ci95'][0]:.3f}, {win['ci95'][1]:.3f}] | {win['n_trials']} | {gain_columns} |")
        selected = metrics["stage1_curve"]["selected"]
        candidates = metrics["stage1_curve"]["candidates"]
        lines.extend([
            "",
            f"The frozen plateau rule first triggers at round **{metrics['stage1_plateau_round'] or 'not detected'}**. The selected descriptive curve is **{selected}** (linear LOO MSE {candidates['linear']['loo_mse']:.6g}; saturating-exponential LOO MSE {candidates['saturating_exponential']['loo_mse']:.6g}).",
            "",
            "## Stage 2: capability versus measured compute", "",
            "| Checkpoint | Cumulative non-padding training tokens | Optimizer steps | Runtime (s) | Peak GPU bytes |",
            "|---|---:|---:|---:|---:|",
        ])
        for checkpoint in metrics["stage2_checkpoints_in_fit"]:
            row = metrics["stage2_compute"][checkpoint]
            lines.append(f"| `{checkpoint}` | {row['cumulative_non_padding_training_tokens']:,} | {row.get('cumulative_optimizer_steps', '—')} | {row.get('cumulative_runtime_seconds', '—')} | {row.get('incremental_peak_allocated_gpu_memory_bytes', '—')} |")
        density = metrics["stage2_density_curve"]
        dc = density["candidates"]
        lines.extend(["", f"The selected descriptive capability–compute curve is **{density['selected']}** (linear LOO MSE {dc['linear']['loo_mse']:.6g}; saturating-exponential LOO MSE {dc['saturating_exponential']['loo_mse']:.6g})."])
    if any(value is not None for value in metrics["stage3_vs_zero_percent"].values()):
        lines.extend(["", "## Stage 3: self-label reliance", "", "### Primary round-4 comparisons", "", "| Self labels | Round-4 win-rate difference vs 0% | 95% CI | N | Holm p |", "|---:|---:|---:|---:|---:|"])
        for percent, value in metrics["stage3_vs_zero_percent"].items():
            if value is None: continue
            lines.append(f"| {percent}% | {value['estimate']:.3f} | [{value['ci95'][0]:.3f}, {value['ci95'][1]:.3f}] | {value['n_trials']} | {value['holm_adjusted_p_value']:.4f} |")
        lines.extend(["", "### Per-round evaluator trajectory and diagnostics", "", "| Self labels | Round | K=3 win rate | 95% CI | N | Mean margin | Mean disagreement | Mean response tokens | Mean KL |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
        for percent, rounds in metrics["stage3_round_metrics"].items():
            for round_index, values in rounds.items():
                win = values["external_evaluator_win_rate"]
                if win is None: continue
                lines.append(f"| {percent}% | {round_index} | {win['estimate']:.3f} | [{win['ci95'][0]:.3f}, {win['ci95'][1]:.3f}] | {win['n_trials']} | {values['external_margin']['estimate']:.3f} | {values['ensemble_disagreement']['estimate']:.3f} | {values['response_tokens']['estimate']:.2f} | {values['kl_from_stage3_start']['estimate']:.4f} |")
        lines.extend(["", "### Training-label agreement with evaluation-only K=3 ensemble", "", "| Self labels | Round | Label source | Agreement | 95% CI | N |", "|---:|---:|---|---:|---:|---:|"])
        for percent, rounds in metrics["stage3_training_label_agreement"].items():
            for round_index, sources in rounds.items():
                for source, value in sources.items():
                    lines.append(f"| {percent}% | {round_index} | {source} | {value['estimate']:.3f} | [{value['ci95'][0]:.3f}, {value['ci95'][1]:.3f}] | {value['n_trials']} |")
        lines.extend(["", "Intervals resample held-out prompts and do not include training-seed uncertainty (one training seed)."])
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
