"""Generated analysis for checkpoint density, plateau, and self-reliance."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from recursive_self_improvement.curves import select_curve
from recursive_self_improvement.statistics import paired_bootstrap, wilson


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


def analyze(config, evaluations, compute_rows, stage3_rows):
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

    stage3 = defaultdict(dict)
    for row in stage3_rows:
        if row.get("valid", True) and row["round"] == config["stage3"]["rounds_per_condition"]:
            stage3[row["percent_self"]][row["prompt_id"]] = row
    comparisons = {}
    baseline = stage3.get(0, {})
    for offset, percent in enumerate(config["stage3"]["label_mixture_percent_self"]):
        if percent == 0: continue
        joint = sorted(set(baseline) & set(stage3.get(percent, {})))
        comparisons[str(percent)] = paired_bootstrap(
            [stage3[percent][key]["external_evaluator_win"] for key in joint],
            [baseline[key]["external_evaluator_win"] for key in joint],
            replicates=config["evaluation"]["bootstrap_replicates"], seed=config["evaluation"]["bootstrap_seed"] + 300 + offset,
        ) if joint else None
    return {
        "study_id": config["study_id"], "capability": capability,
        "stage1_round_gains": gains, "stage1_plateau_round": plateau_round,
        "stage1_curve": round_curve, "stage2_density_curve": density_curve,
        "stage2_checkpoints_in_fit": complete, "stage3_vs_zero_percent": comparisons,
        "training_seeds": len(config["scope"]["training_seeds"]),
    }


def write_results(metrics, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Recursive Self-Improvement Results", "", f"Study: `{metrics['study_id']}`", ""]
    if metrics["stage1_curve"].get("status") == "incomplete":
        lines.append("Status: incomplete; no empirical claim is made.")
    else:
        lines.extend([f"Selected Stage-1 curve: **{metrics['stage1_curve']['selected']}**.", "", f"Plateau round: **{metrics['stage1_plateau_round'] or 'not detected'}**."])
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
