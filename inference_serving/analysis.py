"""Generated serving, quantization-quality, and speculative-decoding analysis."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from inference_serving.data import read_jsonl
from inference_serving.statistics import bootstrap_ci, holm_adjust, paired_bootstrap, paired_sign_permutation, wilson
from llm_judge_summeval.ledger import load_latest


PERFORMANCE_METRICS = ("output_tokens_per_second", "request_throughput", "ttft_ms_p50", "ttft_ms_p95", "itl_ms_p50", "itl_ms_p95", "peak_gpu_memory_bytes")


def _is_confirmatory(row: dict) -> bool:
    """Accept held-out rows plus legacy HF rows recorded before phase was added.

    Pilot HF measurements live in separate smoke ledgers, so an unphased HF row in
    the preregistered raw-trial ledger is an original Stage-1 confirmatory row.
    """
    return row.get("phase") == "heldout" or (
        row.get("phase") is None and row.get("system") == "hf"
    )


def _index_trials(rows: list[dict]) -> dict[tuple, dict]:
    indexed = {}
    for row in rows:
        key = (row["system"], row["target"], row["precision"], bool(row.get("speculative")), row["concurrency"], row["trial_index"])
        if key in indexed:
            raise ValueError(f"Duplicate benchmark trial: {key}")
        indexed[key] = row
    return indexed


def paired_trial_family(indexed: dict, left_filter: tuple, right_filter: tuple, concurrency: int, config: dict) -> dict:
    left = {key[-1]: row for key, row in indexed.items() if key[:4] == left_filter and key[4] == concurrency}
    right = {key[-1]: row for key, row in indexed.items() if key[:4] == right_filter and key[4] == concurrency}
    joint = sorted(set(left) & set(right))
    if not joint:
        return {"status": "missing", "n_trials": 0}
    output = {"status": "complete", "n_trials": len(joint), "metrics": {}}
    p_values = {}
    for offset, metric in enumerate(PERFORMANCE_METRICS):
        left_values = [left[index][metric] for index in joint]
        right_values = [right[index][metric] for index in joint]
        output["metrics"][metric] = paired_bootstrap(
            left_values, right_values,
            replicates=config["statistics"]["bootstrap_replicates"], seed=config["statistics"]["bootstrap_seed"] + offset,
        )
        output["metrics"][metric]["left"] = bootstrap_ci(left_values, replicates=config["statistics"]["bootstrap_replicates"], seed=config["statistics"]["bootstrap_seed"] + 20 + offset)
        output["metrics"][metric]["right"] = bootstrap_ci(right_values, replicates=config["statistics"]["bootstrap_replicates"], seed=config["statistics"]["bootstrap_seed"] + 40 + offset)
        p_values[metric] = paired_sign_permutation(left_values, right_values, permutations=10000, seed=config["statistics"]["bootstrap_seed"] + 60 + offset)
    adjusted = holm_adjust(p_values)
    for metric in PERFORMANCE_METRICS:
        output["metrics"][metric]["paired_permutation_p"] = p_values[metric]
        output["metrics"][metric]["holm_adjusted_p"] = adjusted[metric]
    return output


def _judge_scores(path: Path) -> dict[tuple[str, str], dict]:
    scores = {}
    for row in load_latest(path).values():
        if row.get("status") != "success": continue
        metadata = row["metadata"]
        scores[(metadata["model_label"], metadata["article_id"])] = {
            axis: value["score"] for axis, value in row["parsed"].items()
        }
    return scores


def quality_comparison(ledger: Path, config: dict) -> dict:
    scores = _judge_scores(ledger); labels = ("dpo_gptq", "dpo_fp16")
    ids = sorted({item for label, item in scores if label == labels[0]} & {item for label, item in scores if label == labels[1]})
    planned = config["data"]["heldout_articles"]
    result = {"coverage": wilson(len(ids), planned), "axes": {}}
    for offset, axis in enumerate(config["stage2"]["quality_primary_axes"]):
        metric = paired_bootstrap([scores[(labels[0], item)][axis] for item in ids], [scores[(labels[1], item)][axis] for item in ids],
            replicates=config["statistics"]["bootstrap_replicates"], seed=config["statistics"]["bootstrap_seed"] + 100 + offset)
        metric["equivalent"] = metric["ci95"][0] >= config["stage2"]["quality_equivalence_margin_points"]
        result["axes"][axis] = metric
    result["success"] = result["coverage"]["rate"] >= config["quality_evaluation"]["minimum_valid_fraction"] and all(row["equivalent"] for row in result["axes"].values())
    return result


def speculative_acceptance(rows: list[dict], config: dict) -> dict:
    output = {}
    for target in config["scope"]["targets"]:
        target_rows = [
            row for row in rows
            if row["target"] == target
            and row.get("speculative")
            and row.get("phase") == "heldout"
        ]
        rates = []
        for row in target_rows:
            counters = row.get("speculative_counter_delta", {})
            accepted = sum(
                value for key, value in counters.items()
                if key.startswith("vllm:spec_decode_num_accepted_tokens_total{")
            )
            drafted = sum(
                value for key, value in counters.items()
                if key.startswith("vllm:spec_decode_num_draft_tokens_total{")
            )
            if drafted > 0: rates.append(accepted / drafted)
        output[target] = bootstrap_ci(rates, replicates=config["statistics"]["bootstrap_replicates"],
                                      seed=config["statistics"]["bootstrap_seed"] + 200) if rates else {"status": "instrumentation_incomplete", "n_trials": 0}
    return output


def acceptance_differences(rows: list[dict], config: dict) -> dict:
    by_target = {}
    for row in rows:
        if not row.get("speculative") or row.get("phase") != "heldout":
            continue
        counters = row.get("speculative_counter_delta", {})
        accepted = sum(value for key, value in counters.items()
                       if key.startswith("vllm:spec_decode_num_accepted_tokens_total{"))
        drafted = sum(value for key, value in counters.items()
                      if key.startswith("vllm:spec_decode_num_draft_tokens_total{"))
        if drafted > 0:
            by_target.setdefault(row["target"], {})[row["trial_index"]] = accepted / drafted
    output, p_values = {}, {}
    for offset, (left_name, right_name) in enumerate((("base", "sft"), ("sft", "dpo"))):
        joint = sorted(set(by_target.get(left_name, {})) & set(by_target.get(right_name, {})))
        name = f"{left_name}_minus_{right_name}"
        if not joint:
            output[name] = {"status": "missing", "n_trials": 0}
            continue
        left = [by_target[left_name][index] for index in joint]
        right = [by_target[right_name][index] for index in joint]
        output[name] = paired_bootstrap(
            left, right, replicates=config["statistics"]["bootstrap_replicates"],
            seed=config["statistics"]["bootstrap_seed"] + 300 + offset,
        )
        p_values[name] = paired_sign_permutation(
            left, right, permutations=10000,
            seed=config["statistics"]["bootstrap_seed"] + 320 + offset,
        )
    adjusted = holm_adjust(p_values)
    for name, p_value in p_values.items():
        output[name]["paired_permutation_p"] = p_value
        output[name]["holm_adjusted_p"] = adjusted[name]
    return output


def analyze(config: dict, trial_path: Path, judge_ledger: Path, output_dir: Path) -> dict:
    rows = read_jsonl(trial_path)
    heldout_rows = [row for row in rows if _is_confirmatory(row)]
    indexed = _index_trials(heldout_rows)
    primary = config["stage1"]["primary_concurrency"]
    stage1 = paired_trial_family(indexed, ("vllm", "dpo", "fp16", False), ("hf", "dpo", "fp16", False), primary, config)
    stage2_perf = paired_trial_family(indexed, ("vllm", "dpo", "gptq", False), ("vllm", "dpo", "fp16", False), primary, config)
    stage3_perf = paired_trial_family(indexed, ("vllm", "dpo", "gptq", True), ("vllm", "dpo", "gptq", False), primary, config)
    environment_path = output_dir / "environment_manifest.json"
    environment = json.loads(environment_path.read_text(encoding="utf-8")) if environment_path.exists() else {}
    gpu_names = [row.get("name") for row in environment.get("gpus", []) if row.get("name")]
    metrics = {
        "study_id": config["study_id"], "stage1": stage1, "stage2_performance": stage2_perf,
        "stage2_quality": quality_comparison(judge_ledger, config) if judge_ledger.exists() else {"status": "missing"},
        "stage3_performance": stage3_perf, "stage3_acceptance": speculative_acceptance(heldout_rows, config),
        "stage3_acceptance_differences": acceptance_differences(heldout_rows, config),
        "training_seeds": 1,
        "execution_hardware": {
            "serving_gpu_names": gpu_names,
            "draft_training_gpu": "NVIDIA GeForce RTX 3070 8GB",
            "platform_amendment": "inference_serving_v1_platform_001",
        },
        "provenance": {"config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                       "trials_sha256": hashlib.sha256(trial_path.read_bytes()).hexdigest() if trial_path.exists() else None,
                       "judge_ledger_sha256": hashlib.sha256(judge_ledger.read_bytes()).hexdigest() if judge_ledger.exists() else None,
                       "environment_manifest_sha256": hashlib.sha256(environment_path.read_bytes()).hexdigest() if environment_path.exists() else None,
                       "speculative_selection_sha256": hashlib.sha256((output_dir / "speculative_selection.json").read_bytes()).hexdigest() if (output_dir / "speculative_selection.json").exists() else None}
    }
    for stage in ("stage1", "stage2_performance", "stage3_performance"):
        family = metrics[stage]
        family["throughput_success"] = family.get("status") == "complete" and family["metrics"]["output_tokens_per_second"]["ci95"][0] > 0
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Alignment-Aware Inference Serving Results", "", f"Study: `{config['study_id']}`", "",
             "All estimates below are generated from raw trials; intervals are 95% bootstrap CIs.",
             "The original Hugging Face Stage-1 rows predate the ledger's `phase` field; they are treated as confirmatory because HF pilot/smoke measurements were written to separate ledgers.", ""]
    for title, key in (("Stage 1: vLLM minus Hugging Face", "stage1"), ("Stage 2: GPTQ minus FP16", "stage2_performance"), ("Stage 3: speculative minus ordinary", "stage3_performance")):
        family = metrics[key]; lines.extend([f"## {title}", ""])
        if family.get("status") != "complete": lines.extend(["Status: incomplete.", ""]); continue
        lines.extend(["| Metric | Difference [95% CI] | N trials | Holm p |", "|---|---:|---:|---:|"])
        for name, value in family["metrics"].items():
            lines.append(f"| {name} | {value['estimate']:.4g} [{value['ci95'][0]:.4g}, {value['ci95'][1]:.4g}] | {value['n_trials']} | {value['holm_adjusted_p']:.4g} |")
        lines.extend(["", f"Preregistered throughput criterion: **{'PASS' if family['throughput_success'] else 'FAIL'}**.", ""])
    quality = metrics["stage2_quality"]
    lines.extend(["## Stage 2: frozen-judge quality", ""])
    if quality.get("axes"):
        lines.extend([
            f"Coverage: {quality['coverage']['valid']}/{quality['coverage']['total']} "
            f"({quality['coverage']['rate']:.1%}; Wilson 95% CI "
            f"[{quality['coverage']['wilson_ci95'][0]:.1%}, {quality['coverage']['wilson_ci95'][1]:.1%}]).",
            "",
            "| Axis | GPTQ - FP16 [95% CI] | N pairs | Equivalent |",
            "|---|---:|---:|---:|",
        ])
        for axis, value in quality["axes"].items():
            lines.append(
                f"| {axis} | {value['estimate']:.4g} "
                f"[{value['ci95'][0]:.4g}, {value['ci95'][1]:.4g}] | "
                f"{value['n_trials']} | {'yes' if value['equivalent'] else 'no'} |"
            )
        lines.extend(["", f"Preregistered quality-equivalence criterion: **{'PASS' if quality['success'] else 'FAIL'}**.", ""])
    else:
        lines.extend(["Status: incomplete.", ""])
    lines.extend([
        "## Stage 3: draft-token acceptance by alignment state",
        "",
        "| Target | Acceptance rate [95% CI] | N trials |",
        "|---|---:|---:|",
    ])
    for target, value in metrics["stage3_acceptance"].items():
        if value.get("n_trials", 0):
            lines.append(
                f"| {target} | {value['estimate']:.1%} "
                f"[{value['ci95'][0]:.1%}, {value['ci95'][1]:.1%}] | {value['n_trials']} |"
            )
        else:
            lines.append(f"| {target} | instrumentation incomplete | 0 |")
    lines.extend([
        "",
        "| Planned contrast | Acceptance-rate difference [95% CI] | N paired trials | Holm p |",
        "|---|---:|---:|---:|",
    ])
    for name, value in metrics["stage3_acceptance_differences"].items():
        if value.get("n_trials", 0):
            lines.append(
                f"| {name.replace('_', ' ')} | {value['estimate']:.2%} "
                f"[{value['ci95'][0]:.2%}, {value['ci95'][1]:.2%}] | "
                f"{value['n_trials']} | {value['holm_adjusted_p']:.4g} |"
            )
        else:
            lines.append(f"| {name.replace('_', ' ')} | missing | 0 | - |")
    lines.append("")
    lines.extend(["## Scope boundary", "", config["scope"]["grpo_exclusion_reason"], "",
                  f"Serving measurements ran on {', '.join(gpu_names) if gpu_names else 'the manifest-recorded Linux GPU'}; the draft alone was trained on an RTX 3070. These measurements validate this concrete stack only and are not multi-GPU or datacenter-scale claims.", ""])
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return metrics
