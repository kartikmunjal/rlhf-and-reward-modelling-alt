"""Generated serving, quantization-quality, and speculative-decoding analysis."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from inference_serving.data import read_jsonl
from inference_serving.statistics import bootstrap_ci, holm_adjust, paired_bootstrap, paired_sign_permutation, wilson
from llm_judge_summeval.ledger import load_latest


PERFORMANCE_METRICS = ("output_tokens_per_second", "request_throughput", "ttft_ms_p50", "ttft_ms_p95", "itl_ms_p50", "itl_ms_p95", "peak_gpu_memory_bytes")


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
        target_rows = [row for row in rows if row["target"] == target and row.get("speculative")]
        rates = []
        for row in target_rows:
            counters = row.get("speculative_counter_delta", {})
            accepted = sum(value for key, value in counters.items() if "accepted" in key.lower() and "token" in key.lower())
            drafted = sum(value for key, value in counters.items() if ("draft" in key.lower() or "spec" in key.lower()) and "token" in key.lower() and "accepted" not in key.lower())
            if drafted > 0: rates.append(accepted / drafted)
        output[target] = bootstrap_ci(rates, replicates=config["statistics"]["bootstrap_replicates"],
                                      seed=config["statistics"]["bootstrap_seed"] + 200) if rates else {"status": "instrumentation_incomplete", "n_trials": 0}
    return output


def analyze(config: dict, trial_path: Path, judge_ledger: Path, output_dir: Path) -> dict:
    rows = read_jsonl(trial_path); indexed = _index_trials(rows); primary = config["stage1"]["primary_concurrency"]
    stage1 = paired_trial_family(indexed, ("vllm", "dpo", "fp16", False), ("hf", "dpo", "fp16", False), primary, config)
    stage2_perf = paired_trial_family(indexed, ("vllm", "dpo", "gptq", False), ("vllm", "dpo", "fp16", False), primary, config)
    stage3_perf = paired_trial_family(indexed, ("vllm", "dpo", "gptq", True), ("vllm", "dpo", "gptq", False), primary, config)
    metrics = {
        "study_id": config["study_id"], "stage1": stage1, "stage2_performance": stage2_perf,
        "stage2_quality": quality_comparison(judge_ledger, config) if judge_ledger.exists() else {"status": "missing"},
        "stage3_performance": stage3_perf, "stage3_acceptance": speculative_acceptance(rows, config),
        "training_seeds": 1,
        "provenance": {"config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                       "trials_sha256": hashlib.sha256(trial_path.read_bytes()).hexdigest() if trial_path.exists() else None}
    }
    for stage in ("stage1", "stage2_performance", "stage3_performance"):
        family = metrics[stage]
        family["throughput_success"] = family.get("status") == "complete" and family["metrics"]["output_tokens_per_second"]["ci95"][0] > 0
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Alignment-Aware Inference Serving Results", "", f"Study: `{config['study_id']}`", "",
             "All estimates below are generated from raw trials; intervals are 95% bootstrap CIs.", ""]
    for title, key in (("Stage 1: vLLM minus Hugging Face", "stage1"), ("Stage 2: GPTQ minus FP16", "stage2_performance"), ("Stage 3: speculative minus ordinary", "stage3_performance")):
        family = metrics[key]; lines.extend([f"## {title}", ""])
        if family.get("status") != "complete": lines.extend(["Status: incomplete.", ""]); continue
        lines.extend(["| Metric | Difference [95% CI] | N trials | Holm p |", "|---|---:|---:|---:|"])
        for name, value in family["metrics"].items():
            lines.append(f"| {name} | {value['estimate']:.4g} [{value['ci95'][0]:.4g}, {value['ci95'][1]:.4g}] | {value['n_trials']} | {value['holm_adjusted_p']:.4g} |")
        lines.extend(["", f"Preregistered throughput criterion: **{'PASS' if family['throughput_success'] else 'FAIL'}**.", ""])
    lines.extend(["## Scope boundary", "", config["scope"]["grpo_exclusion_reason"], "",
                  "These single-RTX-3070 measurements validate this concrete serving stack only; they are not multi-GPU or datacenter-scale claims.", ""])
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return metrics
