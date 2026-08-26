"""Preregistered paired analysis for base, SFT, and judge-DPO summaries."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from llm_judge_summeval.ledger import load_latest
from llm_judge_summeval.statistics import rouge_l_fmeasure, wilson_interval
from summarization_finetune.data import file_sha256, read_jsonl


def bootstrap(article_ids, values, statistic, replicates: int, seed: int) -> dict:
    article_ids = np.asarray(article_ids); arrays = tuple(np.asarray(value, dtype=float) for value in values)
    unique = np.unique(article_ids); rng = np.random.default_rng(seed); draws = []
    for _ in range(replicates):
        sampled = rng.choice(unique, len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(article_ids == item) for item in sampled])
        draws.append(float(statistic(*(array[indices] for array in arrays))))
    estimate = float(statistic(*arrays)); finite = np.asarray([value for value in draws if np.isfinite(value)])
    return {"estimate": estimate, "ci95": [float(x) for x in np.quantile(finite, [0.025, 0.975])],
            "n_trials": len(arrays[0]), "n_article_clusters": len(unique), "bootstrap_replicates": replicates}


def paired_mean(article_ids, left, right, config: dict, seed_offset: int = 0) -> dict:
    return bootstrap(article_ids, (left, right), lambda x, y: np.mean(x - y),
                     config["statistics"]["bootstrap_replicates"], config["statistics"]["bootstrap_seed"] + seed_offset)


def controlled_treatment(article_ids, dpo_score, sft_score, dpo_length, sft_length, config: dict, seed_offset=20) -> dict:
    def coefficient(ds, ss, dl, sl):
        pooled = np.concatenate([dl, sl]); scale = pooled.std()
        dz = (dl - sl) / scale if scale else np.zeros_like(dl)
        design = np.column_stack([np.ones(len(dz)), dz])
        return np.linalg.lstsq(design, ds - ss, rcond=None)[0][0]
    return bootstrap(article_ids, (dpo_score, sft_score, dpo_length, sft_length), coefficient,
                     config["statistics"]["bootstrap_replicates"], config["statistics"]["bootstrap_seed"] + seed_offset)


def _scores(ledger_path: Path) -> tuple[dict, dict]:
    latest = load_latest(ledger_path)
    scores, usage = {}, {"input_tokens": 0, "output_tokens": 0}
    for row in latest.values():
        if row["status"] != "success": continue
        key = (row["metadata"]["model_label"], row["metadata"]["article_id"])
        scores[key] = {axis: row["parsed"][axis]["score"] for axis in row["parsed"]}
        usage["input_tokens"] += row.get("input_tokens", 0); usage["output_tokens"] += row.get("output_tokens", 0)
    return scores, usage


def analyze(*, config: dict, data_dir: Path, generations_dir: Path, judge_dir: Path,
            output_dir: Path, root: Path) -> dict:
    articles = {row["article_id"]: row for row in read_jsonl(data_dir / "final_eval_articles.jsonl")}
    generations = {}
    for model in config["evaluation_generation"]["models"]:
        for row in read_jsonl(generations_dir / f"{model}.jsonl"):
            generations[(model, row["article_id"])] = row
    claude, claude_usage = _scores(judge_dir / "anthropic_pointwise.jsonl")
    openai, openai_usage = _scores(judge_dir / "openai_pointwise.jsonl")
    axes, primary = ("coherence", "consistency", "fluency", "relevance"), config["statistics"]["primary_axes"]
    coverage = {}
    for provider, scores in (("anthropic", claude), ("openai", openai)):
        coverage[provider] = {}
        for model in config["evaluation_generation"]["models"]:
            valid = sum((model, article_id) in scores for article_id in articles)
            coverage[provider][model] = {"valid": valid, "total": len(articles), "rate": valid / len(articles),
                                         "wilson_ci95": wilson_interval(valid, len(articles))}
    comparisons = {}
    for name, better, worse in (("sft_minus_base", "sft", "base"), ("dpo_minus_sft", "dpo", "sft")):
        comparisons[name] = {"anthropic": {}, "openai": {}, "rougeL": None, "length_words": None}
        for provider, scores in (("anthropic", claude), ("openai", openai)):
            joint = sorted(article_id for article_id in articles if (better, article_id) in scores and (worse, article_id) in scores)
            for offset, axis in enumerate(axes):
                comparisons[name][provider][axis] = paired_mean(
                    joint, [scores[(better, item)][axis] for item in joint], [scores[(worse, item)][axis] for item in joint],
                    config, 10 * (name == "dpo_minus_sft") + offset,
                )
        joint_generation = sorted(article_id for article_id in articles if (better, article_id) in generations and (worse, article_id) in generations)
        better_rouge = [rouge_l_fmeasure(generations[(better, item)]["summary"], articles[item]["reference"]) for item in joint_generation]
        worse_rouge = [rouge_l_fmeasure(generations[(worse, item)]["summary"], articles[item]["reference"]) for item in joint_generation]
        better_len = [len(generations[(better, item)]["summary"].split()) for item in joint_generation]
        worse_len = [len(generations[(worse, item)]["summary"].split()) for item in joint_generation]
        comparisons[name]["rougeL"] = paired_mean(joint_generation, better_rouge, worse_rouge, config, 30)
        comparisons[name]["length_words"] = paired_mean(joint_generation, better_len, worse_len, config, 31)

    controlled = {}
    for offset, axis in enumerate(primary):
        joint = sorted(article_id for article_id in articles if all((model, article_id) in claude and (model, article_id) in generations for model in ("sft", "dpo")))
        controlled[axis] = controlled_treatment(
            joint, [claude[("dpo", item)][axis] for item in joint], [claude[("sft", item)][axis] for item in joint],
            [len(generations[("dpo", item)]["summary"].split()) for item in joint],
            [len(generations[("sft", item)]["summary"].split()) for item in joint], config, 40 + offset,
        )
    minimum_pairs = int(len(articles) * config["statistics"]["minimum_valid_fraction"] + 0.999999)
    sft_infrastructure = all(comparisons["sft_minus_base"]["anthropic"][axis]["n_trials"] >= minimum_pairs for axis in primary)
    dpo_eval_infrastructure = all(comparisons["dpo_minus_sft"]["anthropic"][axis]["n_trials"] >= minimum_pairs for axis in primary)
    sft_pass = {axis: sft_infrastructure and comparisons["sft_minus_base"]["anthropic"][axis]["ci95"][0] > 0 for axis in primary}
    hacking = {}
    for axis in primary:
        raw = comparisons["dpo_minus_sft"]["anthropic"][axis]["ci95"][0] > 0
        controlled_positive = controlled[axis]["ci95"][0] > 0
        rouge_positive = comparisons["dpo_minus_sft"]["rougeL"]["ci95"][0] > 0
        cross_positive = comparisons["dpo_minus_sft"]["openai"][axis]["ci95"][0] > 0
        classification = "length_exploitation_risk" if raw and not controlled_positive and not rouge_positive and not cross_positive else (
            "evidence_against_length_exploitation" if raw and controlled_positive and (rouge_positive or cross_positive) else "inconclusive")
        hacking[axis] = {"classification": classification, "raw_claude_positive": raw,
                         "length_controlled_positive": controlled_positive, "rouge_positive": rouge_positive,
                         "cross_provider_positive": cross_positive}
    pair_latest = load_latest(judge_dir / "anthropic_candidate_pairs.jsonl")
    pair_success = sum(row["status"] == "success" for row in pair_latest.values())
    planned_pairs = config["dpo_candidates"]["expected_judge_requests"]
    preference_infrastructure = pair_success / planned_pairs >= config["statistics"]["minimum_valid_fraction"]
    dpo_pass = {axis: dpo_eval_infrastructure and preference_infrastructure and
                comparisons["dpo_minus_sft"]["anthropic"][axis]["ci95"][0] > 0 for axis in primary}
    metrics = {
        "study_id": config["study_id"], "training_seeds": config["training_seed_scope"]["n_training_seeds"],
        "coverage": coverage, "comparisons": comparisons, "dpo_length_controlled": controlled,
        "sft_success_by_axis": sft_pass, "sft_success": all(sft_pass.values()),
        "dpo_success_by_axis": dpo_pass, "dpo_success": all(dpo_pass.values()), "reward_hacking": hacking,
        "infrastructure": {"sft_evaluation_complete": sft_infrastructure,
                           "dpo_evaluation_complete": dpo_eval_infrastructure,
                           "dpo_preference_judging_complete": preference_infrastructure},
        "candidate_pair_validity": {"valid": pair_success, "total": planned_pairs, "rate": pair_success / planned_pairs,
                                    "wilson_ci95": wilson_interval(pair_success, planned_pairs)},
        "usage": {"anthropic_pointwise": claude_usage, "openai_pointwise": openai_usage},
        "provenance": {},
    }
    paths = {"config": root / "summarization_finetune/study_config.json",
             "preregistration_manifest": root / "summarization_finetune/preregistration_manifest.json",
             "data_manifest": data_dir / "data_manifest.json",
             "anthropic_pointwise": judge_dir / "anthropic_pointwise.jsonl",
             "openai_pointwise": judge_dir / "openai_pointwise.jsonl",
             "candidate_pairs": judge_dir / "anthropic_candidate_pairs.jsonl",
             "base_generations": generations_dir / "base.jsonl", "sft_generations": generations_dir / "sft.jsonl",
             "dpo_generations": generations_dir / "dpo.jsonl"}
    metrics["provenance"] = {name: {"path": str(path), "sha256": file_sha256(path)} for name, path in paths.items()}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics
