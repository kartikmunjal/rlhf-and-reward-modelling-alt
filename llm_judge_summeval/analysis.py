"""Preregistered held-out analysis for SummEval judge outputs."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from llm_judge_summeval.data import AXES, file_sha256, load_heldout_inputs, load_heldout_labels_for_analysis
from llm_judge_summeval.ledger import load_latest
from llm_judge_summeval.prompts import verify_final_prompt_manifest
from llm_judge_summeval.statistics import (
    article_cluster_bootstrap, fit_bradley_terry, partial_spearman, position_bias_counts, rouge_l_fmeasure, spearman,
    wilson_interval,
)


def _metric(article_ids, x, y, config):
    return article_cluster_bootstrap(article_ids, (x, y), spearman,
                                     replicates=config["statistics"]["bootstrap_replicates"],
                                     seed=config["statistics"]["bootstrap_seed"])


def _difference(article_ids, judge, baseline, human, config):
    return article_cluster_bootstrap(article_ids, (judge, baseline, human),
                                     lambda j, b, h: spearman(j, h) - spearman(b, h),
                                     replicates=config["statistics"]["bootstrap_replicates"],
                                     seed=config["statistics"]["bootstrap_seed"] + 1)


def pointwise_analysis(provider: str, ledger_path: Path, inputs: list[dict], labels: dict, config: dict) -> dict:
    latest = load_latest(ledger_path)
    by_summary = {row["metadata"]["summary_id"]: row for row in latest.values() if row["status"] == "success"}
    valid = [row for row in inputs if row["summary_id"] in by_summary]
    total, successes = len(inputs), len(valid)
    if successes / total < config["missingness"]["minimum_valid_fraction"]:
        status = "infrastructure_incomplete"
    else:
        status = "complete"
    article_ids = np.asarray([row["article_id"] for row in valid])
    rouge = np.asarray([rouge_l_fmeasure(row["summary"], row["reference"]) for row in valid])
    lengths = np.asarray([len(row["summary"].split()) for row in valid])
    axes = {}
    for axis in AXES:
        judge = np.asarray([by_summary[row["summary_id"]]["parsed"][axis]["score"] for row in valid], dtype=float)
        human = np.asarray([labels[row["summary_id"]]["expert_mean"][axis] for row in valid], dtype=float)
        axes[axis] = {
            "judge_human": _metric(article_ids, judge, human, config),
            "rouge_human": _metric(article_ids, rouge, human, config),
            "judge_minus_rouge": _difference(article_ids, judge, rouge, human, config),
            "length_bias": _metric(article_ids, judge, lengths, config),
            "judge_human_controlling_length": article_cluster_bootstrap(
                article_ids, (judge, human, lengths), partial_spearman,
                replicates=config["statistics"]["bootstrap_replicates"],
                seed=config["statistics"]["bootstrap_seed"] + 3,
            ),
        }
    usage = {"input_tokens": sum(row.get("input_tokens", 0) for row in by_summary.values()),
             "output_tokens": sum(row.get("output_tokens", 0) for row in by_summary.values())}
    return {"provider": provider, "status": status, "valid": successes, "total": total,
            "valid_rate": successes / total, "valid_rate_wilson_ci95": wilson_interval(successes, total),
            "usage": usage, "axes": axes,
            "scores": {summary_id: {axis: row["parsed"][axis]["score"] for axis in AXES} for summary_id, row in by_summary.items()}}


def cross_provider_analysis(primary: dict, secondary: dict, inputs: list[dict], config: dict) -> dict:
    joint = sorted(set(primary["scores"]) & set(secondary["scores"]))
    article_by_summary = {row["summary_id"]: row["article_id"] for row in inputs}
    articles = [article_by_summary[item] for item in joint]
    return {"joint_valid": len(joint), "axes": {
        axis: _metric(articles, [primary["scores"][item][axis] for item in joint],
                      [secondary["scores"][item][axis] for item in joint], config) for axis in AXES
    }}


def _underlying_winner(row: dict, axis: str) -> str:
    winner = row["parsed"][axis]["winner"]
    if winner == "tie":
        return "tie"
    return row["metadata"]["display_a_id" if winner == "A" else "display_b_id"]


def _human_winner(left: str, right: str, labels: dict, axis: str) -> str:
    difference = labels[left]["expert_mean"][axis] - labels[right]["expert_mean"][axis]
    return left if difference > 0 else right if difference < 0 else "tie"


def pairwise_analysis(ledger_path: Path, labels: dict, config: dict) -> dict:
    latest = load_latest(ledger_path)
    success = [row for row in latest.values() if row["status"] == "success"]
    expected_requests = config["pairwise"]["expected_requests"]
    parsed_rows = []
    for row in success:
        for axis in AXES:
            parsed_rows.append({"pair_id": row["metadata"]["pair_id"], "article_id": row["metadata"]["article_id"],
                                "axis": axis, "order": row["metadata"]["order"],
                                "left": row["metadata"]["display_a_id"], "right": row["metadata"]["display_b_id"],
                                "underlying_winner": _underlying_winner(row, axis)})
    request_pairs = defaultdict(set)
    for row in success:
        request_pairs[row["metadata"]["pair_id"]].add(row["metadata"]["order"])
    complete_pair_count = sum(orders == {"ab", "ba"} for orders in request_pairs.values())
    expected_pairs = config["pairwise"]["expected_unique_pairs"]
    result = {"status": "complete" if complete_pair_count / expected_pairs >= config["missingness"]["minimum_valid_fraction"] else "infrastructure_incomplete",
              "valid": complete_pair_count, "total": expected_pairs, "valid_rate": complete_pair_count / expected_pairs,
              "valid_rate_wilson_ci95": wilson_interval(complete_pair_count, expected_pairs),
              "valid_requests": len(success), "total_requests": expected_requests, "axes": {},
              "usage": {"input_tokens": sum(row.get("input_tokens", 0) for row in success),
                        "output_tokens": sum(row.get("output_tokens", 0) for row in success)}}
    for axis in AXES:
        rows = [row for row in parsed_rows if row["axis"] == axis]
        counts = position_bias_counts(rows)
        grouped = defaultdict(dict)
        for row in rows:
            grouped[row["pair_id"]][row["order"]] = row
        complete = [orders for orders in grouped.values() if set(orders) == {"ab", "ba"}]
        first_correct, symmetric_correct, pair_ids, flips, tie_instability = [], [], [], [], []
        bt_by_article = defaultdict(list)
        for orders in complete:
            first, second = orders["ab"], orders["ba"]
            human = _human_winner(first["left"], first["right"], labels, axis)
            first_correct.append(float(first["underlying_winner"] == human))
            sym = first["underlying_winner"] if first["underlying_winner"] == second["underlying_winner"] else "tie"
            symmetric_correct.append(float(sym == human))
            pair_ids.append(first["pair_id"])
            flips.append(float("tie" not in (first["underlying_winner"], second["underlying_winner"])
                               and first["underlying_winner"] != second["underlying_winner"]))
            tie_instability.append(float("tie" in (first["underlying_winner"], second["underlying_winner"])
                                         and first["underlying_winner"] != second["underlying_winner"]))
            for verdict in (first, second):
                outcome = 0.5 if verdict["underlying_winner"] == "tie" else float(verdict["underlying_winner"] == verdict["left"])
                bt_by_article[first["article_id"]].append((verdict["left"], verdict["right"], outcome))
        bt_scores, bt_human, bt_articles = [], [], []
        for article_id, comparisons in bt_by_article.items():
            items = sorted({item for left, right, _ in comparisons for item in (left, right)})
            scores = fit_bradley_terry(items, comparisons, config["pairwise"]["bradley_terry_l2"])
            for item in items:
                bt_scores.append(scores[item]); bt_human.append(labels[item]["expert_mean"][axis]); bt_articles.append(article_id)
        mitigation = article_cluster_bootstrap(pair_ids, (symmetric_correct, first_correct),
                                               lambda sym, first: np.mean(sym) - np.mean(first),
                                               replicates=config["statistics"]["bootstrap_replicates"],
                                               seed=config["statistics"]["bootstrap_seed"] + 2,
                                               cluster_unit="unordered_pair")
        flip_bootstrap = article_cluster_bootstrap(
            pair_ids, (flips,), lambda values: np.mean(values),
            replicates=config["statistics"]["bootstrap_replicates"],
            seed=config["statistics"]["bootstrap_seed"] + 4,
            cluster_unit="unordered_pair",
        )
        tie_bootstrap = article_cluster_bootstrap(
            pair_ids, (tie_instability,), lambda values: np.mean(values),
            replicates=config["statistics"]["bootstrap_replicates"],
            seed=config["statistics"]["bootstrap_seed"] + 5,
            cluster_unit="unordered_pair",
        )
        first_agreement = article_cluster_bootstrap(
            pair_ids, (first_correct,), lambda values: np.mean(values),
            replicates=config["statistics"]["bootstrap_replicates"],
            seed=config["statistics"]["bootstrap_seed"] + 6,
            cluster_unit="unordered_pair",
        )
        symmetric_agreement = article_cluster_bootstrap(
            pair_ids, (symmetric_correct,), lambda values: np.mean(values),
            replicates=config["statistics"]["bootstrap_replicates"],
            seed=config["statistics"]["bootstrap_seed"] + 7,
            cluster_unit="unordered_pair",
        )
        result["axes"][axis] = {
            "position": counts,
            "preference_flip_rate": counts["preference_flips"] / counts["complete_pairs"] if counts["complete_pairs"] else None,
            "preference_flip_wilson_ci95": wilson_interval(counts["preference_flips"], counts["complete_pairs"]) if counts["complete_pairs"] else None,
            "preference_flip_cluster_bootstrap": flip_bootstrap,
            "tie_instability_rate": counts["tie_instability"] / counts["complete_pairs"] if counts["complete_pairs"] else None,
            "tie_instability_wilson_ci95": wilson_interval(counts["tie_instability"], counts["complete_pairs"]) if counts["complete_pairs"] else None,
            "tie_instability_cluster_bootstrap": tie_bootstrap,
            "first_order_human_agreement": first_agreement,
            "first_order_human_agreement_wilson_ci95": wilson_interval(int(sum(first_correct)), len(first_correct)),
            "symmetrized_human_agreement": symmetric_agreement,
            "symmetrized_human_agreement_wilson_ci95": wilson_interval(int(sum(symmetric_correct)), len(symmetric_correct)),
            "symmetrized_minus_first": mitigation,
            "bradley_terry_human_spearman": _metric(bt_articles, bt_scores, bt_human, config),
        }
    return result


def analyze(results_dir: Path, processed_dir: Path, prompts_path: Path, prompt_manifest_path: Path, config: dict) -> dict:
    verify_final_prompt_manifest(prompts_path, prompt_manifest_path, config)
    inputs = load_heldout_inputs(processed_dir)
    label_rows = load_heldout_labels_for_analysis(processed_dir, prompt_manifest_verified=True)
    labels = {row["summary_id"]: row for row in label_rows}
    primary = pointwise_analysis("anthropic", results_dir / "anthropic_heldout_pointwise.jsonl", inputs, labels, config)
    secondary = pointwise_analysis("openai", results_dir / "openai_heldout_pointwise.jsonl", inputs, labels, config)
    pairwise = pairwise_analysis(results_dir / "anthropic_pairwise.jsonl", labels, config)
    success = {}
    for axis in config["axes"]["primary"]:
        row = primary["axes"][axis]
        success[axis] = row["judge_human"]["estimate"] >= config["success_rule"]["minimum_point_estimate_rho"] and row["judge_minus_rouge"]["ci95"][0] > 0
    rates = json.loads((prompts_path.parent / "runtime_pricing.json").read_text(encoding="utf-8"))["per_million_tokens"]
    def billed(usage, model):
        return (usage["input_tokens"] * rates[model]["input"] + usage["output_tokens"] * rates[model]["output"]) / 1e6
    actual_cost = {
        "anthropic_usd": billed(primary["usage"], config["judges"]["primary"]["model"])
                         + billed(pairwise["usage"], config["judges"]["primary"]["model"]),
        "openai_usd": billed(secondary["usage"], config["judges"]["secondary"]["model"]),
    }
    actual_cost["total_usd"] = actual_cost["anthropic_usd"] + actual_cost["openai_usd"]
    provenance_paths = {
        "prompts": prompts_path,
        "final_prompt_manifest": prompt_manifest_path,
        "data_manifest": processed_dir / "data_manifest.json",
        "anthropic_pointwise_ledger": results_dir / "anthropic_heldout_pointwise.jsonl",
        "openai_pointwise_ledger": results_dir / "openai_heldout_pointwise.jsonl",
        "anthropic_pairwise_ledger": results_dir / "anthropic_pairwise.jsonl",
        "runtime_pricing": prompts_path.parent / "runtime_pricing.json",
    }
    return {"study_id": config["study_id"], "status": "complete" if all(part["status"] == "complete" for part in (primary, secondary, pairwise)) else "infrastructure_incomplete",
            "primary_success_by_axis": success, "primary_success": all(success.values()),
            "actual_api_cost_at_recorded_rates": actual_cost,
            "provenance": {name: {"path": str(path), "sha256": file_sha256(path)}
                           for name, path in provenance_paths.items()},
            "pointwise": {"anthropic": primary, "openai": secondary},
            "cross_provider": cross_provider_analysis(primary, secondary, inputs, config), "pairwise": pairwise}
