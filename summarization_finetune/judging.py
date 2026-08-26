"""Frozen SummEval-judge execution and conservative DPO preference extraction."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

from llm_judge_summeval.ledger import execute_request, load_latest
from llm_judge_summeval.prompts import load_prompts, render_pairwise, render_pointwise
from llm_judge_summeval.providers import build_provider
from llm_judge_summeval.schemas import pairwise_schema, pointwise_schema
from summarization_finetune.data import read_jsonl, write_jsonl


def verify_frozen_judge(config: dict, root: Path) -> None:
    prompts = root / config["judge"]["prompts_path"]
    actual = hashlib.sha256(prompts.read_bytes()).hexdigest()
    if actual != config["judge"]["prompts_sha256"]:
        raise PermissionError("Frozen judge prompt hash changed")
    manifest = json.loads((root / config["judge"]["final_prompt_manifest_path"]).read_text(encoding="utf-8"))
    if manifest.get("prompts_sha256") != actual or manifest.get("model") != config["judge"]["primary_model"]:
        raise PermissionError("Frozen judge manifest mismatch")


def balanced_pair_order(study_seed: int, pair_id: str, left: dict, right: dict) -> tuple[dict, dict]:
    digest = hashlib.sha256(f"{study_seed}\0{pair_id}\0display-order".encode()).digest()
    return (left, right) if digest[0] % 2 == 0 else (right, left)


def score_pointwise(*, generations_path: Path, ledger_path: Path, provider_name: str,
                    config: dict, root: Path) -> dict:
    verify_frozen_judge(config, root)
    prompts = load_prompts(root / config["judge"]["prompts_path"])
    model = config["judge"]["primary_model"] if provider_name == "anthropic" else config["judge"]["secondary_model"]
    provider = build_provider(provider_name, model)
    counts = {"planned": 0, "success": 0, "failed": 0}
    for row in read_jsonl(generations_path):
        counts["planned"] += 1
        system, user = render_pointwise(prompts, row["source"], row["summary"])
        result = execute_request(
            ledger_path=ledger_path, provider=provider, provider_name=provider_name, model=model,
            kind="pointwise", item_id=f"{row['model']}:{row['article_id']}", system=system, user=user,
            schema=pointwise_schema(), metadata={"article_id": row["article_id"], "model_label": row["model"]},
        )
        counts["success" if result["status"] == "success" else "failed"] += 1
    return counts


def score_candidate_pairs(*, candidates_path: Path, ledger_path: Path, config: dict, root: Path) -> dict:
    verify_frozen_judge(config, root)
    prompts = load_prompts(root / config["judge"]["prompts_path"])
    provider = build_provider("anthropic", config["judge"]["primary_model"])
    by_article = {}
    for row in read_jsonl(candidates_path):
        by_article.setdefault(row["article_id"], []).append(row)
    counts = {"planned": 0, "success": 0, "failed": 0}
    for article_id in sorted(by_article):
        rows = sorted(by_article[article_id], key=lambda row: row["candidate_index"])
        for left, right in itertools.combinations(rows, 2):
            counts["planned"] += 1
            pair_id = f"{article_id}:c{left['candidate_index']}-c{right['candidate_index']}"
            left, right = balanced_pair_order(config["seed"], pair_id, left, right)
            system, user = render_pairwise(prompts, left["source"], left["summary"], right["summary"])
            result = execute_request(
                ledger_path=ledger_path, provider=provider, provider_name="anthropic",
                model=config["judge"]["primary_model"], kind="pairwise", item_id=pair_id,
                system=system, user=user, schema=pairwise_schema(),
                metadata={"article_id": article_id, "pair_id": pair_id,
                          "display_a_id": left["candidate_id"], "display_b_id": right["candidate_id"]},
            )
            counts["success" if result["status"] == "success" else "failed"] += 1
    return counts


def build_preferences(*, candidates_path: Path, ledger_path: Path, output_path: Path) -> dict:
    candidates = {row["candidate_id"]: row for row in read_jsonl(candidates_path)}
    successful = [row for row in load_latest(ledger_path).values() if row["status"] == "success"]
    preferences, excluded_conflict, excluded_tie = [], 0, 0
    for row in sorted(successful, key=lambda item: item["item_id"]):
        relevance = row["parsed"]["relevance"]["winner"]
        consistency = row["parsed"]["consistency"]["winner"]
        if "tie" in (relevance, consistency):
            excluded_tie += 1; continue
        if relevance != consistency:
            excluded_conflict += 1; continue
        chosen_id = row["metadata"]["display_a_id" if relevance == "A" else "display_b_id"]
        rejected_id = row["metadata"]["display_b_id" if relevance == "A" else "display_a_id"]
        chosen, rejected = candidates[chosen_id], candidates[rejected_id]
        preferences.append({"article_id": chosen["article_id"], "pair_id": row["metadata"]["pair_id"],
                            "prompt": f"Article:\n{chosen['source']}\n\nSummary:\n",
                            "source": chosen["source"],
                            "chosen": chosen["summary"], "rejected": rejected["summary"],
                            "chosen_id": chosen_id, "rejected_id": rejected_id})
    write_jsonl(output_path, preferences)
    return {"successful_judgments": len(successful), "preferences": len(preferences),
            "excluded_primary_tie": excluded_tie, "excluded_primary_conflict": excluded_conflict,
            "output": str(output_path)}
