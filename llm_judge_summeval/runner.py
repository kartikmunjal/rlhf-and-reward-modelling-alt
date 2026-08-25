"""Build and execute development or held-out SummEval judge matrices."""

from __future__ import annotations

import json
from pathlib import Path

from llm_judge_summeval.data import load_heldout_inputs, load_prompt_development_rows
from llm_judge_summeval.ledger import execute_request
from llm_judge_summeval.prompts import load_prompts, render_pairwise, render_pointwise, verify_final_prompt_manifest
from llm_judge_summeval.providers import build_provider
from llm_judge_summeval.schemas import pairwise_schema, pointwise_schema


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def run_pointwise(*, phase: str, provider_name: str, config: dict, processed_dir: Path, prompts_path: Path,
                  final_prompt_manifest: Path, ledger_path: Path, limit: int | None = None, transport=None) -> dict:
    if phase == "dev":
        rows = load_prompt_development_rows(processed_dir)
    elif phase == "heldout":
        verify_final_prompt_manifest(prompts_path, final_prompt_manifest, config)
        rows = load_heldout_inputs(processed_dir)
    else:
        raise ValueError("phase must be dev or heldout")
    if limit is not None:
        rows = rows[:limit]
    judge = config["judges"]["primary" if provider_name == "anthropic" else "secondary"]
    provider = build_provider(provider_name, judge["model"], **({"transport": transport} if transport else {}))
    prompts, counts = load_prompts(prompts_path), {"planned": len(rows), "success": 0, "failed": 0}
    for row in rows:
        system, user = render_pointwise(prompts, row["source"], row["summary"])
        result = execute_request(ledger_path=ledger_path, provider=provider, provider_name=provider_name, model=judge["model"],
                                 kind="pointwise", item_id=row["summary_id"], system=system, user=user,
                                 schema=pointwise_schema(), metadata={"phase": phase, "article_id": row["article_id"], "summary_id": row["summary_id"]})
        counts["success" if result["status"] == "success" else "failed"] += 1
    return counts


def run_pairwise_heldout(*, config: dict, processed_dir: Path, prompts_path: Path, final_prompt_manifest: Path,
                         ledger_path: Path, limit: int | None = None, transport=None) -> dict:
    verify_final_prompt_manifest(prompts_path, final_prompt_manifest, config)
    inputs = {row["summary_id"]: row for row in load_heldout_inputs(processed_dir)}
    pairs = read_jsonl(processed_dir / "heldout_pairs.jsonl")
    requests = [(pair, order) for pair in pairs for order in ("ab", "ba")]
    if limit is not None:
        requests = requests[:limit]
    judge = config["judges"]["primary"]
    provider = build_provider("anthropic", judge["model"], **({"transport": transport} if transport else {}))
    prompts, counts = load_prompts(prompts_path), {"planned": len(requests), "success": 0, "failed": 0}
    for pair, order in requests:
        left_id, right_id = pair["summary_a_id"], pair["summary_b_id"]
        if order == "ba":
            left_id, right_id = right_id, left_id
        left, right = inputs[left_id], inputs[right_id]
        if not (left["article_id"] == right["article_id"] == pair["article_id"]):
            raise ValueError("Cross-article pair detected")
        system, user = render_pairwise(prompts, left["source"], left["summary"], right["summary"])
        metadata = {"phase": "heldout", "pair_id": pair["pair_id"], "article_id": pair["article_id"],
                    "order": order, "display_a_id": left_id, "display_b_id": right_id}
        result = execute_request(ledger_path=ledger_path, provider=provider, provider_name="anthropic", model=judge["model"],
                                 kind="pairwise", item_id=f"{pair['pair_id']}:{order}", system=system, user=user,
                                 schema=pairwise_schema(), metadata=metadata)
        counts["success" if result["status"] == "success" else "failed"] += 1
    return counts
