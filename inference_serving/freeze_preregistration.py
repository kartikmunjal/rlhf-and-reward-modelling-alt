#!/usr/bin/env python3
"""Freeze inference-serving protocols and inherited evaluation artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    config = json.loads((MODULE / "study_config.json").read_text(encoding="utf-8"))
    files = {
        "study_config": MODULE / "study_config.json",
        "preregistration": MODULE / "preregistration.md",
        "frozen_judge_prompts": ROOT / "llm_judge_summeval/prompts.json",
        "frozen_judge_manifest": ROOT / "llm_judge_summeval/final_prompt_manifest.json",
        "source_data_manifest": ROOT / "data/processed/summarization_finetune_v1/data_manifest.json",
        "sft_run_manifest": ROOT / "results/summarization_finetune_v1/manifests/sft_run_manifest.json",
        "dpo_run_manifest": ROOT / "results/summarization_finetune_v1/manifests/dpo_run_manifest.json"
    }
    if sha256(files["frozen_judge_prompts"]) != config["quality_evaluation"]["prompts_sha256"]:
        raise SystemExit("Frozen judge prompt hash mismatch")
    manifest = {
        "study_id": config["study_id"],
        "status": "frozen_before_execution",
        "files": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for name, path in files.items()
        }
    }
    (MODULE / "preregistration_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("Wrote", MODULE / "preregistration_manifest.json")


if __name__ == "__main__":
    main()
