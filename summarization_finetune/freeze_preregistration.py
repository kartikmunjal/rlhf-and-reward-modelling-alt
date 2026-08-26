#!/usr/bin/env python3
"""Hash the approved SFT/DPO protocols and immutable judge artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    paths = {
        "study_config": MODULE / "study_config.json",
        "sft_preregistration": MODULE / "sft_preregistration.md",
        "dpo_preregistration": MODULE / "dpo_preregistration.md",
        "frozen_judge_prompts": ROOT / "llm_judge_summeval/prompts.json",
        "frozen_judge_manifest": ROOT / "llm_judge_summeval/final_prompt_manifest.json",
    }
    config = json.loads(paths["study_config"].read_text(encoding="utf-8"))
    if sha(paths["frozen_judge_prompts"]) != config["judge"]["prompts_sha256"]:
        raise SystemExit("Frozen judge prompt hash does not match the approved config")
    manifest = {
        "status": "locked_before_data_or_training",
        "files": {name: {"path": str(path.relative_to(ROOT)), "sha256": sha(path)} for name, path in paths.items()},
    }
    (MODULE / "preregistration_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("Wrote", MODULE / "preregistration_manifest.json")


if __name__ == "__main__":
    main()
