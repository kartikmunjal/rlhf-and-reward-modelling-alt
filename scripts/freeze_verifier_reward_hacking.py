"""Hash-freeze Extension 16 before task generation or model execution."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "verifier_reward_hacking"
OUTPUT = MODULE / "preregistration_manifest.json"
FILES = [
    MODULE / "PREREGISTRATION.md",
    MODULE / "study_config.json",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifest() -> dict:
    config = json.loads((MODULE / "study_config.json").read_text(encoding="utf-8"))
    if config["status"] != "preregistered_not_run":
        raise ValueError("Study status must remain preregistered_not_run at freeze time")
    return {
        "study_id": config["study_id"],
        "status": "frozen_before_data_or_results",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": {
            str(path.relative_to(ROOT)): {"sha256": sha256(path)} for path in FILES
        },
        "external_sources": {
            "human_eval": {
                "revision": config["tasks"]["human_eval_revision"],
                "license": config["tasks"]["human_eval_license"],
                "upstream": config["tasks"]["human_eval_upstream"]
            },
            "base_model": {
                "id": config["base_model"]["id"],
                "revision": config["base_model"]["revision"]
            }
        }
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite frozen manifest: {OUTPUT}")
    OUTPUT.write_text(
        json.dumps(build_manifest(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(OUTPUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
