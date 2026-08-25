#!/usr/bin/env python3
"""Generate the Stage-0 integrity manifest from preregistered source bytes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCES = ("preregistration.md", "study_config.json", "prompts.json")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    manifest = {
        "stage": 0,
        "status": "frozen_before_data_or_results",
        "files": {name: {"sha256": sha256(ROOT / name)} for name in SOURCES},
    }
    (ROOT / "stage0_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("Wrote", ROOT / "stage0_manifest.json")


if __name__ == "__main__":
    main()

