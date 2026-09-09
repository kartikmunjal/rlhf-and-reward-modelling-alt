#!/usr/bin/env python3
"""Freeze or verify the approved data-feasibility amendment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "recursive_self_improvement" / "protocol_amendment_001_data_feasibility.json"
MANIFEST = ROOT / "recursive_self_improvement" / "protocol_amendment_001_manifest.json"


def expected() -> dict:
    return {
        "amendment_id": "recursive_self_improvement_v1_data_001",
        "path": str(SOURCE.relative_to(ROOT)),
        "sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    value = expected()
    if args.verify:
        if json.loads(MANIFEST.read_text(encoding="utf-8")) != value:
            raise SystemExit("Amendment manifest mismatch")
        print("Amendment verified")
    else:
        current = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if current.get("sha256") != "TO_BE_FROZEN":
            raise SystemExit("Refusing to overwrite frozen amendment")
        MANIFEST.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("Amendment frozen")


if __name__ == "__main__":
    main()
