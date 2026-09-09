#!/usr/bin/env python3
"""Create or verify the immutable preregistration manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FILES = (
    ROOT / "recursive_self_improvement" / "study_config.json",
    ROOT / "recursive_self_improvement" / "preregistration.md",
)
MANIFEST = ROOT / "recursive_self_improvement" / "preregistration_manifest.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload() -> dict:
    return {
        "schema_version": 1,
        "study_id": "recursive_self_improvement_v1",
        "files": {
            path.name: {"path": str(path.relative_to(ROOT)), "sha256": digest(path)}
            for path in FILES
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    expected = payload()
    if args.verify:
        if not MANIFEST.exists() or json.loads(MANIFEST.read_text(encoding="utf-8")) != expected:
            raise SystemExit("Preregistration manifest mismatch")
        print("Preregistration verified")
        return
    if MANIFEST.exists():
        raise SystemExit("Refusing to overwrite existing preregistration manifest")
    MANIFEST.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
