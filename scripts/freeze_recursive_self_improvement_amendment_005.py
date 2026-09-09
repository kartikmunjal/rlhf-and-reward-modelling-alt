#!/usr/bin/env python3
"""Freeze amendment 005 using the repository's canonical LF text hash."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from recursive_self_improvement.config import canonical_text_sha256

SOURCE = ROOT / "recursive_self_improvement/protocol_amendment_005_rounds_and_evaluators.json"
MANIFEST = ROOT / "recursive_self_improvement/protocol_amendment_005_manifest.json"

payload = {
    "amendment_id": "recursive_self_improvement_v1_rounds_and_evaluators_005",
    "path": str(SOURCE.relative_to(ROOT)),
    "sha256": canonical_text_sha256(SOURCE),
}
MANIFEST.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(MANIFEST)
