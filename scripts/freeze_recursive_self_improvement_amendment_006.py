#!/usr/bin/env python3
"""Freeze the reward execution correction with canonical LF hashing."""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from recursive_self_improvement.config import canonical_text_sha256

SOURCE = ROOT / "recursive_self_improvement/protocol_amendment_006_reward_execution_correction.json"
MANIFEST = ROOT / "recursive_self_improvement/protocol_amendment_006_manifest.json"
MANIFEST.write_text(json.dumps({
    "amendment_id": "recursive_self_improvement_v1_reward_execution_correction_006",
    "path": str(SOURCE.relative_to(ROOT)),
    "sha256": canonical_text_sha256(SOURCE),
}, indent=2) + "\n", encoding="utf-8")
print(MANIFEST)
