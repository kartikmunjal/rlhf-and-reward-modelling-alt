#!/usr/bin/env python3
"""Create a fail-closed hash manifest for platform amendment 001."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AMENDMENT = ROOT / "inference_serving/protocol_amendment_001_hybrid.json"
OUTPUT = ROOT / "inference_serving/protocol_amendment_001_manifest.json"

if OUTPUT.exists():
    raise SystemExit(f"Refusing to overwrite frozen amendment manifest: {OUTPUT}")
payload = json.loads(AMENDMENT.read_text(encoding="utf-8"))
canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
manifest = {
    "amendment_id": payload["amendment_id"],
    "file": str(AMENDMENT.relative_to(ROOT)).replace("\\", "/"),
    "canonical_sha256": hashlib.sha256(canonical).hexdigest(),
    "bytes_sha256": hashlib.sha256(AMENDMENT.read_bytes()).hexdigest(),
}
OUTPUT.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(manifest, indent=2))
