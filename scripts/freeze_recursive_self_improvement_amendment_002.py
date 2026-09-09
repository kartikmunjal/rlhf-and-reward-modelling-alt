#!/usr/bin/env python3
"""Create or verify the immutable unique-prompt feasibility amendment."""

import argparse, hashlib, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "recursive_self_improvement" / "protocol_amendment_002_unique_prompts.json"
MANIFEST = ROOT / "recursive_self_improvement" / "protocol_amendment_002_manifest.json"

def expected():
    return {"amendment_id": "recursive_self_improvement_v1_unique_prompts_002", "path": str(SOURCE.relative_to(ROOT)), "sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest()}

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--verify", action="store_true"); args = parser.parse_args()
    value = expected()
    if args.verify:
        if not MANIFEST.exists() or json.loads(MANIFEST.read_text()) != value: raise SystemExit("Amendment 002 mismatch")
        print("Amendment 002 verified")
    else:
        if MANIFEST.exists(): raise SystemExit("Refusing to overwrite amendment 002")
        MANIFEST.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("Amendment 002 frozen")

if __name__ == "__main__": main()
