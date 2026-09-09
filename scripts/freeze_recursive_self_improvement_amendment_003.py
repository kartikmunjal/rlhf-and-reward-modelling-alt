#!/usr/bin/env python3
import argparse, hashlib, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "recursive_self_improvement" / "protocol_amendment_003_eval_capacity.json"
MANIFEST = ROOT / "recursive_self_improvement" / "protocol_amendment_003_manifest.json"
def expected(): return {"amendment_id": "recursive_self_improvement_v1_eval_capacity_003", "path": str(SOURCE.relative_to(ROOT)), "sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest()}
def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--verify",action="store_true"); args=parser.parse_args(); value=expected()
    if args.verify:
        if not MANIFEST.exists() or json.loads(MANIFEST.read_text()) != value: raise SystemExit("Amendment 003 mismatch")
        print("Amendment 003 verified")
    else:
        if MANIFEST.exists(): raise SystemExit("Refusing to overwrite amendment 003")
        MANIFEST.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n",encoding="utf-8"); print("Amendment 003 frozen")
if __name__ == "__main__": main()
