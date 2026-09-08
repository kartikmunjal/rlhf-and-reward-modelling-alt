#!/usr/bin/env python3
"""Hash every derived model file and validate successful GPTQ manifests."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DERIVED_ROOT = ROOT / "checkpoints/inference_serving_v1/gptq"
OUTPUT = ROOT / "results/inference_serving_v1/derived_artifact_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    records = []
    for target in ("base", "sft", "dpo"):
        directory = DERIVED_ROOT / target
        quantization = json.loads((directory / "quantization_manifest.json").read_text())
        if quantization.get("status") != "success":
            raise RuntimeError(f"{target} quantization did not succeed")
        files = []
        for path in sorted(item for item in directory.rglob("*") if item.is_file()):
            files.append({
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
        records.append({"target": target, "quantization": quantization, "files": files})
    payload = {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": records,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
