"""Load the frozen config with explicit, hash-verified amendments."""

import hashlib
import json
from pathlib import Path


def load_effective_config(root: Path) -> dict:
    module = root / "recursive_self_improvement"
    config = json.loads((module / "study_config.json").read_text(encoding="utf-8"))
    amendment_path = module / "protocol_amendment_001_data_feasibility.json"
    manifest = json.loads((module / "protocol_amendment_001_manifest.json").read_text(encoding="utf-8"))
    if hashlib.sha256(amendment_path.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("Data amendment hash mismatch")
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    config["data"].update(amendment["approved_partition"])
    config["applied_amendments"] = [amendment["amendment_id"]]
    return config
