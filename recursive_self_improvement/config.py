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
    second_path = module / "protocol_amendment_002_unique_prompts.json"
    second_manifest = json.loads((module / "protocol_amendment_002_manifest.json").read_text(encoding="utf-8"))
    if hashlib.sha256(second_path.read_bytes()).hexdigest() != second_manifest["sha256"]:
        raise ValueError("Unique-prompt amendment hash mismatch")
    second = json.loads(second_path.read_text(encoding="utf-8"))
    config["data"].update(second["approved_partition"])
    config["data"]["duplicate_resolution"] = second["duplicate_resolution"]
    third_path = module / "protocol_amendment_003_eval_capacity.json"
    third_manifest = json.loads((module / "protocol_amendment_003_manifest.json").read_text(encoding="utf-8"))
    if hashlib.sha256(third_path.read_bytes()).hexdigest() != third_manifest["sha256"]:
        raise ValueError("Evaluation-capacity amendment hash mismatch")
    third = json.loads(third_path.read_text(encoding="utf-8"))
    config["data"].update(third["approved_change"])
    fourth_path = module / "protocol_amendment_004_hybrid_lora.json"
    fourth_manifest = json.loads((module / "protocol_amendment_004_manifest.json").read_text(encoding="utf-8"))
    if hashlib.sha256(fourth_path.read_bytes()).hexdigest() != fourth_manifest["sha256"]:
        raise ValueError("Hybrid-LoRA amendment hash mismatch")
    fourth = json.loads(fourth_path.read_text(encoding="utf-8"))
    config["peft"] = fourth["parameter_efficient_method"]
    config["hybrid_execution"] = fourth["hybrid_execution"]
    config["smoke_gate"] = fourth["smoke_gate"]
    config["applied_amendments"] = [amendment["amendment_id"], second["amendment_id"], third["amendment_id"], fourth["amendment_id"]]
    return config
