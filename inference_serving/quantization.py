"""Locked GPTQ quantization with explicit compatibility failure records."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path


def quantize_gptq(model_path: str, output_dir: Path, calibration_texts: list[str], config: dict) -> dict:
    from gptqmodel import GPTQModel, QuantizeConfig
    import gptqmodel
    settings = config["stage2"]
    if gptqmodel.__version__ != settings["gptqmodel_version"]:
        raise RuntimeError(f"Locked GPTQModel {settings['gptqmodel_version']} required, found {gptqmodel.__version__}")
    quant_config = QuantizeConfig(bits=settings["bits"], group_size=settings["group_size"],
                                  sym=settings["symmetric"], desc_act=settings["desc_act"])
    started = time.time(); output_dir.mkdir(parents=True, exist_ok=True)
    try:
        model = GPTQModel.load(model_path, quantize_config=quant_config)
        model.quantize(calibration_texts, batch_size=1)
        model.save(output_dir)
        status, error = "success", None
    except Exception as exc:
        status, error = "incompatible", f"{type(exc).__name__}: {exc}"
    manifest = {"method": "gptq", "status": status, "error": error, "source_model": model_path,
                "settings": {key: settings[key] for key in ("bits", "group_size", "symmetric", "desc_act")},
                "calibration_examples": len(calibration_texts), "runtime_seconds": time.time() - started,
                "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
    (output_dir / "quantization_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if status != "success":
        raise RuntimeError(error)
    return manifest
