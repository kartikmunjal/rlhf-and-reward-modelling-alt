"""Prompt rendering and content-addressed freeze verification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def load_prompts(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def render_pointwise(prompts: dict, source: str, summary: str) -> tuple[str, str]:
    rubric = "\n".join(f"{axis.upper()}: {text}" for axis, text in prompts["shared_rubric"].items())
    user = prompts["pointwise_user_template"].replace("{source}", source).replace("{summary}", summary).replace("{rubric}", rubric)
    return prompts["pointwise_system"], user


def render_pairwise(prompts: dict, source: str, summary_a: str, summary_b: str) -> tuple[str, str]:
    rubric = "\n".join(f"{axis.upper()}: {text}" for axis, text in prompts["shared_rubric"].items())
    user = prompts["pairwise_user_template"].replace("{source}", source).replace("{summary_a}", summary_a)
    user = user.replace("{summary_b}", summary_b).replace("{rubric}", rubric)
    return prompts["pairwise_system"], user


def verify_final_prompt_manifest(prompts_path: Path, manifest_path: Path, config: dict | None = None) -> dict:
    if not manifest_path.is_file():
        raise PermissionError("Held-out execution blocked: final prompt manifest does not exist")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual = hashlib.sha256(prompts_path.read_bytes()).hexdigest()
    if manifest.get("prompts_sha256") != actual or manifest.get("status") != "frozen_after_dev_before_heldout":
        raise PermissionError("Held-out execution blocked: final prompt hash mismatch")
    if config is not None:
        if manifest.get("config_sha256") != canonical_sha256(config):
            raise PermissionError("Held-out execution blocked: registered configuration hash mismatch")
        primary = config["judges"]["primary"]
        if manifest.get("provider") != primary["provider"] or manifest.get("model") != primary["model"]:
            raise PermissionError("Held-out execution blocked: primary model mismatch")
    return manifest
