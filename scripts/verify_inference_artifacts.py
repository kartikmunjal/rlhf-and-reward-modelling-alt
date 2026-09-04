#!/usr/bin/env python3
"""Fail closed on artifact, tokenizer, and frozen-protocol mismatches."""

import argparse, hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference_serving.data import verify_tokenizer_identity

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, default=Path("inference_serving/study_config.json"))
parser.add_argument("--sft", default="checkpoints/summarization_sft_v1/merged")
parser.add_argument("--dpo", default="checkpoints/summarization_dpo_v1/merged")
parser.add_argument("--draft", default="checkpoints/inference_serving_v1/gpt2_small_draft/model")
parser.add_argument("--require-draft", action="store_true")
parser.add_argument("--output", type=Path, default=Path("results/inference_serving_v1/artifact_manifest.json"))
args = parser.parse_args(); root = Path(__file__).resolve().parents[1]
config = json.loads(args.config.read_text(encoding="utf-8")); frozen = json.loads((root / "inference_serving/preregistration_manifest.json").read_text())
for row in frozen["files"].values():
    path = root / row["path"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]: raise SystemExit(f"Frozen artifact mismatch: {path}")
from transformers import AutoTokenizer
tokenizers = {"base": AutoTokenizer.from_pretrained(config["artifacts"]["base"]["model"], revision=config["artifacts"]["base"]["revision"]),
              "sft": AutoTokenizer.from_pretrained(args.sft), "dpo": AutoTokenizer.from_pretrained(args.dpo)}
if args.require_draft: tokenizers["draft"] = AutoTokenizer.from_pretrained(args.draft)
check = verify_tokenizer_identity(tokenizers)
def digest_tree(value):
    path = Path(value)
    if not path.exists(): return {"source": str(value), "kind": "remote_revision_pinned"}
    files = []
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        files.append({"path": str(item.relative_to(path)).replace("\\", "/"), "bytes": item.stat().st_size,
                      "sha256": hashlib.sha256(item.read_bytes()).hexdigest()})
    aggregate = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"source": str(path), "kind": "local_directory", "sha256": aggregate, "files": files}
payload = {"tokenizer_identity": check, "artifacts": {"base": digest_tree(config["artifacts"]["base"]["model"]),
           "sft": digest_tree(args.sft), "dpo": digest_tree(args.dpo)},
           "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest()}
if args.require_draft: payload["artifacts"]["draft"] = digest_tree(args.draft)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
