"""Greedy sandboxed evaluation for base or trained verifier policies."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from verifier_reward_hacking.evaluation import evaluate_visible_hidden
from verifier_reward_hacking.io import append_jsonl, read_jsonl, sha256_file
from verifier_reward_hacking.prompting import build_prompt
from verifier_reward_hacking.sandbox import SandboxConfig
from verifier_reward_hacking.schema import CodeTask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-label", required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--split", choices=["dev", "test", "test_humaneval", "all_test"], required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/verifier_reward_hacking_v1/final"))
    parser.add_argument("--sandbox-gates", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--runtime", choices=["docker", "podman"], default="docker")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import torch
    import transformers
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    study = json.loads((ROOT / "verifier_reward_hacking/study_config.json").read_text())
    data_manifest_path = args.data_dir / "data_manifest.json"
    data_manifest = json.loads(data_manifest_path.read_text())
    task_path = args.data_dir / "tasks_private.jsonl"
    if sha256_file(task_path) != data_manifest["private_task_sha256"]:
        raise RuntimeError("Final task hash mismatch")
    gates = json.loads(args.sandbox_gates.read_text())
    if gates.get("status") != "pass" or gates.get("image") != args.image:
        raise RuntimeError("Sandbox gates are not passing for this image")
    accepted = {args.split} if args.split != "all_test" else {"test", "test_humaneval"}
    task_rows = [row for row in read_jsonl(task_path) if row["split"] in accepted]
    tasks = [CodeTask.from_dict({k: v for k, v in row.items() if k != "split"}) for row in task_rows]
    model_cfg = study["base_model"]
    tokenizer_source = args.adapter if args.adapter else model_cfg["id"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, revision=None if args.adapter else model_cfg["revision"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_cfg["id"], revision=model_cfg["revision"], torch_dtype=torch.float16)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model = model.to(args.device).eval()
    sandbox = SandboxConfig(runtime=args.runtime, image=args.image)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ledger = args.output_dir / "predictions_private.jsonl"
    completed = {row["task_id"] for row in read_jsonl(ledger)} if ledger.exists() else set()
    started = time.perf_counter()
    for index, task in enumerate(tasks):
        if task.task_id in completed:
            continue
        rendered = tokenizer.apply_chat_template([{"role": "user", "content": build_prompt(task, "v0")}], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(rendered, return_tensors="pt", truncation=False).to(args.device)
        if inputs["input_ids"].shape[1] > study["training"]["max_prompt_tokens"]:
            raise RuntimeError(f"Prompt exceeds frozen maximum for {task.task_id}")
        with torch.inference_mode():
            output = model.generate(**inputs, do_sample=False, max_new_tokens=study["training"]["max_completion_tokens"], pad_token_id=tokenizer.pad_token_id)
        completion = tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        evaluation = evaluate_visible_hidden(completion, task, sandbox)
        append_jsonl(ledger, {"task_id": task.task_id, "task_hash": task.task_hash, "source": task.source, "family": task.family, "completion": completion, **evaluation})
        print(f"eval {index + 1}/{len(tasks)} {task.task_id} hidden={evaluation['hidden']['passed']}", flush=True)
    rows = read_jsonl(ledger)
    if len(rows) != len(tasks) or {row["task_id"] for row in rows} != {task.task_id for task in tasks}:
        raise RuntimeError("Evaluation ledger incomplete or duplicated")
    metrics = {
        key: sum(int(row[key]["passed"] if key in {"visible", "hidden"} else row[key]) for row in rows) / len(rows)
        for key in ("visible", "hidden", "exploit", "gap")
    }
    manifest = {
        "study_id": study["study_id"], "status": "complete", "policy_label": args.policy_label,
        "adapter": str(args.adapter) if args.adapter else None, "split": args.split,
        "task_count": len(tasks), "data_manifest_sha256": sha256_file(data_manifest_path),
        "sandbox_gate_sha256": sha256_file(args.sandbox_gates), "sandbox_image": args.image,
        "predictions_sha256": sha256_file(ledger), "descriptive_metrics": metrics,
        "elapsed_seconds": time.perf_counter() - started, "torch_version": torch.__version__,
        "transformers_version": transformers.__version__, "python": platform.python_version(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }
    (args.output_dir / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
