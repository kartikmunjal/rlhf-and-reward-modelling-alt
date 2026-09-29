"""Run the preregistered V0-only base-model task eligibility screen."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from verifier_reward_hacking.evaluation import evaluate_completion
from verifier_reward_hacking.io import append_jsonl, read_jsonl, sha256_file
from verifier_reward_hacking.prompting import build_prompt
from verifier_reward_hacking.sandbox import SandboxConfig
from verifier_reward_hacking.schema import CodeTask


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, default=Path("data/processed/verifier_reward_hacking_v1/candidate_tasks_private.jsonl"))
    parser.add_argument("--candidate-manifest", type=Path, default=Path("data/processed/verifier_reward_hacking_v1/candidate_manifest.json"))
    parser.add_argument("--sandbox-gates", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--runtime", choices=["docker", "podman"], default="docker")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/verifier_reward_hacking_v1/base_qa"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    study = json.loads((ROOT / "verifier_reward_hacking/study_config.json").read_text())
    candidate = json.loads(args.candidate_manifest.read_text())
    gates = json.loads(args.sandbox_gates.read_text())
    if gates.get("status") != "pass" or gates.get("image") != args.image:
        raise RuntimeError("Sandbox gate manifest is not passing for the requested image")
    expected_sha = candidate["task_file"]["sha256"]
    if sha256_file(args.tasks) != expected_sha:
        raise RuntimeError("Candidate task file hash mismatch")

    tasks = [CodeTask.from_dict(row) for row in read_jsonl(args.tasks) if row["source"] == "programmatic_v1"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ledger = args.output_dir / "samples_private.jsonl"
    complete = {(row["task_id"], row["sample_index"]) for row in read_jsonl(ledger)} if ledger.exists() else set()

    model_cfg = study["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["id"], revision=model_cfg["revision"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["id"], revision=model_cfg["revision"], torch_dtype=torch.float16
    ).to(args.device).eval()
    sandbox = SandboxConfig(runtime=args.runtime, image=args.image)
    base_seed = study["tasks"]["split_seed"]

    for task_number, task in enumerate(tasks):
        pending = [index for index in range(8) if (task.task_id, index) not in complete]
        if not pending:
            continue
        prompt = build_prompt(task, "v0")
        messages = [{"role": "user", "content": prompt}]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(rendered, return_tensors="pt", truncation=True, max_length=study["training"]["max_prompt_tokens"]).to(args.device)
        for sample_index in pending:
            seed = base_seed + task_number * 8 + sample_index
            set_seed(seed)
            with torch.inference_mode():
                output = model.generate(
                    **inputs, do_sample=True, temperature=study["training"]["temperature"],
                    top_p=study["training"]["top_p"], max_new_tokens=study["training"]["max_completion_tokens"],
                    pad_token_id=tokenizer.pad_token_id,
                )
            completion = tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            evaluation = evaluate_completion(completion, task, task.visible_tests, sandbox)
            append_jsonl(ledger, {
                "task_id": task.task_id, "task_hash": task.task_hash, "sample_index": sample_index,
                "seed": seed, "completion": completion, "v0": evaluation,
            })
            print(f"base_qa {task_number + 1}/{len(tasks)} sample={sample_index} pass={evaluation['passed']}", flush=True)

    rows = read_jsonl(ledger)
    expected_keys = {(task.task_id, index) for task in tasks for index in range(8)}
    actual_keys = {(row["task_id"], row["sample_index"]) for row in rows}
    if actual_keys != expected_keys or len(rows) != len(expected_keys):
        raise RuntimeError("Base QA ledger is incomplete or contains duplicates")
    counts = {task.task_id: 0 for task in tasks}
    for row in rows:
        counts[row["task_id"]] += int(row["v0"]["passed"])
    summary = args.output_dir / "summary_private.jsonl"
    summary.write_text("".join(json.dumps({"task_id": key, "v0_passes": value, "samples": 8, "v0_pass_rate": value / 8}, sort_keys=True) + "\n" for key, value in sorted(counts.items())), encoding="utf-8")
    manifest = {
        "study_id": study["study_id"], "status": "complete", "git_commit": git_commit(),
        "candidate_task_sha256": expected_sha, "sandbox_gate_sha256": sha256_file(args.sandbox_gates),
        "sandbox_image": args.image, "model": model_cfg, "sample_count": len(rows),
        "task_count": len(tasks), "samples_per_task": 8, "ledger_sha256": sha256_file(ledger),
        "summary_sha256": sha256_file(summary), "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__, "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
