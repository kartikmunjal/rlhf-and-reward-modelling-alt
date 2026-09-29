"""Train one manifest-locked GRPO condition/seed with sandboxed rewards."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from verifier_reward_hacking.io import read_jsonl, sha256_file
from verifier_reward_hacking.prompting import build_prompt
from verifier_reward_hacking.rewards import SandboxedVerifierReward
from verifier_reward_hacking.sandbox import SandboxConfig
from verifier_reward_hacking.schema import CodeTask


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def directory_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(str(item.relative_to(path)).encode())
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["C1", "C2", "C3", "C4"], required=True)
    parser.add_argument("--seed", type=int, choices=[2025, 2026, 2027], required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/verifier_reward_hacking_v1/final"))
    parser.add_argument("--audits", type=Path, default=Path("data/processed/verifier_reward_hacking_v1/candidate_audits.jsonl"))
    parser.add_argument("--sandbox-gates", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--runtime", choices=["docker", "podman"], default="docker")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--run-kind", choices=["pilot", "dev_tuning", "confirmatory"], required=True)
    args = parser.parse_args()

    import torch
    import transformers
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer, set_seed
    from trl import GRPOConfig, GRPOTrainer

    study = json.loads((ROOT / "verifier_reward_hacking/study_config.json").read_text())
    if args.run_kind == "confirmatory" and args.steps != study["training"]["optimizer_steps"]:
        raise RuntimeError("Confirmatory optimizer steps must equal the frozen budget")
    if args.run_kind == "dev_tuning" and (args.seed != 2025 or args.steps != 50):
        raise RuntimeError("Dev tuning is locked to seed 2025 and 50 steps")
    data_manifest_path = args.data_dir / "data_manifest.json"
    data_manifest = json.loads(data_manifest_path.read_text())
    task_path = args.data_dir / "tasks_private.jsonl"
    if sha256_file(task_path) != data_manifest["private_task_sha256"]:
        raise RuntimeError("Final private task hash mismatch")
    gates = json.loads(args.sandbox_gates.read_text())
    if gates.get("status") != "pass" or gates.get("image") != args.image:
        raise RuntimeError("Sandbox gates are not passing for this image")
    all_rows = read_jsonl(task_path)
    tasks = {row["task_id"]: CodeTask.from_dict({k: v for k, v in row.items() if k != "split"}) for row in all_rows if row["split"] == "train"}
    audits = {row["task_id"]: row for row in read_jsonl(args.audits)}
    model_cfg, train_cfg = study["base_model"], study["training"]
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["id"], revision=model_cfg["revision"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    rows = []
    for task in tasks.values():
        # The prompt is held constant across conditions. Hardened cases affect reward only.
        text = tokenizer.apply_chat_template([{"role": "user", "content": build_prompt(task, "v0")}], tokenize=False, add_generation_prompt=True)
        length = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        if length > train_cfg["max_prompt_tokens"]:
            raise RuntimeError(f"Prompt exceeds frozen limit: {task.task_id} has {length} tokens")
        rows.append({"prompt": text, "task_id": task.task_id})
    dataset = Dataset.from_list(rows)
    sandbox = SandboxConfig(runtime=args.runtime, image=args.image)
    reward = SandboxedVerifierReward(tasks, audits, args.condition, sandbox)
    peft = LoraConfig(r=model_cfg["lora_rank"], lora_alpha=model_cfg["lora_alpha"], lora_dropout=model_cfg["lora_dropout"], target_modules=model_cfg["lora_targets"], task_type="CAUSAL_LM")
    training_args = GRPOConfig(
        output_dir=str(args.output_dir), max_steps=args.steps, per_device_train_batch_size=4,
        learning_rate=args.learning_rate, adam_beta1=0.9, adam_beta2=0.95, max_grad_norm=1.0,
        num_generations=train_cfg["generations_per_group"], num_iterations=1,
        max_prompt_length=train_cfg["max_prompt_tokens"], max_completion_length=train_cfg["max_completion_tokens"],
        beta=args.beta, epsilon=0.2, temperature=train_cfg["temperature"], top_p=train_cfg["top_p"],
        fp16=True, gradient_checkpointing=True, logging_steps=1, save_strategy="steps", save_steps=50,
        save_total_limit=None, report_to="none", seed=args.seed, remove_unused_columns=False,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "run_manifest.json"
    manifest = {
        "study_id": study["study_id"], "status": "running", "run_kind": args.run_kind,
        "condition": args.condition, "seed": args.seed, "git_commit": git_commit(),
        "base_model": model_cfg, "data_manifest_sha256": sha256_file(data_manifest_path),
        "sandbox_gate_sha256": sha256_file(args.sandbox_gates), "sandbox_image": args.image,
        "optimizer_steps_planned": args.steps, "learning_rate": args.learning_rate, "beta": args.beta,
        "prompt_policy": "V0 visible prompt fixed across all conditions; reward suite varies",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    started = time.perf_counter()
    try:
        set_seed(args.seed)
        torch.cuda.reset_peak_memory_stats()
        trainer = GRPOTrainer(model=model_cfg["id"], reward_funcs=reward, args=training_args, train_dataset=dataset, processing_class=tokenizer, peft_config=peft)
        trainer.train()
        trainer.save_model(str(args.output_dir / "final"))
        tokenizer.save_pretrained(args.output_dir / "final")
        trajectory_path = args.output_dir / "trajectory.json"
        trajectory_path.write_text(json.dumps(trainer.state.log_history, indent=2) + "\n")
        checkpoints = sorted(path for path in args.output_dir.glob("checkpoint-*") if path.is_dir())
        observed_step = int(trainer.state.global_step)
        if observed_step != args.steps:
            raise RuntimeError(f"Observed {observed_step} optimizer steps, expected {args.steps}")
        manifest.update({
            "status": "complete", "optimizer_steps_observed": observed_step,
            "training_seconds": time.perf_counter() - started,
            "peak_gpu_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
            "gpu": torch.cuda.get_device_name(0), "torch_version": torch.__version__,
            "transformers_version": transformers.__version__, "python": platform.python_version(),
            "trajectory_sha256": sha256_file(trajectory_path),
            "checkpoint_hashes": {path.name: directory_hash(path) for path in checkpoints},
            "final_adapter_sha256": directory_hash(args.output_dir / "final"),
        })
    except BaseException as exc:
        manifest.update({"status": "failed", "failure_type": type(exc).__name__, "failure": str(exc), "traceback": traceback.format_exc(), "elapsed_seconds": time.perf_counter() - started})
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        raise
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
