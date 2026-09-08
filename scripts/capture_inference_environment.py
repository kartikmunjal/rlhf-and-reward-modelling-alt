#!/usr/bin/env python3
"""Capture the immutable runtime record for the inference-serving study."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def run(*args: str) -> str:
    result = subprocess.run(args, check=True, text=True, capture_output=True)
    return result.stdout.strip()


def read_optional(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except (FileNotFoundError, PermissionError):
        return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/inference_serving_v1/environment_manifest.json"),
    )
    parser.add_argument("--vllm-python", type=Path, default=Path(".venv-vllm/bin/python"))
    parser.add_argument("--gptq-python", type=Path, default=Path(".venv-gptq/bin/python"))
    args = parser.parse_args()
    root = args.repo_root.resolve()

    artifacts = [
        root / "checkpoints/inference_serving_v1/gpt2_small_draft/model/model.safetensors",
        root / "checkpoints/summarization_sft_v1/merged/model.safetensors",
        root / "checkpoints/summarization_dpo_v1/merged/model.safetensors",
        root / "data/processed/inference_serving_v1/calibration_articles.jsonl",
        root / "data/processed/inference_serving_v1/pilot_articles.jsonl",
        root / "data/processed/inference_serving_v1/heldout_articles.jsonl",
        root / "data/processed/inference_serving_v1/partition_manifest.json",
        root / "inference_serving/preregistration_manifest.json",
        root / "inference_serving/protocol_amendment_001_manifest.json",
    ]
    missing = [str(path) for path in artifacts if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required artifacts missing: {missing}")

    def env_record(python: Path, imports: str) -> dict[str, object]:
        executable = root / python
        freeze = run(str(executable), "-m", "pip", "freeze", "--all")
        # GPTQModel emits an informational banner to stdout at import time; the
        # child process deliberately prints the machine-readable record last.
        versions = json.loads(run(str(executable), "-c", imports).splitlines()[-1])
        return {
            "python_executable": str(python),
            "versions": versions,
            "pip_freeze": freeze.splitlines(),
            "pip_freeze_sha256": hashlib.sha256(freeze.encode()).hexdigest(),
        }

    vllm_imports = (
        "import json,torch,vllm; print(json.dumps({'python':__import__('sys').version,"
        "'torch':torch.__version__,'torch_cuda':torch.version.cuda,'vllm':vllm.__version__}))"
    )
    gptq_imports = (
        "import json,torch,gptqmodel,torchvision; print(json.dumps({'python':__import__('sys').version,"
        "'torch':torch.__version__,'torch_cuda':torch.version.cuda,'gptqmodel':gptqmodel.__version__,"
        "'torchvision':torchvision.__version__}))"
    )
    gpu_csv = run(
        "nvidia-smi",
        "--query-gpu=name,uuid,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ).splitlines()
    manifest = {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "study": "inference_serving_v1",
        "git": {
            "commit": run("git", "-C", str(root), "rev-parse", "HEAD"),
            "branch": run("git", "-C", str(root), "branch", "--show-current"),
            "status_porcelain": run("git", "-C", str(root), "status", "--porcelain").splitlines(),
        },
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "kernel": platform.release(),
            "machine": platform.machine(),
            "os_release": read_optional("/etc/os-release"),
            "cgroup_cpu_max": read_optional("/sys/fs/cgroup/cpu.max"),
            "cgroup_memory_max": read_optional("/sys/fs/cgroup/memory.max"),
            "cgroup_cpuset": read_optional("/sys/fs/cgroup/cpuset.cpus.effective"),
            "runpod_env": {key: value for key, value in os.environ.items() if key.startswith("RUNPOD_")},
            "user_reported_template": "Runpod Pytorch 2.8.0",
            "template_digest_available": False,
        },
        "gpus": [
            dict(zip(("name", "uuid", "driver_version", "memory_mib"), map(str.strip, row.split(","))))
            for row in gpu_csv
        ],
        "environments": {
            "vllm": env_record(args.vllm_python, vllm_imports),
            "gptq": env_record(args.gptq_python, gptq_imports),
        },
        "artifacts": [file_record(path, root) for path in artifacts],
    }
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
