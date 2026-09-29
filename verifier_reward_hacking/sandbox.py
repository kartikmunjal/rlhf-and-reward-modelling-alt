"""Rootless container runner for untrusted model-generated Python.

Expected values and the test suite are retained by the host. Each container sees
only one invocation's inputs and returns a JSON-serializable result.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from typing import Any, Iterable

from .schema import TestCase


RUNNER = r'''
import json, signal, sys
payload = json.loads(sys.stdin.read())
source = payload.pop("source")
entry = payload.pop("entry_point")
cases = payload.pop("cases")
per_test_timeout = float(payload.pop("per_test_timeout"))
namespace = {"__name__": "__candidate__"}

class PerTestTimeout(Exception):
    pass

def on_timeout(signum, frame):
    raise PerTestTimeout()

signal.signal(signal.SIGALRM, on_timeout)
try:
    exec(compile(source, "<candidate>", "exec"), namespace, namespace)
    function = namespace[entry]
    results = []
    for case in cases:
        try:
            signal.setitimer(signal.ITIMER_REAL, per_test_timeout)
            results.append({"status": "ok", "result": function(*case["args"], **case["kwargs"])})
        except PerTestTimeout:
            results.append({"status": "timeout"})
        except BaseException as exc:
            results.append({"status": "error", "error_type": type(exc).__name__})
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
    print(json.dumps({"status": "ok", "results": results}, allow_nan=False))
except BaseException as exc:
    print(json.dumps({"status": "error", "error_type": type(exc).__name__}))
'''.strip()


@dataclass(frozen=True)
class SandboxConfig:
    runtime: str = "docker"
    image: str = "python:3.11.10-slim@sha256:UNRESOLVED"
    cpu_limit: float = 1.0
    memory_mib: int = 512
    process_limit: int = 64
    per_test_timeout_seconds: float = 2.0
    per_sample_timeout_seconds: float = 10.0
    max_source_bytes: int = 32_768
    max_output_bytes: int = 65_536

    def validate(self) -> None:
        pinned = "@sha256:" in self.image or self.image.startswith("sha256:")
        if "UNRESOLVED" in self.image or not pinned:
            raise ValueError("Container image must be pinned by resolved sha256 digest")
        if self.runtime not in {"docker", "podman"}:
            raise ValueError("runtime must be docker or podman")


@dataclass(frozen=True)
class SandboxResult:
    passed: bool
    status: str
    actual: Any = None
    error_type: str | None = None


def container_command(config: SandboxConfig) -> list[str]:
    config.validate()
    return [
        config.runtime, "run", "--rm", "--interactive",
        "--network", "none", "--read-only", "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges", "--pids-limit", str(config.process_limit),
        "--cpus", str(config.cpu_limit), "--memory", f"{config.memory_mib}m",
        "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=16m",
        "--user", "65534:65534", config.image,
        "python", "-I", "-c", RUNNER,
    ]


def run_suite(source: str, entry_point: str, cases: Iterable[TestCase], config: SandboxConfig) -> list[SandboxResult]:
    cases = list(cases)
    if len(source.encode("utf-8")) > config.max_source_bytes:
        return [SandboxResult(False, "source_limit") for _ in cases]
    payload = json.dumps({
        "source": source,
        "entry_point": entry_point,
        "cases": [{"args": case.args, "kwargs": case.kwargs} for case in cases],
        "per_test_timeout": config.per_test_timeout_seconds,
    }, allow_nan=False).encode("utf-8")
    try:
        completed = subprocess.run(
            container_command(config), input=payload, capture_output=True,
            timeout=config.per_sample_timeout_seconds, check=False,
        )
    except subprocess.TimeoutExpired:
        return [SandboxResult(False, "sample_timeout") for _ in cases]
    if len(completed.stdout) > config.max_output_bytes or len(completed.stderr) > config.max_output_bytes:
        return [SandboxResult(False, "output_limit") for _ in cases]
    try:
        row = json.loads(completed.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return [SandboxResult(False, "container_error") for _ in cases]
    if row.get("status") != "ok":
        return [SandboxResult(False, "candidate_error", error_type=row.get("error_type")) for _ in cases]
    raw_results = row.get("results", [])
    if len(raw_results) != len(cases):
        return [SandboxResult(False, "container_error") for _ in cases]
    results = []
    for case, raw in zip(cases, raw_results):
        if raw.get("status") == "timeout":
            results.append(SandboxResult(False, "test_timeout"))
            continue
        if raw.get("status") != "ok":
            results.append(SandboxResult(False, "candidate_error", error_type=raw.get("error_type")))
            continue
        actual = raw.get("result")
        passed = actual == case.expected and type(actual) is type(case.expected)
        results.append(SandboxResult(passed, "pass" if passed else "wrong_answer", actual=actual))
    return results


def run_case(source: str, entry_point: str, case: TestCase, config: SandboxConfig) -> SandboxResult:
    return run_suite(source, entry_point, (case,), config)[0]
