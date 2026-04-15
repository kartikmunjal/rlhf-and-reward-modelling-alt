"""
Code execution tools for Extension 14: Code Execution Agent.

Provides a sandboxed Python executor: runs code in an isolated subprocess,
captures stdout+stderr, enforces a 10-second timeout. Safe for local
benchmarking — no network access, no persistent filesystem changes.

The Tool subclass here uses 'code' as the input parameter name (vs 'query'
in the base Tool class) so the Anthropic model receives the correct schema.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from typing import Callable, Dict, List, Tuple

from eval.tools import Tool


# ── Sandboxed executor ─────────────────────────────────────────────────────────

def _sandbox_exec(code: str, timeout: int = 10) -> str:
    """Run Python code in a subprocess; return stdout + stderr combined."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        fname = f.name
    try:
        result = subprocess.run(
            [sys.executable, fname],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        out = result.stdout or ""
        err = (result.stderr or "").strip()
        combined = out
        if err:
            combined += ("\n" if combined else "") + err
        if result.returncode != 0 and not err:
            combined += f"\n[exit code {result.returncode}]"
        return combined.strip() or "<no output>"
    except subprocess.TimeoutExpired:
        return "[TIMEOUT: execution exceeded 10 seconds]"
    except Exception as e:
        return f"[EXECUTION ERROR: {e}]"
    finally:
        try:
            os.unlink(fname)
        except OSError:
            pass


# ── Test scorer ────────────────────────────────────────────────────────────────

def score_implementation(
    code: str, test_assertions: List[str]
) -> Tuple[int, int, str]:
    """Run each assertion against the implementation; return (passed, total, detail)."""
    passed = 0
    lines = []
    for i, assertion in enumerate(test_assertions, 1):
        # Run assertion in a try/except so failures are reported, not raised
        runner = (
            f"{code}\n"
            f"try:\n"
            f"    {assertion}\n"
            f"    print('PASS')\n"
            f"except Exception as e:\n"
            f"    print(f'FAIL: {{e}}')\n"
        )
        result = _sandbox_exec(runner)
        if result.strip().startswith("PASS"):
            passed += 1
            lines.append(f"  [{i}] PASS")
        else:
            lines.append(f"  [{i}] FAIL — {result.strip()[:120]}")
    return passed, len(test_assertions), "\n".join(lines)


def make_code_scorer(test_assertions: List[str]) -> Callable:
    """Return a scorer callable(agent_code, ground_truth) → float in [0, 1]."""
    def scorer(agent_code: str, _ground_truth) -> float:
        if not agent_code or not agent_code.strip():
            return 0.0
        passed, total, _ = score_implementation(agent_code, test_assertions)
        return passed / total if total > 0 else 0.0
    return scorer


# ── CodeTool: uses 'code' parameter instead of 'query' ───────────────────────

class CodeTool(Tool):
    """A Tool variant whose Anthropic schema uses 'code' as the input parameter."""

    def to_anthropic_tool_spec(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute in the sandbox",
                    }
                },
                "required": ["code"],
            },
        }


def make_python_exec_tool() -> CodeTool:
    """Return a sandboxed python_exec tool for CodeExecutorAgent."""
    return CodeTool(
        name="python_exec",
        description=(
            "Execute Python code in a sandboxed subprocess and return stdout + stderr. "
            "Use this to test your implementation and verify assertions. "
            "The sandbox has no network access and a 10-second timeout. "
            "Include your implementation AND test assertions in the code you submit."
        ),
        fn=lambda code: _sandbox_exec(code),
    )


def get_code_tools() -> Dict[str, CodeTool]:
    """Return the tool set for code execution tasks."""
    return {"python_exec": make_python_exec_tool()}
