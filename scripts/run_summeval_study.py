#!/usr/bin/env python3
"""Resumable, fail-closed driver for the complete preregistered study."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(*arguments: str) -> None:
    command = [sys.executable, *arguments]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--approve-prompt-freeze", action="store_true",
                        help="Attest that the preregistered prompt is final after reviewing dev diagnostics")
    args = parser.parse_args()
    missing = [name for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY") if not os.getenv(name)]
    if missing:
        raise SystemExit("Missing required environment variables: " + ", ".join(missing))
    run("scripts/run_summeval_judge.py", "--phase", "dev", "--provider", "anthropic")
    run("scripts/analyze_summeval_dev.py")
    if not args.approve_prompt_freeze:
        raise SystemExit("Development run is complete. Review results/summeval_judge_v1/dev_report.md, then resume with --approve-prompt-freeze")
    run("scripts/freeze_summeval_prompt.py", "--approve")
    run("scripts/run_summeval_judge.py", "--phase", "heldout", "--provider", "anthropic")
    run("scripts/run_summeval_judge.py", "--phase", "heldout", "--provider", "openai")
    run("scripts/run_summeval_judge.py", "--phase", "heldout", "--provider", "anthropic", "--kind", "pairwise")
    run("scripts/analyze_summeval_judge.py")


if __name__ == "__main__":
    main()
