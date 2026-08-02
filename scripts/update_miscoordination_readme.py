#!/usr/bin/env python3
"""Regenerate the README miscoordination section from compact metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.report_miscoordination_study import render_readme  # noqa: E402

START = "<!-- MISCOORDINATION-RESULTS:START -->"
END = "<!-- MISCOORDINATION-RESULTS:END -->"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="results/miscoordination_v1/metrics.json")
    parser.add_argument("--readme", default="README.md")
    args = parser.parse_args()
    payload = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    readme_path = Path(args.readme)
    readme = readme_path.read_text(encoding="utf-8")
    if START not in readme or END not in readme:
        raise ValueError("README miscoordination result markers are missing")
    before, remainder = readme.split(START, 1)
    _, after = remainder.split(END, 1)
    readme_path.write_text(
        before + START + "\n" + render_readme(payload) + END + after,
        encoding="utf-8",
        newline="\n",
    )
    print(f"Updated {readme_path} from {args.metrics}")


if __name__ == "__main__":
    main()
