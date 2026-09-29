"""Audit one task or a private task JSONL with AST mutation testing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from verifier_reward_hacking.audit import audit_task
from verifier_reward_hacking.schema import CodeTask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--task-id")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = []
    for line in args.tasks.read_text(encoding="utf-8").splitlines():
        task = CodeTask.from_dict(json.loads(line))
        if args.task_id and task.task_id != args.task_id:
            continue
        rows.append(audit_task(task).to_dict())
    text = "\n".join(json.dumps(row, sort_keys=True) for row in rows) + ("\n" if rows else "")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
