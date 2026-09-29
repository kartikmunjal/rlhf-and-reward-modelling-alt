"""Build post-preregistration candidate tasks and trusted mutation audits."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from verifier_reward_hacking.audit import audit_task
from verifier_reward_hacking.humaneval_tasks import load_selected_tasks
from verifier_reward_hacking.programmatic_tasks import build_candidate_pool


def write_jsonl(path: Path, rows) -> str:
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            line = json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
            handle.write(line)
            digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--humaneval-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/verifier_reward_hacking_v1"))
    parser.add_argument("--variants-per-family", type=int, default=30)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    generated = build_candidate_pool(args.variants_per_family)
    public = load_selected_tasks(args.humaneval_archive)
    task_path = args.output_dir / "candidate_tasks_private.jsonl"
    audit_path = args.output_dir / "candidate_audits.jsonl"
    tasks_sha = write_jsonl(task_path, (task.to_dict() for task in generated + public))
    audits = [audit_task(task).to_dict() for task in generated + public]
    audits_sha = write_jsonl(audit_path, audits)
    manifest = {
        "study_id": "verifier_reward_hacking_v1",
        "status": "candidate_pool_not_final_data",
        "generated_candidates": len(generated),
        "human_eval_candidates": len(public),
        "task_file": {"path": str(task_path), "sha256": tasks_sha},
        "audit_file": {"path": str(audit_path), "sha256": audits_sha},
        "weak_flag_counts": {
            flag: sum(flag in row["weak_flags"] for row in audits)
            for flag in sorted({flag for row in audits for flag in row["weak_flags"]})
        },
        "note": "Candidate artifacts precede base-model V0 eligibility QA and are not analysis data."
    }
    manifest_path = args.output_dir / "candidate_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()
