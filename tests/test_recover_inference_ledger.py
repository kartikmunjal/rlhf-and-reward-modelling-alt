import json
import subprocess
import sys
from pathlib import Path


def test_recovery_removes_only_exact_selected_rows(tmp_path):
    ledger = tmp_path / "raw.jsonl"
    audit = tmp_path / "audit.json"
    rows = [
        {"system": "hf", "target": "dpo", "precision": "fp16", "trial_index": 0},
        {"system": "vllm", "target": "dpo", "precision": "fp16", "trial_index": 0},
        {"system": "vllm", "target": "dpo", "precision": "gptq", "trial_index": 0},
    ]
    ledger.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    subprocess.run(
        [
            sys.executable, str(Path(__file__).parents[1] / "scripts/recover_inference_ledger.py"),
            "--ledger", str(ledger), "--audit", str(audit), "--expected-removed", "1",
            "--system", "vllm", "--target", "dpo", "--precision", "fp16",
            "--reason", "test",
        ],
        check=True,
    )
    retained = [json.loads(line) for line in ledger.read_text().splitlines()]
    record = json.loads(audit.read_text())
    assert retained == [rows[0], rows[2]]
    assert record["removed"] == [rows[1]]
    assert record["removed_rows"] == 1
