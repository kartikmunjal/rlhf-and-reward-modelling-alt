import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_selection_uses_approved_pilot_repetitions_and_smaller_k_tiebreak(tmp_path):
    config = tmp_path / "config.json"
    amendment = tmp_path / "amendment.json"
    trials = tmp_path / "trials.jsonl"
    output = tmp_path / "selection.json"
    config.write_text(json.dumps({"stage3": {"pilot_speculative_tokens": [2, 4, 6]}}))
    amendment.write_text(json.dumps({"user_approved_completion": {"pilot_trials_per_candidate": 2}}))
    rows = []
    for k, rates in ((2, [10, 12]), (4, [11, 11]), (6, [9, 10])):
        rows.extend({"phase": "pilot", "speculative": True, "speculative_tokens": k,
                     "target": "dpo", "output_tokens_per_second": rate} for rate in rates)
    trials.write_text("".join(json.dumps(row) + "\n" for row in rows))
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/select_speculative_tokens.py"),
         "--config", str(config), "--amendment", str(amendment),
         "--trials", str(trials), "--output", str(output)],
        check=True,
    )
    result = json.loads(output.read_text())
    assert result["selected_speculative_tokens"] == 2
    assert result["candidates"]["2"]["n_trials"] == 2
