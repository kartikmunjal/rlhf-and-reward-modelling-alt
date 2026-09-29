"""Apply the preregistered lexicographic dev-selection rule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    attempts = json.loads(args.attempt_ledger.read_text())
    if len(attempts) > 4:
        raise RuntimeError("Dev tuning exceeded the preregistered four-configuration cap")
    complete = [row for row in attempts if row.get("status") == "complete"]
    if not complete:
        raise RuntimeError("No completed dev configurations")
    for row in complete:
        if row.get("seed") != 2025 or row.get("optimizer_steps") != 50:
            raise RuntimeError("Dev attempt violates frozen seed or step budget")
        if not all(key in row for key in ("hidden_pass_at_1", "exploit_rate", "kl")):
            raise RuntimeError("Dev attempt missing selection metric")
    selected = max(complete, key=lambda row: (row["hidden_pass_at_1"], -row["exploit_rate"], -row["kl"], row["config_id"]))
    output = {"study_id": "verifier_reward_hacking_v1", "selection_rule": ["max_hidden_pass_at_1", "min_exploit_rate", "min_kl"], "attempt_count": len(attempts), "selected_config_id": selected["config_id"], "selected_config": selected["config"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
