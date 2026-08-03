#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.ppo_grpo_v2_pilot import load_config
from src.training.ppo_grpo_v2b_pilot import run_v2b_pilot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/ppo_grpo_v2b_pilot.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/ppo_grpo_v2b_pilot"))
    args = parser.parse_args()
    result = run_v2b_pilot(load_config(args.config), args.output_dir)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "pass" else 2)


if __name__ == "__main__":
    main()
