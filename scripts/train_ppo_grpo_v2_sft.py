#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.ppo_grpo_v2_pilot import load_config, train_arithmetic_sft


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/ppo_grpo_v2_pilot.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/ppo_grpo_v2_sft"))
    args = parser.parse_args()
    print(json.dumps(train_arithmetic_sft(load_config(args.config), args.output_dir), indent=2))


if __name__ == "__main__":
    main()
