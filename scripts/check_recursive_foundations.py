#!/usr/bin/env python3
"""Exit 0 when all locked foundations pass, 1 on a gate failure, 2 pending."""
import json, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from recursive_self_improvement.config import load_effective_config

config=load_effective_config(ROOT); base=ROOT/"checkpoints/recursive_self_improvement_v1"
if not (base/"sft/run_manifest.json").exists(): raise SystemExit(2)
for seed in config["reward_ensemble"]["member_seeds"]:
    path=base/f"reward_ensemble/seed{seed}/run_manifest.json"
    if not path.exists(): raise SystemExit(2)
    status=json.loads(path.read_text(encoding="utf-8"))["status"]
    if status != "complete": raise SystemExit(1)
raise SystemExit(0)
