#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from summarization_finetune.training import train_dpo

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, default=Path("summarization_finetune/study_config.json"))
parser.add_argument("--preferences", type=Path, default=Path("data/processed/summarization_finetune_v1/dpo_preferences.jsonl"))
parser.add_argument("--sft-checkpoint", type=Path, default=Path("checkpoints/summarization_sft_v1/merged"))
parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/summarization_dpo_v1"))
args = parser.parse_args(); print(json.dumps(train_dpo(json.loads(args.config.read_text()), args.preferences, args.sft_checkpoint, args.output_dir), indent=2))
