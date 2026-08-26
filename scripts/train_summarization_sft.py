#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from summarization_finetune.training import train_sft

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, default=Path("summarization_finetune/study_config.json"))
parser.add_argument("--train-parquet", type=Path, default=Path("data/processed/summarization_finetune_v1/train.parquet"))
parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/summarization_sft_v1"))
parser.add_argument("--no-resume", action="store_true")
args = parser.parse_args()
print(json.dumps(train_sft(json.loads(args.config.read_text()), args.train_parquet, args.output_dir, resume=not args.no_resume), indent=2))
