#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference_serving.distillation import train_draft

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, default=Path("inference_serving/study_config.json"))
parser.add_argument("--train-parquet", type=Path, default=Path("data/processed/summarization_finetune_v1/train.parquet"))
parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/inference_serving_v1/gpt2_small_draft"))
parser.add_argument("--no-resume", action="store_true")
args = parser.parse_args(); config = json.loads(args.config.read_text(encoding="utf-8"))
print(json.dumps(train_draft(config, args.train_parquet, args.output_dir, resume=not args.no_resume), indent=2))
