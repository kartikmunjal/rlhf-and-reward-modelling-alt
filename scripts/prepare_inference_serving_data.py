#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference_serving.data import prepare_calibration, prepare_partitions

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, default=Path("inference_serving/study_config.json"))
parser.add_argument("--source", type=Path, default=Path("data/processed/summarization_finetune_v1/final_eval_articles.jsonl"))
parser.add_argument("--train-parquet", type=Path, default=Path("data/processed/summarization_finetune_v1/train.parquet"))
parser.add_argument("--output-dir", type=Path, default=Path("data/processed/inference_serving_v1"))
args = parser.parse_args(); config = json.loads(args.config.read_text(encoding="utf-8"))
manifest = prepare_partitions(config, args.source, args.output_dir)
manifest["calibration"] = prepare_calibration(config, args.train_parquet, args.output_dir)
(args.output_dir / "partition_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(manifest, indent=2))
