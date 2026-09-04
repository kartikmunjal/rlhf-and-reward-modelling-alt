#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference_serving.data import prompt_text, read_jsonl
from inference_serving.quantization import quantize_gptq

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, default=Path("inference_serving/study_config.json"))
parser.add_argument("--model", required=True)
parser.add_argument("--calibration", type=Path, default=Path("data/processed/inference_serving_v1/calibration_articles.jsonl"))
parser.add_argument("--output-dir", type=Path, required=True)
args = parser.parse_args(); config = json.loads(args.config.read_text(encoding="utf-8"))
calibration = read_jsonl(args.calibration); texts = [prompt_text(row["source"], config) for row in calibration]
if len(texts) != config["stage2"]["calibration_articles"]: raise SystemExit("Calibration count mismatch")
print(json.dumps(quantize_gptq(args.model, args.output_dir, texts, config), indent=2))
