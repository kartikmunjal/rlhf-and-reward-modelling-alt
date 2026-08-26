#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from summarization_finetune.data import prepare

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, default=Path("summarization_finetune/study_config.json"))
parser.add_argument("--summeval-manifest", type=Path, default=Path("llm_judge_summeval/data_manifest.json"))
parser.add_argument("--output-dir", type=Path, default=Path("data/processed/summarization_finetune_v1"))
args = parser.parse_args()
print(json.dumps(prepare(args.output_dir, json.loads(args.config.read_text()), args.summeval_manifest), indent=2))
