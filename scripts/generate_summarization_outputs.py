#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from summarization_finetune.generation import generate_candidates, generate_evaluation

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, default=Path("summarization_finetune/study_config.json"))
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--mode", choices=("evaluation", "candidates"), required=True)
parser.add_argument("--model-label", choices=("base", "sft", "dpo"))
parser.add_argument("--articles", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--device", default="cuda")
args = parser.parse_args(); config = json.loads(args.config.read_text())
if args.mode == "evaluation":
    if not args.model_label: parser.error("--model-label is required for evaluation")
    result = generate_evaluation(checkpoint=args.checkpoint, model_label=args.model_label, articles_path=args.articles,
                                 output_path=args.output, config=config, device=args.device)
else:
    result = generate_candidates(checkpoint=args.checkpoint, articles_path=args.articles, output_path=args.output,
                                 config=config, device=args.device)
print(json.dumps(result, indent=2))
