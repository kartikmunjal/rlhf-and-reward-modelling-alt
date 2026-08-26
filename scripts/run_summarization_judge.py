#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from summarization_finetune.judging import score_candidate_pairs, score_pointwise

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, default=Path("summarization_finetune/study_config.json"))
parser.add_argument("--kind", choices=("pointwise", "candidate-pairs"), required=True)
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--ledger", type=Path, required=True)
parser.add_argument("--provider", choices=("anthropic", "openai"))
args = parser.parse_args(); config = json.loads(args.config.read_text()); root = Path(__file__).resolve().parents[1]
if args.kind == "pointwise":
    if not args.provider: parser.error("--provider required for pointwise")
    result = score_pointwise(generations_path=args.input, ledger_path=args.ledger, provider_name=args.provider, config=config, root=root)
else:
    result = score_candidate_pairs(candidates_path=args.input, ledger_path=args.ledger, config=config, root=root)
print(json.dumps(result, indent=2))
