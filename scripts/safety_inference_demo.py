#!/usr/bin/env python3
"""Minimal audited CLI inference demo for a completed safety trial."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_safety_classifier import load_safety_model, predict  # noqa: E402
from src.safety.taxonomy import TARGET_LABELS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    thresholds_mapping = (
        metrics["thresholds"]
        if "thresholds" in metrics
        else metrics["performance"]["thresholds"]
    )
    thresholds = np.asarray([thresholds_mapping[name] for name in TARGET_LABELS])
    model_dir = Path(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = load_safety_model(model_dir)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model.to(device)
    probabilities = predict(
        model,
        tokenizer,
        [args.text],
        max_length=256,
        batch_size=1,
        device=device,
    )[0]
    result = {
        "input_sha256": __import__("hashlib").sha256(args.text.encode()).hexdigest(),
        "categories": {
            name: {
                "probability": float(probabilities[index]),
                "threshold": float(thresholds[index]),
                "flagged": bool(probabilities[index] >= thresholds[index]),
            }
            for index, name in enumerate(TARGET_LABELS)
        },
        "warning": (
            "Research decision-support output only. Do not use as the sole basis "
            "for sanctions or high-impact decisions."
        ),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
