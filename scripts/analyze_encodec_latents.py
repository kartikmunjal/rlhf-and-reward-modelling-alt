#!/usr/bin/env python3
"""Analyze codec latent separability across speaker/content/style/environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.encodec_latents import analyze_manifest, load_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PCA/separability diagnostics on audio codec latents")
    parser.add_argument("--manifest", type=Path, required=True, help="CSV with path,speaker,content,style,environment")
    parser.add_argument("--output", type=Path, default=Path("results/encodec_latent_analysis.json"))
    args = parser.parse_args()

    examples = load_manifest(args.manifest)
    metrics = analyze_manifest(examples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2))

    print(json.dumps(metrics, indent=2))
    print(f"Saved latent analysis to {args.output}")


if __name__ == "__main__":
    main()
