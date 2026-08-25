#!/usr/bin/env python3
"""Prepare the locked SummEval dev/held-out artifacts without judge calls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_judge_summeval.data import prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, default=Path("data/raw/summeval/model_annotations.aligned.jsonl"))
    parser.add_argument("--cnndm", type=Path, default=Path("data/raw/summeval/cnndm-3.0.0-test.parquet"))
    parser.add_argument("--config", type=Path, default=Path("llm_judge_summeval/study_config.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/summeval_judge_v1"))
    parser.add_argument("--publish-manifest", type=Path, default=Path("llm_judge_summeval/data_manifest.json"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = prepare_dataset(args.annotations, args.cnndm, args.output_dir, config)
    publishable = dict(manifest)
    publishable["outputs"] = {
        key: {"filename": Path(value["path"]).name, "sha256": value["sha256"]}
        for key, value in manifest["outputs"].items()
    }
    args.publish_manifest.write_text(json.dumps(publishable, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
