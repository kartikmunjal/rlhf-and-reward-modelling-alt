#!/usr/bin/env python3
"""Fit simple power laws to RLHF training curves."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.scaling_laws import (
    EXAMPLE_RLHF_SCALING_POINTS,
    ScalingPoint,
    fit_all_families,
    format_fit_table,
)


def load_points(path: Path) -> list[ScalingPoint]:
    with path.open() as f:
        reader = csv.DictReader(f)
        return [
            ScalingPoint(
                label=row["label"],
                scale=float(row["scale"]),
                loss=float(row["loss"]),
                family=row.get("family", "all") or "all",
            )
            for row in reader
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit RLHF loss-vs-scale power laws")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--csv",
        type=Path,
        help="Measured CSV with label,scale,loss,family columns",
    )
    source.add_argument(
        "--example",
        action="store_true",
        help="Run the explicitly synthetic smoke-test fixture; not research evidence",
    )
    args = parser.parse_args()

    points = load_points(args.csv) if args.csv else EXAMPLE_RLHF_SCALING_POINTS
    fits = fit_all_families(points)
    if args.example:
        print("SYNTHETIC EXAMPLE — NOT A MEASURED RESULT")
    print(format_fit_table(fits))
    print()
    print("Interpretation: alpha is the empirical scaling exponent; larger alpha means loss improves faster with scale.")


if __name__ == "__main__":
    main()
