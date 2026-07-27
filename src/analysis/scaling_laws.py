"""Scaling-law utilities for RLHF training curves.

The functions here fit the standard power-law form used in language-model
scaling work:

    loss(x) = irreducible_loss + coefficient * x ** (-alpha)

where x can be data size, training tokens, optimizer steps, or a rough compute
proxy. The fit is intentionally small and dependency-light so it can run on
checked-in metrics as well as full experiment logs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np


@dataclass(frozen=True)
class ScalingPoint:
    label: str
    scale: float
    loss: float
    family: str = "all"


@dataclass(frozen=True)
class PowerLawFit:
    family: str
    irreducible_loss: float
    coefficient: float
    alpha: float
    r_squared: float
    n_points: int

    def predict(self, scale: float) -> float:
        return self.irreducible_loss + self.coefficient * (scale ** (-self.alpha))


# Synthetic fixture for smoke testing only. These values are not measurements
# and must never be presented as repository results.
EXAMPLE_RLHF_SCALING_POINTS: List[ScalingPoint] = [
    ScalingPoint("rm_2k_pairs", 2_000, 0.641, "reward_model"),
    ScalingPoint("rm_5k_pairs", 5_000, 0.598, "reward_model"),
    ScalingPoint("rm_10k_pairs", 10_000, 0.563, "reward_model"),
    ScalingPoint("sft_2k_examples", 2_000, 3.42, "sft"),
    ScalingPoint("sft_5k_examples", 5_000, 3.03, "sft"),
    ScalingPoint("sft_10k_examples", 10_000, 2.83, "sft"),
    ScalingPoint("dpo_100_steps", 100, 0.694, "dpo"),
    ScalingPoint("dpo_300_steps", 300, 0.612, "dpo"),
    ScalingPoint("dpo_600_steps", 600, 0.571, "dpo"),
    ScalingPoint("iter_dpo_1", 256, 0.850, "iterative_dpo"),
    ScalingPoint("iter_dpo_2", 512, 0.745, "iterative_dpo"),
    ScalingPoint("iter_dpo_3", 768, 0.681, "iterative_dpo"),
]


def fit_power_law(
    points: Sequence[ScalingPoint],
    family: str = "all",
    irreducible_loss: float | None = None,
) -> PowerLawFit:
    selected = [p for p in points if family == "all" or p.family == family]
    if len(selected) < 3:
        raise ValueError("At least three points are required to fit a power law")

    scales = np.asarray([p.scale for p in selected], dtype=float)
    losses = np.asarray([p.loss for p in selected], dtype=float)
    if np.any(scales <= 0):
        raise ValueError("All scale values must be positive")

    floor = float(irreducible_loss) if irreducible_loss is not None else max(0.0, float(losses.min()) * 0.9)
    adjusted = losses - floor
    if np.any(adjusted <= 0):
        raise ValueError("irreducible_loss must be below all observed losses")

    slope, intercept = np.polyfit(np.log(scales), np.log(adjusted), deg=1)
    alpha = -float(slope)
    coefficient = float(np.exp(intercept))
    predictions = floor + coefficient * (scales ** (-alpha))

    ss_res = float(np.sum((losses - predictions) ** 2))
    ss_tot = float(np.sum((losses - losses.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return PowerLawFit(
        family=family,
        irreducible_loss=floor,
        coefficient=coefficient,
        alpha=alpha,
        r_squared=r_squared,
        n_points=len(selected),
    )


def fit_all_families(points: Iterable[ScalingPoint]) -> List[PowerLawFit]:
    points = list(points)
    families = sorted({p.family for p in points})
    return [fit_power_law(points, family=f) for f in families if sum(p.family == f for p in points) >= 3]


def format_fit_table(fits: Sequence[PowerLawFit]) -> str:
    lines = [
        "| Family | n | alpha | irreducible loss | R^2 |",
        "|---|---:|---:|---:|---:|",
    ]
    for fit in fits:
        lines.append(
            f"| {fit.family} | {fit.n_points} | {fit.alpha:.3f} | "
            f"{fit.irreducible_loss:.3f} | {fit.r_squared:.3f} |"
        )
    return "\n".join(lines)
