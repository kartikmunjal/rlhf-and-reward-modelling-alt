"""Locked v2 multi-label loss implementations."""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F


def prevalence_positive_weights(labels: torch.Tensor, cap: float | None = None) -> torch.Tensor:
    positives = labels.sum(dim=0)
    if torch.any(positives <= 0):
        raise ValueError("Every category requires a positive training example")
    weights = (labels.shape[0] - positives) / positives
    return weights.clamp(max=cap) if cap is not None else weights


def multilabel_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    specification: Mapping[str, object],
    positive_weights: torch.Tensor,
) -> torch.Tensor:
    kind = str(specification["kind"])
    if kind == "bce":
        weighting = str(specification["positive_weighting"])
        weights = torch.ones_like(positive_weights) if weighting == "none" else positive_weights
        return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=weights)
    if kind == "focal":
        gamma = float(specification["gamma"])
        alpha = float(specification["alpha"])
        bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        probabilities = torch.sigmoid(logits)
        p_t = probabilities * labels + (1 - probabilities) * (1 - labels)
        alpha_t = alpha * labels + (1 - alpha) * (1 - labels)
        return (alpha_t * (1 - p_t).pow(gamma) * bce).mean()
    raise ValueError(f"Unknown loss kind: {kind}")
