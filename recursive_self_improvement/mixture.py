"""Deterministic self/external label assignment and self-likelihood ranking."""

from __future__ import annotations

import hashlib


def use_self_label(seed: int, percent_self: int, round_index: int, prompt_id: str) -> bool:
    if not 0 <= percent_self <= 100:
        raise ValueError("percent_self must be in [0, 100]")
    digest = hashlib.sha256(f"{seed}:{percent_self}:{round_index}:{prompt_id}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") / 2**64
    return bucket < percent_self / 100


def choose_by_normalized_log_likelihood(candidate_a: str, candidate_b: str, score_a: float, score_b: float) -> tuple[str, str, str]:
    """Choose using caller-computed mean response-token log likelihoods."""
    if score_a == score_b:
        # Content hash makes ties deterministic without privileging display order.
        chosen = min((candidate_a, candidate_b), key=lambda text: hashlib.sha256(text.encode()).hexdigest())
        return chosen, candidate_b if chosen == candidate_a else candidate_a, "hash_tie_break"
    if score_a > score_b:
        return candidate_a, candidate_b, "self_likelihood"
    return candidate_b, candidate_a, "self_likelihood"
