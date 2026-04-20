"""
Programmatic Reward Signal — The Third Reward Methodology.

Alongside preference-based (Bradley-Terry RM) and rubric/criteria-based (Rubric RM)
reward models, programmatic rewards define quality through explicit, deterministic
rules rather than learned weights. They are zero-cost to compute (no model forward
pass), zero-bias (no spurious correlations from training data), and fully inspectable
— but they cannot capture nuance or surface novel quality signals.

Three-methodology comparison
-----------------------------
┌─────────────────────┬─────────────────────┬────────────────────────┐
│ Preference (BT RM)  │ Rubric (LLM-graded) │ Programmatic (rules)   │
├─────────────────────┼─────────────────────┼────────────────────────┤
│ Pairwise acc: 72.4% │ OOD ρ: 0.71         │ No training required   │
│ Captures nuance     │ Interpretable        │ Zero length bias       │
│ Length bias +0.147  │ Acc cost: −2.3 pp   │ Cannot capture nuance  │
│ OOD degrades        │ Needs LLM judge      │ Binary gradient        │
└─────────────────────┴─────────────────────┴────────────────────────┘

Design
------
Two scoring functions are provided:

1. `score_binary(response, tokenizer)` — returns 1.0 (PASS) or 0.0 (FAIL).
   PASS conditions (both must hold):
     (a) Response length ≤ LENGTH_THRESHOLD tokens
     (b) Response does NOT start with a hollow affirmation

   This is the strictest version. Suitable for a hard gate: "I will not reward
   any response that is too long or starts sycophantically, regardless of content."

2. `ContinuousProgrammaticReward.score(response, tokenizer)` — returns [0, 1].
   Combines:
     - Length score: max(0, 1 − (n_tokens − LENGTH_THRESHOLD) / LENGTH_THRESHOLD)
       = 1.0 when length ≤ threshold, decreasing linearly to 0 at 2× threshold
     - Directness score: 1.0 if no hollow affirmation, 0.0 if detected
   Combined: 0.5 * length_score + 0.5 * directness_score

   Suitable as a continuous training signal (composite reward with BT RM or Rubric RM).

Hollow affirmations (BT-PPO learns to produce these — see Key Findings, Ext 15)
--------------------------------------------------------------------------------
  "That's a great question!" — hollow opener; adds no content
  "Absolutely!" / "Certainly!" / "Of course!" — sycophantic openers
  "I'd be happy to help!" — performative enthusiasm before actual answer
  "Great question!" / "Excellent question!" — flatters the user

Usage
-----
    from src.training.programmatic_reward import (
        score_binary,
        ContinuousProgrammaticReward,
        ProgrammaticRewardConfig,
    )

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")

    # Binary gate
    r = score_binary("Of course! Here is a detailed answer...", tokenizer)
    # r == 0.0 (hollow affirmation detected)

    r = score_binary("The key factor is preparation.", tokenizer)
    # r == 1.0 (direct, concise response)

    # Continuous signal
    scorer = ContinuousProgrammaticReward()
    r = scorer.score("The key factor is preparation.", tokenizer)
    # r ≈ 1.0 (short + direct)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional


# ── Constants ──────────────────────────────────────────────────────────────────

LENGTH_THRESHOLD = 150  # tokens; responses longer than this lose length credit

# Hollow affirmation regex (case-insensitive, matches at start of response or
# as the first non-whitespace clause)
_HOLLOW_PATTERNS = [
    r"^(that'?s\s+(a\s+)?(great|excellent|wonderful|fantastic|good)\s+question)",
    r"^(great|excellent|wonderful|fantastic|amazing)\s+question",
    r"^absolutely[!,.]",
    r"^certainly[!,.]",
    r"^of\s+course[!,.]",
    r"^sure[!,]\s",
    r"^i('d| would) be (happy|glad|delighted) to\b",
    r"^(what\s+a\s+)?(great|excellent|interesting|wonderful)\s+(question|topic|point)",
]
_HOLLOW_RE = re.compile(
    "|".join(_HOLLOW_PATTERNS),
    re.IGNORECASE | re.MULTILINE,
)


# ── Configuration ──────────────────────────────────────────────────────────────

@dataclass
class ProgrammaticRewardConfig:
    """Configuration for the programmatic reward scorer.

    Parameters
    ----------
    length_threshold : int
        Maximum token count for a response to receive full length credit.
        Default 150 (empirically, direct answers to conversational prompts
        are rarely longer than this).
    length_weight : float
        Weight of the length score in the continuous composite (default 0.5).
    directness_weight : float
        Weight of the directness score (1 − hollow_flag) in the composite (default 0.5).
    """
    length_threshold: int = LENGTH_THRESHOLD
    length_weight: float = 0.5
    directness_weight: float = 0.5

    def __post_init__(self):
        if abs(self.length_weight + self.directness_weight - 1.0) > 1e-6:
            raise ValueError(
                f"length_weight + directness_weight must sum to 1.0, "
                f"got {self.length_weight + self.directness_weight}"
            )


# ── Core scoring functions ─────────────────────────────────────────────────────

def _n_tokens(response: str, tokenizer) -> int:
    """Count tokens in response without special tokens."""
    return len(tokenizer.encode(response, add_special_tokens=False))


def _has_hollow_affirmation(response: str) -> bool:
    """Return True if response begins with a hollow affirmation."""
    return bool(_HOLLOW_RE.search(response.lstrip()))


def score_binary(response: str, tokenizer) -> float:
    """Binary programmatic reward: 1.0 (PASS) or 0.0 (FAIL).

    PASS conditions (both must hold):
      (a) len(tokens) ≤ LENGTH_THRESHOLD
      (b) response does not start with a hollow affirmation

    Parameters
    ----------
    response : str
        The model's response text (not including the prompt).
    tokenizer : HuggingFace tokenizer
        Used to count tokens.

    Returns
    -------
    float — 1.0 or 0.0
    """
    if _has_hollow_affirmation(response):
        return 0.0
    if _n_tokens(response, tokenizer) > LENGTH_THRESHOLD:
        return 0.0
    return 1.0


def score_binary_batch(responses: List[str], tokenizer) -> List[float]:
    """Score a list of responses with the binary programmatic reward."""
    return [score_binary(r, tokenizer) for r in responses]


# ── Continuous scorer ──────────────────────────────────────────────────────────

class ContinuousProgrammaticReward:
    """Continuous programmatic reward in [0, 1].

    Combines a length score (linearly penalises excess length) and a
    directness score (0 if hollow affirmation detected, 1 otherwise).

    Parameters
    ----------
    config : ProgrammaticRewardConfig
        Controls threshold and component weights.
    """

    def __init__(self, config: Optional[ProgrammaticRewardConfig] = None):
        self.config = config or ProgrammaticRewardConfig()

    def score(self, response: str, tokenizer) -> float:
        """Score a single response.

        Returns
        -------
        float in [0, 1]
        """
        cfg = self.config
        n_tok = _n_tokens(response, tokenizer)

        # Length score: 1.0 at or below threshold, decreasing linearly to 0
        # at 2× threshold, capped at 0.
        if n_tok <= cfg.length_threshold:
            length_score = 1.0
        else:
            excess = n_tok - cfg.length_threshold
            length_score = max(0.0, 1.0 - excess / cfg.length_threshold)

        # Directness score
        directness_score = 0.0 if _has_hollow_affirmation(response) else 1.0

        return (
            cfg.length_weight * length_score
            + cfg.directness_weight * directness_score
        )

    def score_batch(self, responses: List[str], tokenizer) -> List[float]:
        """Score a list of responses."""
        return [self.score(r, tokenizer) for r in responses]

    def score_breakdown(self, response: str, tokenizer) -> dict:
        """Return per-component scores for inspection.

        Returns
        -------
        dict with keys: "length_score", "directness_score",
        "composite", "n_tokens", "hollow_affirmation_detected"
        """
        cfg = self.config
        n_tok = _n_tokens(response, tokenizer)
        has_hollow = _has_hollow_affirmation(response)

        if n_tok <= cfg.length_threshold:
            length_score = 1.0
        else:
            excess = n_tok - cfg.length_threshold
            length_score = max(0.0, 1.0 - excess / cfg.length_threshold)

        directness_score = 0.0 if has_hollow else 1.0
        composite = cfg.length_weight * length_score + cfg.directness_weight * directness_score

        return {
            "length_score": length_score,
            "directness_score": directness_score,
            "composite": composite,
            "n_tokens": n_tok,
            "hollow_affirmation_detected": has_hollow,
        }


# ── Composite reward helper ────────────────────────────────────────────────────

def composite_reward(
    bt_score: float,
    prog_score: float,
    rubric_conc_score: float,
    w_bt: float = 1.0,
    w_prog: float = 0.0,
    w_conc: float = 0.0,
) -> float:
    """Blend BT RM, programmatic, and rubric conciseness rewards.

    Parameters
    ----------
    bt_score : float
        Bradley-Terry RM output (raw scalar, typically in [-3, 3]).
        Normalised to [0,1] via sigmoid before blending.
    prog_score : float
        Programmatic reward in [0, 1].
    rubric_conc_score : float
        Rubric RM conciseness sub-score, normalised to [0, 1].
    w_bt, w_prog, w_conc : float
        Weights (should sum to 1.0).

    Returns
    -------
    float — composite reward
    """
    import math
    bt_normalised = 1.0 / (1.0 + math.exp(-bt_score))  # sigmoid normalisation
    return w_bt * bt_normalised + w_prog * prog_score + w_conc * rubric_conc_score


# ── Quick demo ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    scorer = ContinuousProgrammaticReward()

    examples = [
        ("That's a great question! Anxiety before interviews is normal and there are "
         "several research-backed strategies. First, thorough preparation is key — when you "
         "know your material well, uncertainty decreases. Second, controlled breathing "
         "activates the parasympathetic nervous system. Third, reframing the interview as "
         "a two-way conversation can shift your posture. Many people find light exercise "
         "the morning of helpful. Would you like more detail on any of these?",
         "PPO verbose response (hollow affirmation + long)"),
        ("Preparation is the biggest lever — most interview anxiety comes from uncertainty. "
         "On the day, try box breathing before you go in. Also: they invited you because "
         "your CV already cleared a bar.",
         "DPO concise response (direct, under 150 tok)"),
        ("One thing that can help is to prepare thoroughly — research the company, practice "
         "common questions, and remind yourself it's normal to feel nervous.",
         "SFT response (concise, direct)"),
    ]

    print("Programmatic Reward Score Breakdown")
    print("=" * 65)
    for resp, label in examples:
        bd = scorer.score_breakdown(resp, tokenizer)
        flag = "HOLLOW" if bd["hollow_affirmation_detected"] else "OK"
        print(f"\n  {label}")
        print(f"  Tokens: {bd['n_tokens']:3d}  Length: {bd['length_score']:.2f}  "
              f"Directness: {bd['directness_score']:.1f} [{flag}]  "
              f"→ Composite: {bd['composite']:.3f}")
