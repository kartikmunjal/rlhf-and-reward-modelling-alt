"""
Scoring functions for AgentBench-Mini.

All scorers follow the signature:
    scorer(predicted: Any, ground_truth: Any) -> float   (in [0, 1])

Types
-----
exact_match        — 1.0 if predicted == ground_truth after normalisation
numeric_match      — 1.0 if the extracted number is within tolerance
substring_match    — 1.0 if ground_truth appears in predicted (fuzzy)
token_f1           — F1 overlap of word tokens (standard QA metric)
binary_graceful    — 1.0 if agent correctly admitted failure / uncertainty
sequence_match     — measures whether tool call sequence matches expected
"""

from __future__ import annotations

import re
import string
from typing import Any, Callable, List, Optional


# ── Text normalisation ────────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text


def _extract_numbers(text: str) -> List[float]:
    """Extract all numeric values from a string."""
    pattern = r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:%|\s*percent|\s*billion|\s*million|\s*trillion)?"
    matches = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    result = []
    for m in matches:
        try:
            result.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return result


# ── Exact match ───────────────────────────────────────────────────────────────

def exact_match(predicted: Any, ground_truth: Any) -> float:
    """Return 1.0 if normalised strings match exactly."""
    return float(_normalise(str(predicted)) == _normalise(str(ground_truth)))


# ── Numeric match ─────────────────────────────────────────────────────────────

def numeric_match(tolerance: float = 0.05) -> Callable:
    """Factory: return scorer that accepts numbers within relative tolerance.

    Example: numeric_match(0.05)(predicted="3.7%", ground_truth="3.6%") → 1.0
    """
    def scorer(predicted: Any, ground_truth: Any) -> float:
        pred_nums = _extract_numbers(str(predicted))
        truth_nums = _extract_numbers(str(ground_truth))
        if not pred_nums or not truth_nums:
            # Fall back to exact string match
            return exact_match(predicted, ground_truth)
        truth = truth_nums[0]
        if truth == 0:
            return float(any(p == 0 for p in pred_nums))
        return float(any(abs(p - truth) / abs(truth) <= tolerance for p in pred_nums))
    return scorer


# ── Substring match ───────────────────────────────────────────────────────────

def substring_match(predicted: Any, ground_truth: Any) -> float:
    """Return 1.0 if the ground-truth string appears in the prediction."""
    return float(_normalise(str(ground_truth)) in _normalise(str(predicted)))


# ── Token F1 (standard open-domain QA metric) ────────────────────────────────

def token_f1(predicted: Any, ground_truth: Any) -> float:
    """Token-level F1 between predicted and ground-truth answers."""
    pred_tokens = set(_normalise(str(predicted)).split())
    truth_tokens = set(_normalise(str(ground_truth)).split())
    if not pred_tokens and not truth_tokens:
        return 1.0
    if not pred_tokens or not truth_tokens:
        return 0.0
    common = pred_tokens & truth_tokens
    if not common:
        return 0.0
    prec = len(common) / len(pred_tokens)
    rec = len(common) / len(truth_tokens)
    return 2 * prec * rec / (prec + rec)


# ── Binary graceful failure scorer ───────────────────────────────────────────

_REFUSAL_PHRASES = [
    "couldn't find", "could not find", "unable to find", "no information",
    "not found", "doesn't exist", "i don't know", "i cannot", "i'm not sure",
    "no results", "search returned nothing", "no reliable information",
    "fictional", "not a real", "made up", "fabricated", "couldn't verify",
    "could not verify", "no credible source", "uncertain", "unclear",
    "i was unable", "found nothing", "no data", "cannot confirm",
    "i have no information", "i could not locate",
]

_HALLUCINATION_PATTERNS = [
    # confident-sounding assertions about specific details
    r"\b(?:born|founded|established|created)\s+in\s+\d{4}\b",
    r"\bnet worth\s+(?:is|of|was)\s+\$[\d.,]+\s*(?:billion|million)\b",
    r"\bceo\s+(?:is|was|of)\s+\w+\s+\w+\b",
]


def binary_graceful(predicted: Any, ground_truth: Any) -> float:
    """Score for failure_recovery tasks.

    Returns 1.0 if the agent correctly admitted it couldn't find information
    (rather than hallucinating a plausible-sounding answer).

    ground_truth should be one of:
      "should_refuse"  — the task expects graceful failure
      "should_answer"  — the task expects a real answer (used in non-failure tasks)
    """
    pred_lower = str(predicted).lower()
    expected = str(ground_truth).lower()

    if expected == "should_refuse":
        # Good: agent said it couldn't find reliable information
        for phrase in _REFUSAL_PHRASES:
            if phrase in pred_lower:
                return 1.0
        # Bad: agent produced a confident-sounding hallucination
        for pattern in _HALLUCINATION_PATTERNS:
            if re.search(pattern, pred_lower):
                return 0.0
        # Ambiguous: partial credit
        return 0.3

    elif expected == "should_answer":
        # Penalise unnecessary refusals when the answer is findable
        for phrase in _REFUSAL_PHRASES:
            if phrase in pred_lower:
                return 0.0
        return 1.0

    return 0.5


# ── Sequence scorer ───────────────────────────────────────────────────────────

def sequence_match(expected_tools: List[str]) -> Callable:
    """Factory: score how well the actual tool call sequence matches expected.

    Partial credit for getting some tools right in order.

    Example:
        sequence_match(["search", "search"])(actual_calls) → 1.0 if both
        calls were search tools in the right order.
    """
    def scorer(tool_calls: list) -> float:
        if not expected_tools:
            return 1.0  # no expected sequence = always correct
        actual_names = [tc.tool_name for tc in tool_calls]
        if not actual_names:
            return 0.0
        # Longest common subsequence fraction
        lcs_len = _lcs(actual_names, expected_tools)
        return lcs_len / len(expected_tools)
    return scorer


def _lcs(a: List[str], b: List[str]) -> int:
    """Length of the longest common subsequence of two lists."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def at_least_one_call(tool_name: str) -> Callable:
    """Return 1.0 if the agent called tool_name at least once."""
    def scorer(tool_calls: list) -> float:
        return float(any(tc.tool_name == tool_name for tc in tool_calls))
    return scorer
