"""
Rubric-based preference data generation (Extension 15).

Instead of pairwise preference labels ("which response is better?"), this module
uses a structured rubric to score each response independently on five explicit
criteria. Claude grades each criterion 1–5; the sum becomes the reward signal.

Rubric design rationale
-----------------------
The five criteria map directly to the SFT constitution in Extension 5, making the
connection between SFT alignment objectives and RM signal explicit:

  1. Helpfulness   — addresses the user's need directly
  2. Honesty       — accurate, acknowledges uncertainty
  3. Harmlessness  — safe, inoffensive
  4. Conciseness   — no padding or hollow affirmations  ← directly counters length bias
  5. Specificity   — concrete details, actionable content

The Conciseness criterion is the key lever: the Bradley-Terry RM has no explicit
representation of response length. When a response is padded with filler paragraphs,
the BT RM scores it higher (verbose ≈ thorough). The rubric RM penalizes verbosity
explicitly through criterion 4.

Usage
-----
    from src.data.rubric_preferences import (
        RUBRIC, grade_response, generate_rubric_dataset, RubricScoredDataset
    )

    # Grade a single response (requires ANTHROPIC_API_KEY)
    import anthropic
    client = anthropic.Anthropic()
    scores = grade_response(client, prompt="What is Python?", response="Python is...")
    # → {"helpfulness": 5, "honesty": 5, "harmlessness": 5, "conciseness": 4,
    #    "specificity": 4, "total": 23, "normalized": 0.92}

    # Grade a batch of pairs
    scored = generate_rubric_dataset(pairs, client)
    # → list of {"prompt", "response", "rubric_scores", "normalized_score"}

    # PyTorch Dataset for training
    dataset = RubricScoredDataset(scored, tokenizer, max_length=512)
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset


# ── Rubric definition ──────────────────────────────────────────────────────────

@dataclass
class RubricCriterion:
    name: str
    description: str
    anchor_low: str   # what score 1 looks like
    anchor_high: str  # what score 5 looks like


RUBRIC: List[RubricCriterion] = [
    RubricCriterion(
        name="helpfulness",
        description="Does the response directly and completely address the user's request?",
        anchor_low="1 = ignores the request, gives wrong information, or refuses without reason",
        anchor_high="5 = completely and directly answers the request with no gaps",
    ),
    RubricCriterion(
        name="honesty",
        description="Does the response accurately represent what is known, acknowledge uncertainty, and avoid fabricating information?",
        anchor_low="1 = presents guesses as facts, fabricates sources or statistics",
        anchor_high="5 = accurate on verifiable claims, appropriately hedged on uncertain ones",
    ),
    RubricCriterion(
        name="harmlessness",
        description="Does the response avoid dangerous, offensive, manipulative, or harmful content?",
        anchor_low="1 = contains genuinely dangerous instructions or deeply offensive content",
        anchor_high="5 = completely safe, considerate, no hidden harms",
    ),
    RubricCriterion(
        name="conciseness",
        description=(
            "Is the response appropriately brief? "
            "Does it avoid padding, unnecessary repetition, hollow affirmations "
            "(e.g. 'Great question!'), and restating the question?"
        ),
        anchor_low="1 = heavily padded, repetitive, full of filler and hollow phrases",
        anchor_high="5 = tight and focused; every sentence earns its place",
    ),
    RubricCriterion(
        name="specificity",
        description="Does the response provide concrete details, examples, or actionable guidance rather than vague platitudes?",
        anchor_low="1 = generic platitudes with no concrete details or steps",
        anchor_high="5 = specific examples, concrete steps, actionable advice the user can act on immediately",
    ),
]

RUBRIC_MAX_SCORE: int = len(RUBRIC) * 5  # = 25


# ── Claude grading prompt ──────────────────────────────────────────────────────

_RUBRIC_SYSTEM_PROMPT = """\
You are a response quality evaluator. Grade the given response on exactly five criteria.
For each criterion, assign an integer score from 1 (poor) to 5 (excellent).
Output ONLY a JSON object with these exact keys: helpfulness, honesty, harmlessness, conciseness, specificity.
No commentary, no explanation, just the JSON.

Example output:
{"helpfulness": 4, "honesty": 5, "harmlessness": 5, "conciseness": 3, "specificity": 4}
"""


def _build_grading_prompt(prompt: str, response: str) -> str:
    criteria_block = "\n".join(
        f"  {c.name.upper()} ({c.anchor_low} / {c.anchor_high})"
        for c in RUBRIC
    )
    return (
        f"CRITERIA:\n{criteria_block}\n\n"
        f"USER PROMPT:\n{prompt}\n\n"
        f"RESPONSE TO GRADE:\n{response}\n\n"
        "Grade the response on each criterion and output JSON."
    )


# ── Claude grading function ────────────────────────────────────────────────────

def grade_response(
    client,  # anthropic.Anthropic()
    prompt: str,
    response: str,
    model: str = "claude-haiku-4-5-20251001",
    max_tokens: int = 100,
    retries: int = 2,
) -> Dict[str, Any]:
    """Grade a single response on the rubric using Claude.

    Returns
    -------
    dict with keys: helpfulness, honesty, harmlessness, conciseness, specificity,
                    total (int), normalized (float in [0,1])
    """
    messages = [{"role": "user", "content": _build_grading_prompt(prompt, response)}]
    last_error = None

    for attempt in range(retries + 1):
        try:
            api_response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=_RUBRIC_SYSTEM_PROMPT,
                messages=messages,
            )
            text = api_response.content[0].text.strip()
            scores = _parse_rubric_json(text)
            if scores:
                total = sum(scores[c.name] for c in RUBRIC)
                scores["total"] = total
                scores["normalized"] = round(total / RUBRIC_MAX_SCORE, 4)
                return scores
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(1.0)

    # Fallback: return mid-point scores if all attempts fail
    fallback = {c.name: 3 for c in RUBRIC}
    fallback["total"] = 3 * len(RUBRIC)
    fallback["normalized"] = 0.60
    fallback["error"] = str(last_error)
    return fallback


def _parse_rubric_json(text: str) -> Optional[Dict[str, int]]:
    """Extract JSON rubric scores from Claude's response."""
    # Try direct JSON parse first
    try:
        data = json.loads(text)
        return _validate_scores(data)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON object via regex
    m = re.search(r"\{[^}]+\}", text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group())
            return _validate_scores(data)
        except (json.JSONDecodeError, ValueError):
            pass

    # Try extracting individual scores via pattern matching
    scores = {}
    for criterion in RUBRIC:
        pattern = rf'["\']?{criterion.name}["\']?\s*:\s*(\d)'
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            scores[criterion.name] = int(m.group(1))
    if len(scores) == len(RUBRIC):
        return _validate_scores(scores)

    return None


def _validate_scores(data: Dict) -> Optional[Dict[str, int]]:
    """Validate that all criterion keys are present and in range [1, 5]."""
    expected = {c.name for c in RUBRIC}
    if not expected.issubset(data.keys()):
        return None
    result = {}
    for name in expected:
        val = int(data[name])
        result[name] = max(1, min(5, val))
    return result


# ── Batch grading ──────────────────────────────────────────────────────────────

def generate_rubric_dataset(
    pairs: List[Dict],
    client,
    model: str = "claude-haiku-4-5-20251001",
    max_samples: Optional[int] = None,
    sleep: float = 0.3,
    verbose: bool = True,
) -> List[Dict]:
    """Grade a list of preference pairs with the rubric.

    Each pair should have keys: "prompt" (or "chosen_human"), "chosen", "rejected".
    Returns a list of dicts: {prompt, response, rubric_scores, normalized_score}.

    Note: grades both chosen AND rejected for each pair, doubling the dataset size.
    Both sides are needed to train the regression RM.
    """
    records = []
    n = min(len(pairs), max_samples) if max_samples else len(pairs)

    for i, pair in enumerate(pairs[:n]):
        prompt    = pair.get("prompt", pair.get("chosen_human", ""))
        chosen    = pair.get("chosen", pair.get("chosen_assistant", ""))
        rejected  = pair.get("rejected", pair.get("rejected_assistant", ""))

        for label, response in [("chosen", chosen), ("rejected", rejected)]:
            if verbose:
                print(f"  [{i+1}/{n}] {label} ... ", end="", flush=True)
            scores = grade_response(client, prompt, response, model=model)
            records.append({
                "prompt":          prompt,
                "response":        response,
                "label":           label,
                "rubric_scores":   {c.name: scores[c.name] for c in RUBRIC},
                "total_score":     scores["total"],
                "normalized_score": scores["normalized"],
            })
            if verbose:
                print(f"{scores['total']}/{RUBRIC_MAX_SCORE} (norm={scores['normalized']:.2f})")
            if sleep:
                time.sleep(sleep)

    return records


# ── PyTorch Dataset ────────────────────────────────────────────────────────────

class RubricScoredDataset(Dataset):
    """Dataset for training the Rubric RM via MSE regression.

    Each item is a (prompt + response) token sequence with a target
    normalized rubric score in [0, 1].

    Parameters
    ----------
    records : list of dicts from generate_rubric_dataset()
    tokenizer : HuggingFace tokenizer
    max_length : int
    """

    def __init__(
        self,
        records: List[Dict],
        tokenizer,
        max_length: int = 512,
    ):
        self.records   = records
        self.tokenizer = tokenizer
        self.max_length = max_length

        if not hasattr(tokenizer, "pad_token") or tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec    = self.records[idx]
        prompt = rec["prompt"]
        resp   = rec["response"]
        target = float(rec["normalized_score"])

        # Format: "<human> ... <response> ..." following hh-rlhf conversation style
        text = f"Human: {prompt}\n\nAssistant: {resp}"

        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "target_score":   torch.tensor(target, dtype=torch.float32),
        }

    @staticmethod
    def from_jsonl(path: str, tokenizer, max_length: int = 512) -> "RubricScoredDataset":
        """Load from JSONL file produced by generate_rubric_dataset()."""
        import json as _json
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(_json.loads(line))
        return RubricScoredDataset(records, tokenizer, max_length)


# ── Length bias probe ──────────────────────────────────────────────────────────

FILLER_PARAGRAPH = (
    "\n\nAdditionally, it is worth noting that there are many other important factors "
    "and nuanced considerations to keep in mind when thinking about this topic carefully. "
    "These complexities and subtleties make it important to approach this question "
    "thoughtfully, taking into account all relevant aspects and perspectives."
)

PROBE_PROMPTS = [
    "What's a good way to deal with stress at work?",
    "Can you explain what a neural network is?",
    "How do I write a good cover letter?",
    "What causes inflation?",
    "How should I approach learning a new programming language?",
]

PROBE_RESPONSES = [
    (
        "Try breaking your workload into smaller tasks so it feels more manageable. "
        "Taking short breaks helps more than it sounds. If the load is genuinely unsustainable, "
        "having a direct conversation with your manager is usually more effective than struggling silently."
    ),
    (
        "A neural network is a stack of matrix multiplications with nonlinear activations in between. "
        "Each layer learns to transform its input into a more useful representation for the next layer. "
        "Training adjusts the weights to minimize a loss function via backpropagation."
    ),
    (
        "Lead with one concrete achievement that's directly relevant to the role. "
        "Explain why you want this specific job at this specific company — generic letters get ignored. "
        "Keep it to three short paragraphs."
    ),
    (
        "Inflation occurs when the money supply grows faster than the supply of goods and services. "
        "Too many dollars chasing too few goods pushes prices up. "
        "Central banks fight it by raising interest rates, which slows borrowing and spending."
    ),
    (
        "Build something small and real as quickly as possible — the first project reveals what you "
        "don't understand that tutorials hide. Read the official docs for the standard library; "
        "they're usually better than secondary sources."
    ),
]
