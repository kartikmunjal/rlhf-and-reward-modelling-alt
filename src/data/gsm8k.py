"""
GSM8K dataset preprocessing for Process vs Outcome Reward Model comparison.

Dataset: openai/gsm8k  (grade-school math, ~8.5k training examples)
Each example has:
    question : str   — a multi-step arithmetic word problem
    answer   : str   — a step-by-step solution ending with "#### <final_answer>"

Step parsing
------------
Each line of the answer is one step (blank lines are skipped).
Calculator annotations of the form <<3*4=12>> appear in some steps;
we use these as ground-truth intermediate values to label steps as correct
or incorrect when perturbing solutions.

PRM training signal
-------------------
For each training example we produce:
  - A list of (step_text, is_correct) pairs
  - The final answer (extracted from "#### <number>")

To create negative examples (incorrect steps) we apply:
  1. Arithmetic perturbation  — change a number in a <<A=B>> annotation to B±ε
  2. Step omission            — drop a middle step (changes the causal chain)

ORM training signal
-------------------
Binary label: final answer is correct (1) or incorrect (0).
For ORM training, we pair each correct solution with a perturbed version.
"""

from __future__ import annotations

import re
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import PreTrainedTokenizer


# ── Parsing helpers ───────────────────────────────────────────────────────────

_FINAL_ANSWER_RE = re.compile(r"####\s*(-?[\d,]+)")
_CALC_RE = re.compile(r"<<([^>]+)=(-?[\d.]+)>>")


def parse_steps(answer_text: str) -> List[str]:
    """Split a GSM8K answer into individual reasoning steps.

    Each non-blank line is treated as one step.  The "#### <answer>" line
    is preserved as the final step.
    """
    return [line.strip() for line in answer_text.strip().split("\n") if line.strip()]


def extract_final_answer(answer_text: str) -> Optional[str]:
    """Extract the numeric final answer from the '#### <N>' marker."""
    match = _FINAL_ANSWER_RE.search(answer_text)
    if match:
        return match.group(1).replace(",", "")
    return None


def verify_step(step: str) -> bool:
    """Return True if all calculator annotations in this step are correct.

    A step is considered correct if every <<A=B>> annotation evaluates to B.
    Steps without annotations are assumed correct (can't verify them cheaply).
    """
    for expr, stated_result in _CALC_RE.findall(step):
        try:
            computed = eval(expr, {"__builtins__": {}})  # safe: no builtins
            if abs(float(computed) - float(stated_result)) > 0.01:
                return False
        except Exception:
            pass  # unevaluable expression → assume correct
    return True


def perturb_step(step: str, rng: random.Random) -> Tuple[str, bool]:
    """Return a perturbed version of a step and whether it differs meaningfully.

    Perturbation strategy: find a <<expr=result>> annotation and change the
    stated result by a small random amount.  This creates a plausible-looking
    but arithmetically incorrect step — a realistic error a student might make.
    """
    matches = list(_CALC_RE.finditer(step))
    if not matches:
        return step, False  # nothing to perturb

    m = rng.choice(matches)
    original_result = m.group(2)
    try:
        val = float(original_result)
        delta = rng.choice([-2, -1, 1, 2])
        wrong_val = val + delta
        wrong_str = str(int(wrong_val)) if wrong_val == int(wrong_val) else str(wrong_val)
        perturbed = step[:m.start(2)] + wrong_str + step[m.end(2):]
        return perturbed, True
    except ValueError:
        return step, False


# ── Dataset classes ───────────────────────────────────────────────────────────

class ORMDataset(Dataset):
    """Outcome Reward Model dataset for GSM8K.

    Each item is a (question + solution, label) pair where:
      - label=1 : correct solution (from the dataset)
      - label=0 : perturbed solution with at least one wrong step
    """

    def __init__(
        self,
        split: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        num_samples: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        raw = load_dataset("openai/gsm8k", "main", split=split)
        if num_samples:
            raw = raw.select(range(min(num_samples, len(raw))))

        self.tokenizer = tokenizer
        self.max_length = max_length
        rng = random.Random(seed)

        self.items: List[Dict] = []
        for ex in raw:
            question = ex["question"]
            answer = ex["answer"]
            steps = parse_steps(answer)
            correct_solution = question + "\n" + answer

            # Perturb one step to create a negative example
            perturbed_steps = steps.copy()
            perturbed = False
            idxs = list(range(len(steps) - 1))  # don't perturb final "####" step
            rng.shuffle(idxs)
            for idx in idxs:
                new_step, changed = perturb_step(steps[idx], rng)
                if changed:
                    perturbed_steps[idx] = new_step
                    perturbed = True
                    break

            if not perturbed:
                continue  # skip examples where we couldn't create a negative

            wrong_solution = question + "\n" + "\n".join(perturbed_steps)
            self.items.append({"text": correct_solution, "label": 1})
            self.items.append({"text": wrong_solution, "label": 0})

        rng.shuffle(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.items[idx]
        enc = self.tokenizer(
            item["text"], max_length=self.max_length, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(item["label"], dtype=torch.float),
        }


@dataclass
class StepAnnotation:
    text: str
    is_correct: bool
    token_start: int   # token index of the first token of this step
    token_end: int     # token index of the last token of this step (inclusive)


class PRMDataset(Dataset):
    """Process Reward Model dataset for GSM8K.

    Each item provides step-level correctness labels.  The model receives
    the full (question + solution) sequence and must predict, at each step
    boundary, whether that step is correct.

    Representation
    --------------
    We mark step boundaries with a special token (we reuse GPT-2's EOS token
    as a separator since GPT-2 has no dedicated [SEP]).  For each separator
    position, we have a binary label (1=correct, 0=incorrect).

    Training loss
    -------------
    Binary cross-entropy at each step boundary position, averaged over steps.
    Only positions corresponding to step boundaries contribute to the loss.
    """

    def __init__(
        self,
        split: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        num_samples: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        raw = load_dataset("openai/gsm8k", "main", split=split)
        if num_samples:
            raw = raw.select(range(min(num_samples, len(raw))))

        self.tokenizer = tokenizer
        self.max_length = max_length
        rng = random.Random(seed)

        sep_token = tokenizer.eos_token  # reuse EOS as step separator
        self.sep_token_id = tokenizer.eos_token_id

        self.items: List[Dict] = []
        for ex in raw:
            question = ex["question"]
            steps = parse_steps(ex["answer"])

            # Perturb 0–2 steps to create mixed-correctness solutions
            step_correctness = [True] * len(steps)
            n_perturb = rng.randint(0, min(2, len(steps) - 1))
            perturb_idxs = rng.sample(range(len(steps) - 1), n_perturb)
            for idx in perturb_idxs:
                new_step, changed = perturb_step(steps[idx], rng)
                if changed:
                    steps[idx] = new_step
                    step_correctness[idx] = False

            # Build the token sequence: question + sep + step_1 + sep + step_2 + ...
            full_text = question + f" {sep_token} " + f" {sep_token} ".join(steps)
            enc = tokenizer(
                full_text, max_length=max_length, truncation=True,
                padding="max_length", return_tensors="pt",
            )
            input_ids = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)

            # Find positions of separator tokens
            sep_positions = (input_ids == self.sep_token_id).nonzero(as_tuple=True)[0].tolist()

            # Match separator positions to step labels
            # sep_positions[0] = after question; sep_positions[i+1] = after step i
            step_labels_at_sep = [-1] * max_length  # -1 = not a step boundary
            for i, pos in enumerate(sep_positions[1:]):   # skip the question separator
                if i < len(step_correctness) and pos < max_length:
                    step_labels_at_sep[pos] = int(step_correctness[i])

            self.items.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "step_labels": torch.tensor(step_labels_at_sep, dtype=torch.long),
                "num_steps": len(steps),
            })

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.items[idx]
