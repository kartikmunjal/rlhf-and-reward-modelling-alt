"""
Constitutional AI (CAI) / RLAIF — AI-generated preference labels.

Motivation
----------
Human annotation is the bottleneck in RLHF pipelines: expensive, slow to scale,
and subject to annotator disagreement.  Anthropic's Constitutional AI (Bai et al.,
2022) replaces the human labeler with a second LLM guided by a *constitution* — a
fixed set of principles like "prefer honest responses" and "prefer responses that
avoid harm".

The resulting pipeline is:
    1. Sample a prompt x from the dataset
    2. Generate two candidate responses (y_A, y_B) from the SFT model
    3. Call Claude with the constitution and both responses → AI labels a preference
    4. Collect (x, y_chosen, y_rejected) triples and train a reward model on them

This module provides the constitution, the Claude API caller, and a dataset class
that wraps AI-generated preference pairs into the same interface as the human-labeled
PreferenceDataset so the two are drop-in replaceable.

Usage
-----
    client = anthropic.Anthropic()
    pair = get_ai_preference(client, prompt, resp_a, resp_b, CONSTITUTION)
    # → {"chosen": ..., "rejected": ..., "preferred": "A", "reasoning": "..."}

Ablation
--------
Train reward models on both data sources and compare pairwise accuracy:
  - Human labels  (hh-rlhf)   → typically 0.72 accuracy
  - AI labels     (CAI)        → typically 0.67–0.70 accuracy
AI labels are noisier on edge cases but can be generated at arbitrary scale
and are refreshable as the constitution evolves.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

# ── Constitution ─────────────────────────────────────────────────────────────

CONSTITUTION: List[str] = [
    "Prefer the response that is more genuinely helpful and directly addresses "
    "what the person is asking for, without unnecessary padding or hedging.",
    "Prefer the response that is more honest — it acknowledges uncertainty when "
    "appropriate and does not fabricate facts.",
    "Prefer the response that avoids potential harm to the person or others.",
    "Prefer the response that is clearer and better organised, using concrete "
    "language rather than vague generalities.",
    "Prefer the response that respects the person's autonomy and does not "
    "lecture or moralize beyond what is relevant.",
    "Prefer the response that is appropriately concise — it does not pad with "
    "hollow affirmations like 'Great question!' or repeat the question back.",
]

# ── API caller ────────────────────────────────────────────────────────────────

def _build_cai_prompt(
    human_prompt: str,
    response_a: str,
    response_b: str,
    constitution: List[str],
) -> str:
    """Build the Claude prompt for CAI preference labeling."""
    principles = "\n".join(f"  {i+1}. {p}" for i, p in enumerate(constitution))
    return (
        "You are a careful AI assistant evaluating two responses to the same prompt.\n"
        "Use the following principles to decide which response is better:\n\n"
        f"{principles}\n\n"
        "---\n"
        f"Prompt:\n{human_prompt}\n\n"
        f"Response A:\n{response_a}\n\n"
        f"Response B:\n{response_b}\n\n"
        "---\n"
        "Which response better follows the principles above?\n"
        "Reply in exactly this format:\n"
        "PREFERRED: [A or B]\n"
        "CONFIDENCE: [high / medium / low]\n"
        "REASONING: [one sentence]\n"
    )


def get_ai_preference(
    client,                          # anthropic.Anthropic instance
    human_prompt: str,
    response_a: str,
    response_b: str,
    constitution: List[str] = CONSTITUTION,
    model: str = "claude-haiku-4-5-20251001",
    max_tokens: int = 150,
    retry_on_error: bool = True,
) -> Dict:
    """Call Claude to label a single preference pair.

    Parameters
    ----------
    client:
        An initialised ``anthropic.Anthropic()`` client.
    human_prompt:
        The conversation prompt (Human/Assistant format, excluding final response).
    response_a, response_b:
        The two candidate responses to compare.
    constitution:
        List of principle strings.  Defaults to the module-level CONSTITUTION.

    Returns
    -------
    dict with keys:
        preferred   : "A" or "B"
        confidence  : "high" | "medium" | "low"
        reasoning   : short rationale string
        chosen      : the preferred response text
        rejected    : the dispreferred response text
    """
    prompt_text = _build_cai_prompt(human_prompt, response_a, response_b, constitution)

    for attempt in range(3 if retry_on_error else 1):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt_text}],
            )
            text = response.content[0].text.strip()
            break
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)   # exponential backoff

    # Parse structured output
    preferred_match = re.search(r"PREFERRED:\s*([AB])", text, re.IGNORECASE)
    confidence_match = re.search(r"CONFIDENCE:\s*(high|medium|low)", text, re.IGNORECASE)
    reasoning_match = re.search(r"REASONING:\s*(.+)", text, re.IGNORECASE | re.DOTALL)

    preferred = preferred_match.group(1).upper() if preferred_match else "A"
    confidence = confidence_match.group(1).lower() if confidence_match else "medium"
    reasoning = reasoning_match.group(1).strip().split("\n")[0] if reasoning_match else ""

    chosen = response_a if preferred == "A" else response_b
    rejected = response_b if preferred == "A" else response_a

    return {
        "preferred": preferred,
        "confidence": confidence,
        "reasoning": reasoning,
        "chosen": chosen,
        "rejected": rejected,
        "full_prompt": human_prompt,
    }


# ── Preference generation pipeline ───────────────────────────────────────────

@dataclass
class CAIConfig:
    sft_checkpoint: str = "checkpoints/sft"
    output_path: str = "data/cai_preferences.jsonl"
    num_pairs: int = 2_000             # number of preference pairs to generate
    max_prompt_length: int = 256
    max_new_tokens: int = 128
    annotator_model: str = "claude-haiku-4-5-20251001"
    generation_temperature: float = 0.8
    # Only keep pairs where Claude answered with high or medium confidence
    min_confidence: str = "medium"    # "high" | "medium" | "low"
    requests_per_minute: int = 40     # stay well inside API rate limits


def generate_cai_preferences(cfg: CAIConfig) -> None:
    """Generate AI-labeled preference pairs for reward model training.

    For each prompt we:
    1. Sample *two different* responses from the SFT model (different temperatures
       or different seeds) to get meaningful variation.
    2. Ask Claude to label which is better given the constitution.
    3. Write the (prompt, chosen, rejected, confidence, reasoning) tuple to a JSONL.

    The resulting file can be used as a drop-in replacement for the hh-rlhf dataset
    in ``train_reward_model.py`` via the ``--cai_data`` flag.
    """
    import anthropic
    from datasets import load_dataset
    from transformers import AutoTokenizer, GPT2LMHeadModel
    from src.data.preprocessing import extract_prompt_and_response

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(cfg.sft_checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    sft_model = GPT2LMHeadModel.from_pretrained(cfg.sft_checkpoint).to(device)
    sft_model.eval()

    client = anthropic.Anthropic()

    raw = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base", split="train")
    raw = raw.select(range(min(cfg.num_pairs * 2, len(raw))))  # 2× buffer for filtering

    os.makedirs(os.path.dirname(cfg.output_path) or ".", exist_ok=True)

    results = []
    rate_delay = 60.0 / cfg.requests_per_minute

    from tqdm.auto import tqdm
    pbar = tqdm(raw, desc="CAI labeling")

    for ex in pbar:
        if len(results) >= cfg.num_pairs:
            break

        prompt, _ = extract_prompt_and_response(ex["chosen"])
        enc = tokenizer(
            prompt, return_tensors="pt", max_length=cfg.max_prompt_length,
            truncation=True,
        ).to(device)

        gen_kwargs = dict(
            max_new_tokens=cfg.max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )

        with torch.no_grad():
            # Response A: top-p sampling
            out_a = sft_model.generate(
                **enc, do_sample=True, top_p=0.9,
                temperature=cfg.generation_temperature, **gen_kwargs
            )
            # Response B: slightly higher temperature for more variation
            out_b = sft_model.generate(
                **enc, do_sample=True, top_p=0.95,
                temperature=cfg.generation_temperature + 0.2, **gen_kwargs
            )

        resp_a = tokenizer.decode(out_a[0], skip_special_tokens=True)[len(prompt):]
        resp_b = tokenizer.decode(out_b[0], skip_special_tokens=True)[len(prompt):]

        # Skip pairs that are nearly identical (Levenshtein distance proxy: word overlap)
        words_a = set(resp_a.lower().split())
        words_b = set(resp_b.lower().split())
        overlap = len(words_a & words_b) / max(len(words_a | words_b), 1)
        if overlap > 0.85:
            continue

        try:
            result = get_ai_preference(
                client, prompt, resp_a, resp_b,
                model=cfg.annotator_model,
            )
        except Exception as e:
            pbar.set_postfix({"error": str(e)[:40]})
            continue

        # Filter by confidence
        conf_order = {"high": 3, "medium": 2, "low": 1}
        if conf_order[result["confidence"]] < conf_order[cfg.min_confidence]:
            continue

        results.append(result)
        pbar.set_postfix({"n": len(results), "conf": result["confidence"]})
        time.sleep(rate_delay)

    # Write JSONL
    with open(cfg.output_path, "w") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")

    print(f"\nGenerated {len(results)} AI-labeled preference pairs → {cfg.output_path}")
    conf_dist = {c: sum(1 for r in results if r["confidence"] == c) for c in ("high", "medium", "low")}
    print(f"Confidence distribution: {conf_dist}")


# ── Dataset class ─────────────────────────────────────────────────────────────

class CAIPreferenceDataset(Dataset):
    """Preference dataset backed by AI-generated (CAI) labels.

    Identical interface to :class:`src.data.preprocessing.PreferenceDataset`
    so reward model training code works unchanged with either data source.
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        min_confidence: str = "medium",
    ) -> None:
        conf_order = {"high": 3, "medium": 2, "low": 1}
        threshold = conf_order[min_confidence]

        self.data = []
        with open(jsonl_path) as f:
            for line in f:
                item = json.loads(line)
                if conf_order.get(item.get("confidence", "low"), 0) >= threshold:
                    self.data.append(item)

        self.tokenizer = tokenizer
        self.max_length = max_length
        print(f"Loaded {len(self.data)} CAI preference pairs from {jsonl_path}")

    def __len__(self) -> int:
        return len(self.data)

    def _tok(self, text: str) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            text, max_length=self.max_length, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in enc.items()}

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        full_chosen = item["full_prompt"] + item["chosen"]
        full_rejected = item["full_prompt"] + item["rejected"]
        c = self._tok(full_chosen)
        r = self._tok(full_rejected)
        return {
            "chosen_input_ids": c["input_ids"],
            "chosen_attention_mask": c["attention_mask"],
            "rejected_input_ids": r["input_ids"],
            "rejected_attention_mask": r["attention_mask"],
        }
