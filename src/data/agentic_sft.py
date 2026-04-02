"""
Extension 9 — Agentic Post-Training Data Generation.

The gap this closes
--------------------
Standard SFT trains on (prompt, conversational_response) pairs from hh-rlhf.
Those pairs teach the model what good answers look like, but they contain no
tool use, no step-by-step reasoning, and no information about *when* to search
vs. *when* to answer from memory.

A model trained on conversational data and then asked to use tools at inference
time is doing something it has never seen in training. It improvises the ReAct
format rather than having practiced it.

Agentic post-training data fixes this: we generate expert demonstrations of
complete tool-use trajectories — Thought / Action / Observation / Thought / Answer —
and train the model via SFT on those sequences. The model learns the scaffold
from examples rather than having to infer it from a system prompt alone.

The empirical prediction
--------------------------
A model fine-tuned on agentic trajectory data should outperform a model fine-tuned
on general conversation data on AgentBench-Mini tool-use and multi-step tasks,
even if general preference accuracy stays similar. That delta is the empirical
argument for why agentic post-training matters — exactly what Anthropic's post-training
team is investigating at scale.

Trajectory format
------------------
Each generated trajectory follows the ReAct format, serialised as a single string:

    Human: {task_prompt}

    Assistant: Let me work through this step by step.

    Thought: {what I know, what I need to search}
    Action: web_search(query="{search query}")
    Observation: {tool result}
    Thought: {reasoning on result, is this enough?}
    [Action: ... / Observation: ... repeated as needed]
    Final answer: {answer}

This string becomes the `labels` in SFT training. The model learns the entire
agentic scaffold as its response distribution.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


# ── Trajectory format constants ───────────────────────────────────────────────

TRAJECTORY_SYSTEM_PROMPT = """You are a careful AI assistant that reasons step by step before answering.

When given a task that requires looking up information, use this format:

Thought: [reason about what you need to find out]
Action: web_search(query="[your search query]")
Observation: [the search result will appear here]
Thought: [reason about what you learned and whether you need more information]
Action: web_search(query="[another query if needed]")
Observation: [result]
Final answer: [your concise answer]

If the answer is something you already know with high confidence, you may skip
the search steps and go straight to:
Final answer: [your answer]

If a search returns no results, try a different query. If all searches fail,
say so explicitly rather than guessing."""


# ── Task catalogue ─────────────────────────────────────────────────────────────
# Diverse tool-use tasks drawn from the three AgentBench-Mini categories.
# We generate expert demonstrations for each.

AGENTIC_TASK_CATALOGUE: List[Dict] = [
    # ── Tool use / retrieval ───────────────────────────────────────────────────
    {
        "prompt": "What was the unemployment rate in the United States in March 2022?",
        "category": "tool_use",
        "ground_truth": "3.6%",
        "requires_search": True,
    },
    {
        "prompt": "What was the annual US GDP growth rate in 2023?",
        "category": "tool_use",
        "ground_truth": "2.5%",
        "requires_search": True,
    },
    {
        "prompt": "What is the approximate population of India as of 2023?",
        "category": "tool_use",
        "ground_truth": "1.43 billion",
        "requires_search": True,
    },
    {
        "prompt": "How tall is Mount Everest in meters according to the 2020 survey?",
        "category": "tool_use",
        "ground_truth": "8848.86 meters",
        "requires_search": True,
    },
    {
        "prompt": "Who was the CEO of Apple Inc. in 2023?",
        "category": "tool_use",
        "ground_truth": "Tim Cook",
        "requires_search": True,
    },
    {
        "prompt": "What was the US CPI inflation rate for the 12 months ending March 2022?",
        "category": "tool_use",
        "ground_truth": "8.0%",
        "requires_search": True,
    },
    {
        "prompt": "What is the speed of light in meters per second?",
        "category": "tool_use",
        "ground_truth": "299,792,458 m/s",
        "requires_search": False,  # Known constant — should NOT search
    },
    {
        "prompt": "What is the capital of France?",
        "category": "tool_use",
        "ground_truth": "Paris",
        "requires_search": False,  # Stable knowledge — no search needed
    },
    # ── Multi-step ─────────────────────────────────────────────────────────────
    {
        "prompt": "Find the company that acquired DeepMind, then find the name of that company's CEO as of 2023.",
        "category": "multi_step",
        "ground_truth": "Sundar Pichai",
        "requires_search": True,
        "n_steps": 2,
    },
    {
        "prompt": "Find the world's largest company by revenue in 2023, then find who founded it.",
        "category": "multi_step",
        "ground_truth": "Sam Walton",
        "requires_search": True,
        "n_steps": 2,
    },
    {
        "prompt": "Find the country with the highest GDP per capita in 2023, then find its population.",
        "category": "multi_step",
        "ground_truth": "Luxembourg, approximately 660,000",
        "requires_search": True,
        "n_steps": 2,
    },
    {
        "prompt": (
            "Find the average distance from Earth to the Moon in km, then find "
            "the speed of light in km/s, then compute how many seconds it takes "
            "light to travel from Earth to the Moon."
        ),
        "category": "multi_step",
        "ground_truth": "approximately 1.28 seconds",
        "requires_search": True,
        "n_steps": 3,
    },
    {
        "prompt": (
            "Find the founding year of Google, then calculate how many years "
            "ago that was from 2024."
        ),
        "category": "multi_step",
        "ground_truth": "26 years",
        "requires_search": True,
        "n_steps": 2,
    },
    {
        "prompt": (
            "Find the US inflation rate for March 2022, then find the unemployment "
            "rate for the same month, and tell me which figure was higher."
        ),
        "category": "multi_step",
        "ground_truth": "Inflation (8.0%) was higher than unemployment (3.6%)",
        "requires_search": True,
        "n_steps": 2,
    },
    # ── Failure recovery ────────────────────────────────────────────────────���──
    {
        "prompt": (
            "Search for information about the Xylofrobnic International Organization "
            "and summarize their 2024 annual report."
        ),
        "category": "failure_recovery",
        "ground_truth": "should_refuse",
        "requires_search": True,
        "expect_empty": True,
    },
    {
        "prompt": "Who is the current CEO of Zorblax Corporation?",
        "category": "failure_recovery",
        "ground_truth": "should_refuse",
        "requires_search": True,
        "expect_empty": True,
    },
    {
        "prompt": (
            "Look up the ranking of Quimbleton University in the 2024 Times Higher "
            "Education World University Rankings."
        ),
        "category": "failure_recovery",
        "ground_truth": "should_refuse",
        "requires_search": True,
        "expect_empty": True,
    },
]


# ── Trajectory generation ─────────────────────────────────────────────────────

def _build_generation_prompt(task: Dict, mock_search_fn) -> str:
    """Build a meta-prompt that asks Claude to write out the full trajectory."""
    search_note = ""
    if task.get("expect_empty"):
        search_note = (
            "\nNOTE: When you search for this entity, the search will return no results. "
            "Please demonstrate the correct behavior: try at least one rephrased query, "
            "then clearly state you could not find reliable information rather than guessing."
        )
    elif task.get("n_steps", 1) > 1:
        n = task["n_steps"]
        search_note = (
            f"\nNOTE: This task requires {n} separate search queries. "
            "Please show each search call and its result before moving to the next step."
        )

    return (
        f"Please demonstrate how a helpful AI assistant should answer the following question "
        f"using the ReAct format (Thought / Action / Observation / Answer).\n\n"
        f"Question: {task['prompt']}\n"
        f"{search_note}\n\n"
        f"Show the complete step-by-step trajectory, including the web_search tool calls "
        f"formatted as:\n"
        f"  Action: web_search(query=\"...\")\n"
        f"  Observation: [search result]\n\n"
        f"End with:\n"
        f"  Final answer: {task['ground_truth']}"
    )


def generate_trajectory(
    client,
    task: Dict,
    mock_search_fn=None,
    model: str = "claude-haiku-4-5-20251001",
    max_tokens: int = 800,
) -> Dict:
    """Generate a complete ReAct-format tool-use trajectory for one task.

    Parameters
    ----------
    client:
        Anthropic client.
    task:
        Task dict from AGENTIC_TASK_CATALOGUE.
    mock_search_fn:
        Optional callable(query) → str for injecting realistic search results.
        If None, Claude is asked to write plausible Observation text itself.
    model:
        Claude model to use for generation.

    Returns
    -------
    dict with keys: prompt, trajectory, category, ground_truth
    """
    system = TRAJECTORY_SYSTEM_PROMPT
    gen_prompt = _build_generation_prompt(task, mock_search_fn)

    for attempt in range(3):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": gen_prompt}],
            )
            trajectory_text = resp.content[0].text.strip()
            break
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)

    # Ensure trajectory ends with Final answer
    if "final answer:" not in trajectory_text.lower():
        trajectory_text += f"\nFinal answer: {task['ground_truth']}"

    return {
        "prompt": task["prompt"],
        "trajectory": trajectory_text,
        "category": task["category"],
        "ground_truth": task["ground_truth"],
        "requires_search": task.get("requires_search", True),
        "n_tool_calls": trajectory_text.lower().count("action: web_search"),
    }


# ── Batch generation pipeline ─────────────────────────────────────────────────

@dataclass
class AgenticSFTConfig:
    output_path: str = "data/agentic_sft.jsonl"
    model: str = "claude-haiku-4-5-20251001"
    max_tokens: int = 800
    # How many times to generate each task (diversity through temperature variation)
    generations_per_task: int = 3
    temperatures: List[float] = field(default_factory=lambda: [0.3, 0.6, 0.9])
    requests_per_minute: int = 40
    seed_tasks: Optional[List[Dict]] = None   # None = use AGENTIC_TASK_CATALOGUE


def generate_agentic_sft_dataset(cfg: AgenticSFTConfig) -> None:
    """Generate a JSONL of expert tool-use trajectories for SFT training.

    Each task in the catalogue is demonstrated `generations_per_task` times
    at different temperatures to add diversity. The resulting dataset teaches
    the model when to use tools, how to formulate queries, how to chain calls,
    and how to handle empty results gracefully.
    """
    import anthropic
    from tqdm.auto import tqdm

    client = anthropic.Anthropic()
    tasks = cfg.seed_tasks or AGENTIC_TASK_CATALOGUE
    rate_delay = 60.0 / cfg.requests_per_minute

    os.makedirs(os.path.dirname(cfg.output_path) or ".", exist_ok=True)
    results = []

    total = len(tasks) * cfg.generations_per_task
    pbar = tqdm(total=total, desc="Generating agentic trajectories")

    for task in tasks:
        for gen_idx in range(cfg.generations_per_task):
            try:
                result = generate_trajectory(
                    client, task,
                    model=cfg.model,
                    max_tokens=cfg.max_tokens,
                )
                results.append(result)
                pbar.set_postfix({
                    "cat": task["category"][:8],
                    "calls": result["n_tool_calls"],
                })
            except Exception as e:
                pbar.set_postfix({"error": str(e)[:40]})
            pbar.update(1)
            time.sleep(rate_delay)

    pbar.close()

    with open(cfg.output_path, "w") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")

    cat_counts = {}
    for r in results:
        cat_counts[r["category"]] = cat_counts.get(r["category"], 0) + 1

    print(f"\nGenerated {len(results)} agentic trajectories → {cfg.output_path}")
    print(f"Category breakdown: {cat_counts}")
    avg_calls = sum(r["n_tool_calls"] for r in results) / max(len(results), 1)
    print(f"Average tool calls per trajectory: {avg_calls:.1f}")


# ── Dataset class ─────────────────────────────────────────────────────────────

class AgenticSFTDataset(Dataset):
    """SFT dataset of expert tool-use trajectories.

    Each example is formatted as a full Human/Assistant turn where the
    assistant's response IS the complete ReAct trajectory.  The model learns
    to reproduce the Thought/Action/Observation/Answer scaffold as its
    default response to tool-requiring questions.

    Compared to conversational SFT (hh-rlhf):
    - The target sequence is much longer (trajectory vs. one response)
    - The target contains explicit tool calls and observations
    - The model learns the ReAct scaffold as a generative pattern, not just
      as a prompt-injected instruction

    This is the key mechanism: conversational SFT teaches *what* to say;
    agentic SFT teaches *how* to decide, search, and synthesise.
    """

    HUMAN_PREFIX = "Human: "
    ASSISTANT_PREFIX = "\n\nAssistant: "

    def __init__(
        self,
        jsonl_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 768,
        categories: Optional[List[str]] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data: List[Dict] = []

        with open(jsonl_path) as f:
            for line in f:
                item = json.loads(line)
                if categories and item.get("category") not in categories:
                    continue
                if item.get("trajectory"):
                    self.data.append(item)

        print(
            f"Loaded {len(self.data)} agentic SFT trajectories from {jsonl_path}"
            + (f" (categories: {categories})" if categories else "")
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        prompt_text = self.HUMAN_PREFIX + item["prompt"]
        full_text = prompt_text + self.ASSISTANT_PREFIX + item["trajectory"]

        enc = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        # Mask prompt tokens — loss only on the trajectory (assistant turn)
        prompt_enc = self.tokenizer(
            prompt_text + self.ASSISTANT_PREFIX,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        prompt_len = prompt_enc["input_ids"].shape[1]

        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
