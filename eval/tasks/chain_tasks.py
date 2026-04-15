"""
Synthetic chain tasks for the context-window ablation (Extension 13 addendum).

Each task is a linear N-hop chain where each hop's answer is used as the
entity for the next lookup. The chain uses interconnected facts in the mock
search DB:

  Hop 1: CEO of Alphabet 2023          → "Sundar Pichai"
  Hop 2: Sundar Pichai birth year      → "1972"
  Hop 3: US President in 1972          → "Richard Nixon"
  Hop 4: Year Nixon resigned           → "1974"
  Hop 5: US President who succeeded Nixon → "Gerald Ford"
  Hop 6: Gerald Ford's birth state     → "Nebraska"
  Hop 7: Capital of Nebraska           → "Lincoln"
  Hop 8: Year Abraham Lincoln assassinated → "1865"

We construct four tasks: 2-hop, 4-hop, 6-hop, 8-hop. Each task's ground truth
is the final hop's answer. This lets the ablation measure whether the
coordinator maintains accuracy as chain depth increases.

These tasks are used by scripts/run_context_ablation.py — they are NOT
included in the standard AgentBench-Mini 36-task run.
"""

from __future__ import annotations

from eval.tasks.base import EvalTask
from eval.scorers import exact_match, numeric_match

# ── Chain structure ────────────────────────────────────────────────────────────
# Each step: (description, search_query, expected_answer)

_CHAIN = [
    ("Find the CEO of Alphabet in 2023",
     "CEO Alphabet 2023",
     "Sundar Pichai"),
    ("Find the birth year of Sundar Pichai",
     "Sundar Pichai birth year",
     "1972"),
    ("Find the US President in 1972",
     "US President 1972",
     "Richard Nixon"),
    ("Find the year Richard Nixon resigned",
     "year Richard Nixon resigned",
     "1974"),
    ("Find the US President who succeeded Nixon in 1974",
     "US President succeeded Nixon 1974",
     "Gerald Ford"),
    ("Find Gerald Ford's birth state",
     "Gerald Ford birth state",
     "Nebraska"),
    ("Find the capital of Nebraska",
     "capital of Nebraska",
     "Lincoln"),
    ("Find the year Abraham Lincoln was assassinated",
     "year Abraham Lincoln assassinated",
     "1865"),
]


def _make_chain_prompt(n_hops: int) -> str:
    steps = _CHAIN[:n_hops]
    chain_desc = "\n".join(
        f"  {i+1}. {desc}" for i, (desc, _, _) in enumerate(steps)
    )
    return (
        f"Answer this {n_hops}-step chain: starting from the first fact, "
        f"each answer feeds into the next lookup.\n\n"
        f"Steps:\n{chain_desc}\n\n"
        f"Report only the final answer."
    )


CHAIN_2HOP = EvalTask(
    task_id="chain_002",
    category="chain",
    prompt=_make_chain_prompt(2),
    ground_truth="1972",
    scorer=exact_match,
    notes="2-hop: Alphabet CEO → birth year of CEO",
)

CHAIN_4HOP = EvalTask(
    task_id="chain_004",
    category="chain",
    prompt=_make_chain_prompt(4),
    ground_truth="1974",
    scorer=exact_match,
    notes="4-hop: Alphabet CEO → birth year → US president → year resigned",
)

CHAIN_6HOP = EvalTask(
    task_id="chain_006",
    category="chain",
    prompt=_make_chain_prompt(6),
    ground_truth="Nebraska",
    scorer=exact_match,
    notes="6-hop: Alphabet CEO → … → president's successor → birth state",
)

CHAIN_8HOP = EvalTask(
    task_id="chain_008",
    category="chain",
    prompt=_make_chain_prompt(8),
    ground_truth="1865",
    scorer=exact_match,
    notes="8-hop: Alphabet CEO → … → state capital → Lincoln assassination year",
)

CHAIN_TASKS = [CHAIN_2HOP, CHAIN_4HOP, CHAIN_6HOP, CHAIN_8HOP]
