"""
Category 1: Tool Use and Retrieval (12 tasks)

Each task requires the agent to:
  1. Decide to call the search tool rather than answer from parametric memory
  2. Parse the (imperfect) search result to extract the specific value asked for
  3. Return a clean, specific answer — not a hallucination from training data

What makes this hard:
  - The agent must learn *when* to call the tool (vs. answering from memory)
  - Search results return paragraphs; the agent must extract the right number
  - Some results may mention a related statistic before the exact one asked for

Ground truth format: exact value (string or numeric)
Scoring: numeric_match (±5% tolerance) for numbers, token_f1 for names/facts
Expected sequence: ["web_search"] — one tool call suffices
"""

from __future__ import annotations

from eval.tasks.base import EvalTask
from eval.scorers import (
    at_least_one_call,
    exact_match,
    numeric_match,
    sequence_match,
    substring_match,
    token_f1,
)

TOOL_USE_TASKS = [
    EvalTask(
        task_id="tu_001",
        category="tool_use",
        prompt=(
            "What was the unemployment rate in the United States in March 2022? "
            "Give the exact percentage."
        ),
        ground_truth="3.6%",
        scorer=numeric_match(tolerance=0.05),
        sequence_scorer=at_least_one_call("web_search"),
        expected_tool_sequence=["web_search"],
        notes="Classic factual lookup. Agent must search, not hallucinate 2022 figure.",
    ),

    EvalTask(
        task_id="tu_002",
        category="tool_use",
        prompt="What was the annual US GDP growth rate in 2023? Give just the percentage.",
        ground_truth="2.5%",
        scorer=numeric_match(tolerance=0.1),
        sequence_scorer=at_least_one_call("web_search"),
        expected_tool_sequence=["web_search"],
        notes="Requires recent data beyond training cutoff of smaller models.",
    ),

    EvalTask(
        task_id="tu_003",
        category="tool_use",
        prompt="What is the approximate population of India as of 2023?",
        ground_truth="1.43 billion",
        scorer=numeric_match(tolerance=0.05),
        sequence_scorer=at_least_one_call("web_search"),
        expected_tool_sequence=["web_search"],
        notes="Agent should search rather than recall a potentially stale figure.",
    ),

    EvalTask(
        task_id="tu_004",
        category="tool_use",
        prompt="What is the capital of France?",
        ground_truth="Paris",
        scorer=exact_match,
        sequence_scorer=sequence_match([]),  # agent may or may not call search
        expected_tool_sequence=[],
        notes=(
            "Control task: any capable model knows this. "
            "Measures whether agent wastes a tool call on a trivial question."
        ),
    ),

    EvalTask(
        task_id="tu_005",
        category="tool_use",
        prompt="At what temperature does water boil in Celsius at standard atmospheric pressure?",
        ground_truth="100",
        scorer=numeric_match(tolerance=0.01),
        sequence_scorer=sequence_match([]),
        expected_tool_sequence=[],
        notes="Constant — should be answered from memory, no tool call needed.",
    ),

    EvalTask(
        task_id="tu_006",
        category="tool_use",
        prompt="How tall is Mount Everest in meters? Use the most recent official measurement.",
        ground_truth="8848.86",
        scorer=numeric_match(tolerance=0.001),
        sequence_scorer=at_least_one_call("web_search"),
        expected_tool_sequence=["web_search"],
        notes="2020 survey produced a new official height; older models may have stale data.",
    ),

    EvalTask(
        task_id="tu_007",
        category="tool_use",
        prompt="Who was the CEO of Apple Inc. in 2023?",
        ground_truth="Tim Cook",
        scorer=token_f1,
        sequence_scorer=at_least_one_call("web_search"),
        expected_tool_sequence=["web_search"],
        notes="Named entity retrieval. Agent should confirm via search.",
    ),

    EvalTask(
        task_id="tu_008",
        category="tool_use",
        prompt="In what year was Google founded?",
        ground_truth="1998",
        scorer=numeric_match(tolerance=0.0),
        sequence_scorer=sequence_match([]),
        expected_tool_sequence=[],
        notes="Well-known fact that may or may not need tool use.",
    ),

    EvalTask(
        task_id="tu_009",
        category="tool_use",
        prompt="What is the average distance from the Earth to the Moon in kilometers?",
        ground_truth="384400",
        scorer=numeric_match(tolerance=0.01),
        sequence_scorer=at_least_one_call("web_search"),
        expected_tool_sequence=["web_search"],
        notes="Exact figure — agent should verify rather than approximate.",
    ),

    EvalTask(
        task_id="tu_010",
        category="tool_use",
        prompt="What is the speed of light in a vacuum in meters per second?",
        ground_truth="299792458",
        scorer=numeric_match(tolerance=0.001),
        sequence_scorer=sequence_match([]),
        expected_tool_sequence=[],
        notes="Physical constant — should be recalled, not searched.",
    ),

    EvalTask(
        task_id="tu_011",
        category="tool_use",
        prompt=(
            "What was the US Consumer Price Index (CPI) inflation rate for the "
            "12 months ending March 2022?"
        ),
        ground_truth="8.0%",
        scorer=numeric_match(tolerance=0.1),
        sequence_scorer=at_least_one_call("web_search"),
        expected_tool_sequence=["web_search"],
        notes="Requires search — specific month and year disambiguates from general knowledge.",
    ),

    EvalTask(
        task_id="tu_012",
        category="tool_use",
        prompt="Retrieve information from the ML glossary about what overfitting means.",
        ground_truth="model memorises training data fails to generalise",
        scorer=token_f1,
        sequence_scorer=at_least_one_call("retrieve_document"),
        expected_tool_sequence=["retrieve_document"],
        notes="Document retrieval task — agent must use retrieve_document, not web_search.",
    ),
]
