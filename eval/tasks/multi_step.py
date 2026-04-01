"""
Category 2: Multi-Step Reasoning with Tool Calls (12 tasks)

Each task requires chaining 2-4 tool calls with intermediate reasoning.
The agent must extract information from one call and use it to formulate
the next query — the "context threading" challenge.

What makes this interesting:
  - Process-level metric: is the tool-call *sequence* correct?
    (analogous to PRM vs ORM in the reward modeling section)
  - Even if the final answer is wrong, a partially-correct sequence earns
    partial sequence_score
  - First call result must inform the second call's query

Ground truth: final answer (name, number, or entity)
Scoring: answer = token_f1 or numeric_match; sequence = sequence_match
Expected sequences: 2-step ["web_search", "web_search"], etc.
"""

from __future__ import annotations

from eval.tasks.base import EvalTask
from eval.scorers import (
    numeric_match,
    sequence_match,
    substring_match,
    token_f1,
)

MULTI_STEP_TASKS = [
    EvalTask(
        task_id="ms_001",
        category="multi_step",
        prompt=(
            "Find the company that acquired DeepMind, then find the name of "
            "that company's CEO as of 2023."
        ),
        ground_truth="Sundar Pichai",
        scorer=token_f1,
        sequence_scorer=sequence_match(["web_search", "web_search"]),
        expected_tool_sequence=["web_search", "web_search"],
        notes=(
            "Classic 2-hop reasoning: DeepMind → Alphabet → Sundar Pichai. "
            "Agent must pass 'Alphabet' from call 1 into call 2."
        ),
    ),

    EvalTask(
        task_id="ms_002",
        category="multi_step",
        prompt=(
            "Find the current CEO of the company that acquired DeepMind, "
            "then find that CEO's estimated net worth."
        ),
        ground_truth="1.3 billion",
        scorer=numeric_match(tolerance=0.3),
        sequence_scorer=sequence_match(["web_search", "web_search", "web_search"]),
        expected_tool_sequence=["web_search", "web_search", "web_search"],
        notes=(
            "3-hop chain: DeepMind → Alphabet → Sundar Pichai → net worth. "
            "Each call depends on the previous result."
        ),
    ),

    EvalTask(
        task_id="ms_003",
        category="multi_step",
        prompt=(
            "Find the world's largest company by revenue in 2023, then find "
            "who founded that company."
        ),
        ground_truth="Sam Walton",
        scorer=token_f1,
        sequence_scorer=sequence_match(["web_search", "web_search"]),
        expected_tool_sequence=["web_search", "web_search"],
        notes="2-hop: largest by revenue (Walmart) → founder (Sam Walton).",
    ),

    EvalTask(
        task_id="ms_004",
        category="multi_step",
        prompt=(
            "Find the world's largest company by revenue in 2023, then find "
            "who founded it, then find that person's net worth at time of death."
        ),
        ground_truth="8.6 billion",
        scorer=numeric_match(tolerance=0.3),
        sequence_scorer=sequence_match(["web_search", "web_search", "web_search"]),
        expected_tool_sequence=["web_search", "web_search", "web_search"],
        notes="3-hop: Walmart → Sam Walton → net worth at death ($8.6B).",
    ),

    EvalTask(
        task_id="ms_005",
        category="multi_step",
        prompt=(
            "Find the country with the highest GDP per capita in 2023, "
            "then find its approximate population."
        ),
        ground_truth="660000",
        scorer=numeric_match(tolerance=0.1),
        sequence_scorer=sequence_match(["web_search", "web_search"]),
        expected_tool_sequence=["web_search", "web_search"],
        notes=(
            "2-hop: highest GDP/capita (Luxembourg) → Luxembourg population (~660k). "
            "Small country — many agents will hallucinate a much larger number."
        ),
    ),

    EvalTask(
        task_id="ms_006",
        category="multi_step",
        prompt=(
            "Find which country was first to land humans on the Moon, "
            "then find the birth year of the first astronaut to walk on it."
        ),
        ground_truth="1930",
        scorer=numeric_match(tolerance=0.0),
        sequence_scorer=sequence_match(["web_search", "web_search"]),
        expected_tool_sequence=["web_search", "web_search"],
        notes="2-hop: US Moon landing → Neil Armstrong → born 1930.",
    ),

    EvalTask(
        task_id="ms_007",
        category="multi_step",
        prompt=(
            "Find who invented email and in what year, then find when "
            "that person was born."
        ),
        ground_truth="1941",
        scorer=numeric_match(tolerance=0.0),
        sequence_scorer=sequence_match(["web_search", "web_search"]),
        expected_tool_sequence=["web_search", "web_search"],
        notes="2-hop: email inventor (Ray Tomlinson, 1971) → Tomlinson birth year (1941).",
    ),

    EvalTask(
        task_id="ms_008",
        category="multi_step",
        prompt=(
            "Look up the abstract of the RLHF paper from the ML document store, "
            "then search the web to find what year InstructGPT was published."
        ),
        ground_truth="2022",
        scorer=numeric_match(tolerance=0.0),
        sequence_scorer=sequence_match(["retrieve_document", "web_search"]),
        expected_tool_sequence=["retrieve_document", "web_search"],
        notes=(
            "Mixed tool chain: document retrieval → web search. "
            "Tests whether the agent can use different tool types in sequence."
        ),
    ),

    EvalTask(
        task_id="ms_009",
        category="multi_step",
        prompt=(
            "Search for the founding year of Google, then calculate how many "
            "years ago that was from 2024 and give me the number."
        ),
        ground_truth="26",
        scorer=numeric_match(tolerance=0.0),
        sequence_scorer=sequence_match(["web_search"]),
        expected_tool_sequence=["web_search"],
        notes=(
            "1 tool call + arithmetic reasoning. Agent must search (1998) then "
            "compute 2024 - 1998 = 26 without a second search."
        ),
    ),

    EvalTask(
        task_id="ms_010",
        category="multi_step",
        prompt=(
            "Find the CEO of Alphabet in 2023, then search for that person's "
            "role before becoming CEO of Google."
        ),
        ground_truth="senior vice president",
        scorer=substring_match,
        sequence_scorer=sequence_match(["web_search", "web_search"]),
        expected_tool_sequence=["web_search", "web_search"],
        notes=(
            "2-hop requiring specific role lookup. Tests whether the agent "
            "correctly threads the name from call 1 into the role query."
        ),
    ),

    EvalTask(
        task_id="ms_011",
        category="multi_step",
        prompt=(
            "Find the average distance from Earth to the Moon in km, "
            "then find the speed of light in km/s, then compute how many "
            "seconds it takes light to travel from Earth to the Moon."
        ),
        ground_truth="1.28",
        scorer=numeric_match(tolerance=0.05),
        sequence_scorer=sequence_match(["web_search", "web_search"]),
        expected_tool_sequence=["web_search", "web_search"],
        notes=(
            "2 tool calls + arithmetic: 384400 km / 299792 km/s ≈ 1.28 seconds. "
            "Tests multi-step reasoning interleaved with tool calls."
        ),
    ),

    EvalTask(
        task_id="ms_012",
        category="multi_step",
        prompt=(
            "Find the US inflation rate for March 2022, then find the "
            "unemployment rate for the same month, and tell me which was higher."
        ),
        ground_truth="inflation",
        scorer=substring_match,
        sequence_scorer=sequence_match(["web_search", "web_search"]),
        expected_tool_sequence=["web_search", "web_search"],
        notes=(
            "2-hop comparison: inflation 8.0% vs unemployment 3.6%. "
            "Agent must compare two retrieved numbers correctly."
        ),
    ),
]
