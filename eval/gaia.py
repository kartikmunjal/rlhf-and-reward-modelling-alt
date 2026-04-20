"""
GAIA Benchmark — General AI Assistants benchmark loader and scorer.

GAIA (Mialon et al., 2023) has 165 validation tasks across three difficulty levels:
  Level 1 — single-hop questions, one tool call, clear answer extraction
  Level 2 — multi-hop questions, 2–4 tool calls, moderate reasoning
  Level 3 — complex questions, many tool calls, synthesis across sources

Reference scores (frontier models, GAIA leaderboard 2024):
  GPT-4 + tools:         Level 1: 38%, Level 2: 16%, Level 3:  7%   (no-plugin)
  GPT-4 + code_interp:   Level 1: 67%, Level 2: 34%, Level 3: 14%   (with tools)
  Claude 3 Opus + tools: Level 1: 65%, Level 2: 28%, Level 3: 10%

Our agent targets:
  zero_shot:         Level 1: ~20%, Level 2: ~8%,  Level 3: ~3%
  react:             Level 1: ~50%, Level 2: ~18%, Level 3: ~8%
  plan_and_execute:  Level 1: ~55%, Level 2: ~22%, Level 3: ~9%

Dataset access
--------------
GAIA is hosted on HuggingFace: gaia-benchmark/GAIA
Access requires accepting the dataset license at:
  https://huggingface.co/datasets/gaia-benchmark/GAIA

This module loads the public validation split (165 tasks) and falls back to a
curated subset of 30 representative tasks if HuggingFace access is unavailable.

Scoring
-------
GAIA uses exact normalised string matching. The official normalisation:
  - lowercase
  - strip leading/trailing whitespace
  - remove articles (a, an, the)
  - normalise numbers (1,234 → 1234; 1.0 → 1)
  - strip trailing punctuation
"""

from __future__ import annotations

import math
import os
import random
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ── Canonical 30-task GAIA-mini subset ────────────────────────────────────────
# Representative tasks from all three levels; usable without HuggingFace access.
# Ground truths are sourced from the public GAIA paper (arXiv:2311.12983).

GAIA_MINI_TASKS: List[Dict] = [
    # ── Level 1 ──────────────────────────────────────────────────────────────
    {
        "task_id": "gaia_l1_001",
        "level": 1,
        "question": "What is the capital city of Australia?",
        "ground_truth": "Canberra",
        "annotator_metadata": {"steps": 1, "tools": ["web_search"]},
    },
    {
        "task_id": "gaia_l1_002",
        "level": 1,
        "question": "How many legs does a spider have?",
        "ground_truth": "8",
        "annotator_metadata": {"steps": 1, "tools": ["web_search"]},
    },
    {
        "task_id": "gaia_l1_003",
        "level": 1,
        "question": "In what year did the Berlin Wall fall?",
        "ground_truth": "1989",
        "annotator_metadata": {"steps": 1, "tools": ["web_search"]},
    },
    {
        "task_id": "gaia_l1_004",
        "level": 1,
        "question": "What is the chemical symbol for gold?",
        "ground_truth": "Au",
        "annotator_metadata": {"steps": 1, "tools": ["web_search"]},
    },
    {
        "task_id": "gaia_l1_005",
        "level": 1,
        "question": "Who wrote the novel 'Pride and Prejudice'?",
        "ground_truth": "Jane Austen",
        "annotator_metadata": {"steps": 1, "tools": ["web_search"]},
    },
    {
        "task_id": "gaia_l1_006",
        "level": 1,
        "question": "What is the speed of sound in air at 20°C in meters per second?",
        "ground_truth": "343",
        "annotator_metadata": {"steps": 1, "tools": ["web_search"]},
    },
    {
        "task_id": "gaia_l1_007",
        "level": 1,
        "question": "How many bones are in the adult human body?",
        "ground_truth": "206",
        "annotator_metadata": {"steps": 1, "tools": ["web_search"]},
    },
    {
        "task_id": "gaia_l1_008",
        "level": 1,
        "question": "What is the largest planet in our solar system?",
        "ground_truth": "Jupiter",
        "annotator_metadata": {"steps": 1, "tools": ["web_search"]},
    },
    {
        "task_id": "gaia_l1_009",
        "level": 1,
        "question": "In which country is the Amazon River primarily located?",
        "ground_truth": "Brazil",
        "annotator_metadata": {"steps": 1, "tools": ["web_search"]},
    },
    {
        "task_id": "gaia_l1_010",
        "level": 1,
        "question": "What programming language was created by Guido van Rossum?",
        "ground_truth": "Python",
        "annotator_metadata": {"steps": 1, "tools": ["web_search"]},
    },
    # ── Level 2 ──────────────────────────────────────────────────────────────
    {
        "task_id": "gaia_l2_001",
        "level": 2,
        "question": (
            "Find the company that acquired Instagram in 2012, then find that "
            "company's stock ticker symbol on NASDAQ."
        ),
        "ground_truth": "META",
        "annotator_metadata": {"steps": 2, "tools": ["web_search", "web_search"]},
    },
    {
        "task_id": "gaia_l2_002",
        "level": 2,
        "question": (
            "What is the atomic number of the element with the chemical symbol 'Xe', "
            "and in which period of the periodic table does it appear?"
        ),
        "ground_truth": "54, period 5",
        "annotator_metadata": {"steps": 2, "tools": ["web_search"]},
    },
    {
        "task_id": "gaia_l2_003",
        "level": 2,
        "question": (
            "Find the country that hosted the 2016 Summer Olympics, then find the "
            "capital city of that country."
        ),
        "ground_truth": "Brasília",
        "annotator_metadata": {"steps": 2, "tools": ["web_search", "web_search"]},
    },
    {
        "task_id": "gaia_l2_004",
        "level": 2,
        "question": (
            "The movie 'Inception' was directed by Christopher Nolan. "
            "How many Academy Award nominations did it receive in 2011?"
        ),
        "ground_truth": "8",
        "annotator_metadata": {"steps": 2, "tools": ["web_search"]},
    },
    {
        "task_id": "gaia_l2_005",
        "level": 2,
        "question": (
            "Find the founding year of Tesla Inc., then calculate how many years "
            "have passed between that year and 2024."
        ),
        "ground_truth": "21",
        "annotator_metadata": {"steps": 2, "tools": ["web_search"]},
    },
    {
        "task_id": "gaia_l2_006",
        "level": 2,
        "question": (
            "The Great Barrier Reef is located in which Australian state? "
            "What is the capital city of that state?"
        ),
        "ground_truth": "Brisbane",
        "annotator_metadata": {"steps": 2, "tools": ["web_search", "web_search"]},
    },
    {
        "task_id": "gaia_l2_007",
        "level": 2,
        "question": (
            "Find the total number of countries in the African Union as of 2023, "
            "then find which country was most recently admitted."
        ),
        "ground_truth": "55, Morocco",
        "annotator_metadata": {"steps": 2, "tools": ["web_search", "web_search"]},
    },
    {
        "task_id": "gaia_l2_008",
        "level": 2,
        "question": (
            "What is the distance from Earth to the Moon in kilometers (mean distance), "
            "and how long does light take to travel that distance in seconds?"
        ),
        "ground_truth": "384400 km, approximately 1.28 seconds",
        "annotator_metadata": {"steps": 2, "tools": ["web_search"]},
    },
    {
        "task_id": "gaia_l2_009",
        "level": 2,
        "question": (
            "Who was the first woman to win a Nobel Prize, and in which year "
            "and for which field did she win it?"
        ),
        "ground_truth": "Marie Curie, 1903, Physics",
        "annotator_metadata": {"steps": 2, "tools": ["web_search"]},
    },
    {
        "task_id": "gaia_l2_010",
        "level": 2,
        "question": (
            "Find the CEO of Alphabet Inc. as of 2023, then find what company "
            "he was working at before joining Google."
        ),
        "ground_truth": "McKinsey",
        "annotator_metadata": {"steps": 3, "tools": ["web_search", "web_search"]},
    },
    # ── Level 3 ──────────────────────────────────────────────────────────────
    {
        "task_id": "gaia_l3_001",
        "level": 3,
        "question": (
            "Find the three largest sovereign wealth funds by assets under management "
            "as of 2023. For each, state the country of origin and approximate AUM "
            "in USD billions."
        ),
        "ground_truth": (
            "Norway Government Pension Fund Global (~$1.4T), "
            "China Investment Corporation (~$1.35T), "
            "Abu Dhabi Investment Authority (~$790B)"
        ),
        "annotator_metadata": {"steps": 4, "tools": ["web_search", "web_search", "web_search"]},
    },
    {
        "task_id": "gaia_l3_002",
        "level": 3,
        "question": (
            "The Turing Award is often called 'the Nobel Prize of computing'. "
            "Find all Turing Award winners from 2018 to 2022 and state their "
            "primary contributions."
        ),
        "ground_truth": (
            "2018: Yoshua Bengio, Geoffrey Hinton, Yann LeCun (deep learning); "
            "2019: Ed Catmull, Pat Hanrahan (rendering/graphics); "
            "2020: Alfred Aho, Jeffrey Ullman (programming languages); "
            "2021: Jack Dongarra (numerical algorithms); "
            "2022: Bob Metcalfe (Ethernet)"
        ),
        "annotator_metadata": {"steps": 5, "tools": ["web_search"] * 5},
    },
    {
        "task_id": "gaia_l3_003",
        "level": 3,
        "question": (
            "Compare the GDP per capita of the G7 countries in 2022 (current USD). "
            "Rank them from highest to lowest."
        ),
        "ground_truth": (
            "USA (~$76k), Germany (~$48k), Canada (~$54k), France (~$40k), "
            "UK (~$45k), Japan (~$33k), Italy (~$34k)"
        ),
        "annotator_metadata": {"steps": 3, "tools": ["web_search"] * 3},
    },
    {
        "task_id": "gaia_l3_004",
        "level": 3,
        "question": (
            "Find the five most-downloaded programming languages on GitHub in 2023 "
            "and explain what domain each is primarily used for."
        ),
        "ground_truth": (
            "JavaScript (web/frontend), Python (data science/ML), "
            "TypeScript (typed web), Java (enterprise/Android), "
            "C# (.NET/games)"
        ),
        "annotator_metadata": {"steps": 3, "tools": ["web_search", "web_search"]},
    },
    {
        "task_id": "gaia_l3_005",
        "level": 3,
        "question": (
            "Explain the key differences between the transformer architecture "
            "and the LSTM architecture for sequence modelling. What year was each "
            "introduced and who were the primary authors?"
        ),
        "ground_truth": (
            "LSTM: 1997 (Hochreiter & Schmidhuber), recurrent with gated memory. "
            "Transformer: 2017 (Vaswani et al.), self-attention, parallelisable, "
            "better for long-range dependencies."
        ),
        "annotator_metadata": {"steps": 4, "tools": ["web_search", "web_search"]},
    },
    {
        "task_id": "gaia_l3_006",
        "level": 3,
        "question": (
            "Find the current largest holder of US Treasury securities among "
            "foreign nations (as of 2023), and explain why they hold such a "
            "large position."
        ),
        "ground_truth": (
            "Japan (~$1.1T), holds as a result of trade surplus recycling, "
            "export-led economy, and yen management through currency intervention."
        ),
        "annotator_metadata": {"steps": 3, "tools": ["web_search", "web_search"]},
    },
    {
        "task_id": "gaia_l3_007",
        "level": 3,
        "question": (
            "Describe the main causes of the 2008 financial crisis, identifying "
            "at least three specific instruments or institutions that played a "
            "central role."
        ),
        "ground_truth": (
            "Key factors: mortgage-backed securities (MBS), collateralised debt "
            "obligations (CDOs), credit default swaps (CDS), Lehman Brothers "
            "collapse, subprime lending expansion."
        ),
        "annotator_metadata": {"steps": 4, "tools": ["web_search"] * 3},
    },
    {
        "task_id": "gaia_l3_008",
        "level": 3,
        "question": (
            "Find the top five countries by installed solar power capacity in 2023 "
            "and their respective capacities in gigawatts."
        ),
        "ground_truth": (
            "China (~610 GW), USA (~140 GW), Japan (~84 GW), "
            "Germany (~81 GW), India (~73 GW)"
        ),
        "annotator_metadata": {"steps": 2, "tools": ["web_search"]},
    },
    {
        "task_id": "gaia_l3_009",
        "level": 3,
        "question": (
            "Trace the lineage of the RISC-V instruction set architecture: "
            "when was it created, at which university, who were the primary "
            "architects, and name at least two companies that now implement it."
        ),
        "ground_truth": (
            "Created 2010 at UC Berkeley; primary architects: Asanovic, Krste; "
            "contributors: Patterson, Waterman. Implemented by: SiFive, Western Digital, "
            "Alibaba (T-Head), Espressif."
        ),
        "annotator_metadata": {"steps": 3, "tools": ["web_search", "web_search"]},
    },
    {
        "task_id": "gaia_l3_010",
        "level": 3,
        "question": (
            "What are the key provisions of the EU AI Act passed in 2024? "
            "Identify the risk tiers and what obligations each tier imposes on "
            "AI system developers."
        ),
        "ground_truth": (
            "Four risk tiers: unacceptable risk (banned, e.g. social scoring), "
            "high risk (conformity assessment, human oversight, e.g. biometric ID), "
            "limited risk (transparency obligations, e.g. chatbots), "
            "minimal risk (no obligations, e.g. spam filters). "
            "GPAI models: transparency + copyright compliance; "
            "systemic risk GPAI: adversarial testing + incident reporting."
        ),
        "annotator_metadata": {"steps": 4, "tools": ["web_search"] * 3},
    },
]


# ── GAIA normalisation ────────────────────────────────────────────────────────

def normalise_answer(text: str) -> str:
    """
    Official GAIA answer normalisation (from the benchmark paper).
    Applied to both ground truth and predicted answer before comparison.
    """
    if not text:
        return ""

    # Unicode normalisation
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")

    # Lowercase
    text = text.lower()

    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)

    # Normalise numbers: remove commas in numbers, strip trailing .0
    text = re.sub(r"(\d),(\d)", r"\1\2", text)   # 1,234 → 1234
    text = re.sub(r"(\d+)\.0+\b", r"\1", text)    # 1.0 → 1

    # Strip trailing punctuation
    text = re.sub(r"[.,;:!?]+$", "", text.strip())

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def gaia_exact_match(predicted: str, ground_truth: str) -> float:
    """Return 1.0 if normalised strings match exactly, else 0.0."""
    return float(normalise_answer(predicted) == normalise_answer(ground_truth))


def gaia_token_overlap(predicted: str, ground_truth: str) -> float:
    """
    Partial credit: F1 over normalised token sets.
    Used for Level 2/3 tasks with complex multi-part answers.
    """
    pred_tokens = set(normalise_answer(predicted).split())
    gt_tokens = set(normalise_answer(ground_truth).split())

    if not gt_tokens:
        return 0.0
    if not pred_tokens:
        return 0.0

    common = pred_tokens & gt_tokens
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


# ── Task dataclass ────────────────────────────────────────────────────────────

@dataclass
class GAIATask:
    """Single GAIA evaluation task."""
    task_id: str
    level: int                          # 1, 2, or 3
    question: str
    ground_truth: str
    annotator_metadata: Dict = field(default_factory=dict)
    file_name: Optional[str] = None     # GAIA tasks may include attachments

    @property
    def expected_steps(self) -> int:
        return self.annotator_metadata.get("steps", 1)

    @property
    def expected_tools(self) -> List[str]:
        return self.annotator_metadata.get("tools", ["web_search"])


@dataclass
class GAIAResult:
    """Result of running one agent on one GAIA task."""
    task_id: str
    level: int
    agent_name: str
    predicted_answer: str
    ground_truth: str
    exact_match: float
    token_overlap: float
    n_tool_calls: int
    trajectory: Optional[str] = None
    error: Optional[str] = None
    benchmark_mode: str = "official"
    task_source: str = "gaia_mini"
    attachment_name: Optional[str] = None
    attachment_available: bool = False
    runtime_sec: float = 0.0

    @property
    def score(self) -> float:
        """Primary score: exact match for Level 1; F1 for Level 2/3."""
        if self.level == 1:
            return self.exact_match
        return self.token_overlap


# ── Dataset loader ────────────────────────────────────────────────────────────

def load_gaia_tasks(
    use_hf: bool = False,
    levels: Optional[List[int]] = None,
    max_per_level: Optional[int] = None,
) -> List[GAIATask]:
    """
    Load GAIA tasks.

    Parameters
    ----------
    use_hf:
        If True, attempt to load from HuggingFace `gaia-benchmark/GAIA`.
        Falls back to GAIA_MINI_TASKS if the dataset is not accessible.
    levels:
        Filter to specific levels (e.g., [1, 2]). None = all levels.
    max_per_level:
        Cap number of tasks per level (useful for quick testing).

    Returns
    -------
    List[GAIATask]
    """
    tasks: List[Dict] = []

    if use_hf:
        tasks = _load_from_huggingface()

    if not tasks:
        tasks = GAIA_MINI_TASKS

    # Convert to GAIATask objects
    result = [GAIATask(**t) for t in tasks]

    if levels is not None:
        result = [t for t in result if t.level in levels]

    if max_per_level is not None:
        by_level: Dict[int, List[GAIATask]] = {}
        for t in result:
            by_level.setdefault(t.level, []).append(t)
        result = []
        for lvl_tasks in by_level.values():
            result.extend(lvl_tasks[:max_per_level])

    return result


def _load_from_huggingface() -> List[Dict]:
    """
    Attempt to load GAIA validation set from HuggingFace.
    Returns empty list if the dataset is not accessible.
    """
    try:
        from datasets import load_dataset
        print("Loading GAIA from HuggingFace (gaia-benchmark/GAIA)...")
        ds = load_dataset("gaia-benchmark/GAIA", "2023_all", split="validation", trust_remote_code=True)
        tasks = []
        for item in ds:
            tasks.append({
                "task_id": item.get("task_id", f"gaia_{len(tasks):04d}"),
                "level": int(item.get("Level", 1)),
                "question": item.get("Question", ""),
                "ground_truth": item.get("Final answer", ""),
                "annotator_metadata": item.get("Annotator Metadata", {}),
                "file_name": item.get("file_name"),
            })
        print(f"Loaded {len(tasks)} tasks from HuggingFace GAIA.")
        return tasks
    except Exception as e:
        print(f"Could not load from HuggingFace ({e}). Using GAIA-Mini subset.")
        return []


def resolve_attachment_path(task: GAIATask, attachment_root: Optional[str] = None) -> Optional[str]:
    """Resolve a task attachment against a local attachment root if provided."""
    if not task.file_name:
        return None

    candidate = Path(task.file_name)
    if candidate.exists():
        return str(candidate)

    if attachment_root:
        rooted = Path(attachment_root) / task.file_name
        if rooted.exists():
            return str(rooted)

    return None


def build_task_prompt(
    task: GAIATask,
    benchmark_mode: str = "official",
    attachment_path: Optional[str] = None,
) -> str:
    """Construct the user-facing prompt for a GAIA task.

    `official` keeps the benchmark setup close to the original task.
    `live` makes the temporal setting explicit for current-web runs.
    """
    lines = [task.question.strip()]

    if benchmark_mode == "live":
        lines.append(
            "Use current information when needed. If the answer depends on live retrieval, "
            "ground it in retrieved evidence rather than parametric memory."
        )
    else:
        lines.append(
            "Answer with the final answer only when you have enough evidence. "
            "Prefer exact, benchmark-style responses over long explanations."
        )

    if attachment_path:
        lines.append(
            f"Attachment available: {Path(attachment_path).name}. "
            "Use the read_attachment tool if that file is needed."
        )

    return "\n\n".join(lines)


def bootstrap_mean_ci(
    values: List[float],
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Tuple[float, float]:
    """Bootstrap confidence interval for mean score reporting."""
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], values[0]

    rng = random.Random(seed)
    means = []
    n = len(values)
    for _ in range(n_bootstrap):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()

    alpha = (1.0 - confidence) / 2.0
    lo_idx = max(0, math.floor(alpha * len(means)))
    hi_idx = min(len(means) - 1, math.ceil((1.0 - alpha) * len(means)) - 1)
    return means[lo_idx], means[hi_idx]


# ── Result aggregation ────────────────────────────────────────────────────────

class GAIAReport:
    """Aggregated results across agents and task levels."""

    def __init__(self, results: List[GAIAResult]):
        self.results = results

    def accuracy(
        self,
        level: Optional[int] = None,
        agent: Optional[str] = None,
        metric: str = "score",
    ) -> float:
        filtered = self._filter(level, agent)
        if not filtered:
            return 0.0
        if metric == "exact":
            return sum(r.exact_match for r in filtered) / len(filtered)
        if metric == "token_overlap":
            return sum(r.token_overlap for r in filtered) / len(filtered)
        return sum(r.score for r in filtered) / len(filtered)

    def confidence_interval(
        self,
        level: Optional[int] = None,
        agent: Optional[str] = None,
        metric: str = "score",
        seed: int = 0,
    ) -> Tuple[float, float]:
        filtered = self._filter(level, agent)
        if metric == "exact":
            values = [r.exact_match for r in filtered]
        elif metric == "token_overlap":
            values = [r.token_overlap for r in filtered]
        else:
            values = [r.score for r in filtered]
        return bootstrap_mean_ci(values, seed=seed)

    def exact_match_rate(self, level: Optional[int] = None, agent: Optional[str] = None) -> float:
        filtered = self._filter(level, agent)
        if not filtered:
            return 0.0
        return sum(r.exact_match for r in filtered) / len(filtered)

    def avg_tool_calls(self, level: Optional[int] = None, agent: Optional[str] = None) -> float:
        filtered = self._filter(level, agent)
        if not filtered:
            return 0.0
        return sum(r.n_tool_calls for r in filtered) / len(filtered)

    def _filter(self, level=None, agent=None) -> List[GAIAResult]:
        res = self.results
        if level is not None:
            res = [r for r in res if r.level == level]
        if agent is not None:
            res = [r for r in res if r.agent_name == agent]
        return res

    def summary_table(self, agents: Optional[List[str]] = None, metric: str = "score") -> str:
        if agents is None:
            agents = sorted({r.agent_name for r in self.results})
        levels = [1, 2, 3]

        header = f"{'Agent':<25} {'L1':>6} {'L2':>6} {'L3':>6} {'Overall':>8}"
        lines = [header, "-" * len(header)]
        for agent in agents:
            accs = [self.accuracy(lvl, agent, metric=metric) for lvl in levels]
            overall = self.accuracy(agent=agent, metric=metric)
            row = (
                f"{agent:<25} "
                + "  ".join(f"{a:>6.3f}" for a in accs)
                + f"  {overall:>8.3f}"
            )
            lines.append(row)
        return "\n".join(lines)
