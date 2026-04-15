"""
Tool implementations for AgentBench-Mini.

Design: pluggable interface so you can swap mock tools (for deterministic
offline testing) with real tools (live web search via Serper/Bing API).

All tools follow the signature:
    tool(query: str, **kwargs) -> str

The agent receives a tool as a callable and a name/description dict.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, Optional


# ── Tool descriptor ────────────────────────────────────────────────────────────

class Tool:
    """A named, callable tool with a description for the agent's system prompt."""

    def __init__(self, name: str, description: str, fn):
        self.name = name
        self.description = description
        self._fn = fn

    def __call__(self, **kwargs) -> str:
        return self._fn(**kwargs)

    def to_anthropic_tool_spec(self) -> Dict:
        """Return the tool spec dict for the Anthropic messages API."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query or document lookup key",
                    }
                },
                "required": ["query"],
            },
        }


# ── Mock search tool ──────────────────────────────────────────────────────────
#
# Canned responses keyed on normalised query substrings.
# This makes the benchmark deterministic and runnable without an API key.
# Swap out _mock_search with _live_search to use real web search.

_MOCK_SEARCH_DB: Dict[str, str] = {
    # Tool use tasks
    "unemployment rate us march 2022": (
        "US Unemployment Rate - March 2022\n"
        "The U.S. Bureau of Labor Statistics reported that the unemployment rate "
        "fell to 3.6 percent in March 2022, down from 3.8 percent in February."
    ),
    "gdp growth usa 2023": (
        "US GDP growth rate 2023: The U.S. economy grew at an annual rate of 2.5% "
        "in 2023, according to the Bureau of Economic Analysis."
    ),
    "population india 2023": (
        "India's population in 2023 surpassed 1.43 billion, making it the most "
        "populous country in the world, overtaking China."
    ),
    "capital france": "The capital of France is Paris.",
    "boiling point water celsius": (
        "Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at "
        "standard atmospheric pressure (1 atm)."
    ),
    "height mount everest meters": (
        "Mount Everest is 8,848.86 meters (29,031.7 feet) tall, as measured "
        "by a 2020 survey by China and Nepal."
    ),
    "ceo apple 2023": (
        "Tim Cook is the CEO of Apple Inc. He has served in this role since "
        "August 2011, succeeding Steve Jobs."
    ),
    "founded year google": (
        "Google was founded on September 4, 1998, by Larry Page and Sergey Brin "
        "while they were PhD students at Stanford University."
    ),
    "distance earth moon km": (
        "The average distance from Earth to the Moon is approximately 384,400 "
        "kilometers (238,855 miles)."
    ),
    "speed light ms": (
        "The speed of light in a vacuum is exactly 299,792,458 meters per second "
        "(approximately 3 × 10^8 m/s)."
    ),
    "inflation rate us 2022": (
        "US inflation rate 2022: The Consumer Price Index rose 8.0% for the "
        "12 months ending March 2022, the highest since 1981."
    ),
    # Multi-step tasks
    "who acquired deepmind": (
        "DeepMind was acquired by Google (now Alphabet Inc.) in 2014 for "
        "approximately $500 million. DeepMind became a subsidiary of Alphabet."
    ),
    "ceo alphabet google": (
        "Sundar Pichai is the CEO of both Google LLC and its parent company "
        "Alphabet Inc. He became Google CEO in 2015 and Alphabet CEO in 2019."
    ),
    "sundar pichai net worth": (
        "Sundar Pichai's net worth is estimated at approximately $1.3 billion "
        "as of 2023, based on his compensation packages and stock holdings."
    ),
    "largest company by revenue 2023": (
        "Walmart was the world's largest company by revenue in 2023, with annual "
        "revenues of approximately $611 billion."
    ),
    "founder walmart": (
        "Walmart was founded by Sam Walton in 1962. He opened the first Walmart "
        "store in Rogers, Arkansas."
    ),
    "sam walton net worth": (
        "Sam Walton died in 1992. His estate was worth approximately $8.6 billion "
        "at the time of his death. The Walton family's combined net worth is now "
        "estimated at over $200 billion."
    ),
    "country highest gdp per capita 2023": (
        "Luxembourg has the highest GDP per capita in the world at approximately "
        "$135,000 USD (2023 estimate). Monaco and Liechtenstein are close behind."
    ),
    "population luxembourg": (
        "Luxembourg has a population of approximately 660,000 people (2023 estimate), "
        "making it one of the smallest countries in Europe."
    ),
    "first country to land moon": (
        "The United States was the first country to land humans on the Moon. "
        "Apollo 11 landed on July 20, 1969, with astronauts Neil Armstrong and "
        "Buzz Aldrin."
    ),
    "neil armstrong biography": (
        "Neil Armstrong (1930–2012) was an American astronaut and the first human "
        "to walk on the Moon. He was born in Wapakoneta, Ohio."
    ),
    "year first email sent": (
        "The first email was sent in 1971 by Ray Tomlinson, who also introduced "
        "the @ symbol for email addresses."
    ),
    "ray tomlinson inventor": (
        "Ray Tomlinson (1941–2016) was an American computer programmer who "
        "invented email in 1971. He worked at BBN Technologies."
    ),
    # Context-window ablation chain (hop 1–8, each answer feeds the next query)
    # Chain: Alphabet CEO → birth year → US president that year → year resigned
    #        → successor → successor's birth state → state capital → Lincoln assassination year
    "ceo alphabet 2023": (
        "Sundar Pichai is the CEO of Alphabet Inc. as of 2023. He was born in 1972."
    ),
    "sundar pichai birth year": (
        "Sundar Pichai was born on June 10, 1972, in Madurai, Tamil Nadu, India."
    ),
    "us president 1972": (
        "Richard Nixon was the President of the United States in 1972. He won "
        "re-election in November 1972 by a landslide over George McGovern."
    ),
    "year richard nixon resigned": (
        "Richard Nixon resigned from the presidency on August 9, 1974, becoming "
        "the only US president to resign from office."
    ),
    "us president succeeded nixon 1974": (
        "Gerald Ford became the 38th President of the United States on August 9, 1974, "
        "after Richard Nixon's resignation. Ford had been Vice President since December 1973."
    ),
    "gerald ford birth state": (
        "Gerald Ford was born on July 14, 1913, in Omaha, Nebraska. Nebraska is his birth state."
    ),
    "capital of nebraska": (
        "The capital of Nebraska is Lincoln, named after President Abraham Lincoln. "
        "Lincoln has been the state capital since 1869."
    ),
    "year abraham lincoln assassinated": (
        "Abraham Lincoln was assassinated on April 14, 1865, at Ford's Theatre in Washington, D.C. "
        "He died the following morning, April 15, 1865."
    ),
    # Failure recovery — these should return no useful results
    "xylofrobnic organization 2024": "",
    "blarpian federation history": "",
    "zorblax corporation ceo": "",
    "quimbleton university ranking": "",
    "flimbleworth award winners": "",
    "glarbian economic summit 2023": "",
    "phortrix algorithm inventor": "",
    "scrumblethwaite research institute": "",
    "vronkish cultural traditions": "",
    "tromblesian philosophy": "",
}


def _normalise_query(q: str) -> str:
    """Lowercase and strip for lookup."""
    return re.sub(r"\s+", " ", q.lower().strip().rstrip("?.,"))


def _mock_search(query: str) -> str:
    """Return canned search results for deterministic benchmarking."""
    key = _normalise_query(query)
    # Exact match first
    if key in _MOCK_SEARCH_DB:
        result = _MOCK_SEARCH_DB[key]
        return result if result else f"No results found for: {query}"
    # Partial match
    for db_key, value in _MOCK_SEARCH_DB.items():
        if db_key in key or key in db_key:
            return value if value else f"No results found for: {query}"
    # Check for fictional entity keywords
    fictional_markers = ["xylof", "blarp", "zorbl", "quimbl", "flimbl", "glarb",
                         "phortrix", "scrumble", "vronk", "trombles"]
    if any(m in key for m in fictional_markers):
        return ""
    return f"Search results for '{query}': No highly relevant results found."


def _live_search(query: str, api_key: Optional[str] = None) -> str:
    """Live web search via Serper API (requires SERPER_API_KEY env var).

    Falls back to mock search if no API key is available.
    """
    key = api_key or os.environ.get("SERPER_API_KEY")
    if not key:
        return _mock_search(query)

    try:
        import requests
        response = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": key, "Content-Type": "application/json"},
            json={"q": query, "num": 3},
            timeout=10,
        )
        data = response.json()
        results = []
        if "answerBox" in data:
            results.append(data["answerBox"].get("answer", ""))
        for item in data.get("organic", [])[:3]:
            snippet = item.get("snippet", "")
            if snippet:
                results.append(f"{item['title']}: {snippet}")
        return "\n".join(results) if results else f"No results for: {query}"
    except Exception as e:
        return f"Search error: {e}"


# ── Document retrieval tool ───────────────────────────────────────────────────

_MOCK_DOCS: Dict[str, str] = {
    "ml_glossary": (
        "Machine Learning Glossary\n"
        "Supervised learning: training on labeled (input, output) pairs.\n"
        "Unsupervised learning: discovering patterns without labels.\n"
        "Reinforcement learning: learning from rewards and penalties.\n"
        "Gradient descent: iterative optimisation by following the negative gradient.\n"
        "Overfitting: model memorises training data, fails to generalise.\n"
        "Regularisation: technique to reduce overfitting (L1, L2, dropout).\n"
    ),
    "rlhf_paper_abstract": (
        "Abstract: We present a method for training language models to follow "
        "instructions using human feedback. Starting from an initial language "
        "model, we use supervised fine-tuning followed by reinforcement learning "
        "from human feedback (RLHF) to align the model's outputs with human "
        "preferences. Our InstructGPT models are preferred over GPT-3 outputs "
        "despite having 100x fewer parameters."
    ),
}


def _mock_retrieve(query: str) -> str:
    key = _normalise_query(query)
    for doc_key, content in _MOCK_DOCS.items():
        if doc_key in key or any(w in key for w in doc_key.split("_")):
            return content
    return f"Document not found: {query}"


# ── Exported tool instances ───────────────────────────────────────────────────

def make_search_tool(use_live: bool = False) -> Tool:
    """Create a web search tool (mock or live)."""
    fn = _live_search if use_live and os.environ.get("SERPER_API_KEY") else _mock_search
    return Tool(
        name="web_search",
        description=(
            "Search the web for factual information. "
            "Returns a short excerpt from relevant results. "
            "Use this when you need a specific fact, number, or recent event."
        ),
        fn=lambda query: fn(query),
    )


def make_retrieve_tool() -> Tool:
    """Create a document retrieval tool."""
    return Tool(
        name="retrieve_document",
        description=(
            "Retrieve a document by name or topic. "
            "Use for technical references, paper abstracts, or glossaries."
        ),
        fn=lambda query: _mock_retrieve(query),
    )


def get_default_tools(use_live: bool = False) -> Dict[str, Tool]:
    """Return the default tool set for benchmark runs."""
    return {
        "web_search": make_search_tool(use_live),
        "retrieve_document": make_retrieve_tool(),
    }
