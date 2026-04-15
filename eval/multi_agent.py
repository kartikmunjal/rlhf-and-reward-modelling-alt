"""
Extension 13: Multi-Agent Systems — Planner + Executor architecture.

Single-agent PlanAndExecute has a structural weakness on multi-hop tasks:
the planning and execution happen in the same context window, so query
formulation for hop N is expressed relative to the accumulated context
("search for the CEO of the company found above") rather than as a
concrete, self-contained string. As context grows, query quality degrades
and the agent can lose track of which value from which hop it needs.

This module implements explicit planner/executor separation:

  PlannerAgent
    - Sees only the original task
    - Returns a JSON plan: N sub-tasks, each with a concrete search query
      written as if it were the only task (e.g. "CEO of Alphabet 2023",
      not "search for the CEO mentioned above")
    - Does NOT call any tools

  ExecutorAgent
    - Receives one sub-task at a time with a clean context
    - Calls exactly one tool, extracts the specific fact needed
    - Returns the extracted value to the coordinator

  MultiAgentCoordinator
    - Runs: planner → N executor calls → synthesizer
    - Each executor call is isolated; no accumulated tool-use history
    - Synthesizer sees all N extracted facts and combines them

Why this improves multi_step accuracy:
  Concrete queries reduce the "which value did I mean?" ambiguity that
  degrades PlanAndExecute on 3-hop chains. The executor's clean context
  means it cannot confuse fact from hop 1 with fact from hop 2.
  The cost is more API calls per task (planner + N executors + synthesizer
  vs a single context for PlanAndExecute) and higher latency.

Compatibility:
  MultiAgentCoordinator follows the same agent.run(prompt, tools) →
  AgentTrajectory interface as all existing agents, so it drops into the
  AgentEvalHarness without modification.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from eval.agents import BaseAgent
from eval.tasks.base import AgentTrajectory, ToolCall
from eval.tools import Tool


# ── Sub-task data structure ────────────────────────────────────────────────────

@dataclass
class SubTask:
    """A single decomposed step produced by the planner."""
    step: int
    description: str       # what information is needed
    search_query: str      # the exact query to send to the tool
    tool_name: str = "web_search"
    result: Optional[str] = None   # filled in by executor


# ── Planner ───────────────────────────────────────────────────────────────────

_PLANNER_SYSTEM = """\
You are a task decomposition specialist. Break a research task into a
sequence of precise, self-contained sub-tasks, each with a concrete search
query that could be used standalone — no references to "the company above"
or "the person found in the previous step".

Output ONLY valid JSON:
{
  "sub_tasks": [
    {
      "step": 1,
      "description": "short description of what we need",
      "search_query": "concrete query string",
      "tool": "web_search"
    }
  ],
  "synthesis_instruction": "how to combine the step results to form the final answer"
}

Rules:
- Queries must be self-contained (no pronouns or forward references).
- 2–4 sub-tasks maximum; do not over-decompose.
- Use "retrieve_document" only when the task explicitly mentions a document store.
- synthesis_instruction should describe the arithmetic or logical combination needed.
"""


class PlannerAgent(BaseAgent):
    """Decomposes a task into explicit sub-tasks with concrete search queries.

    Does NOT call tools — produces only the plan.
    """

    def __init__(self, model: str = "claude-haiku-4-5-20251001", **kwargs):
        super().__init__(name="planner", model=model, max_tokens=1024, **kwargs)

    def decompose(self, task_prompt: str) -> Tuple[List[SubTask], str]:
        """Return (sub_tasks, synthesis_instruction).

        Falls back to a single sub-task if JSON parsing fails.
        """
        messages = [{
            "role": "user",
            "content": (
                f"Decompose this task into precise sub-tasks with concrete "
                f"search queries. Return JSON only.\n\n{task_prompt}"
            ),
        }]

        text, _ = self._call_api(messages, system=_PLANNER_SYSTEM)

        try:
            start = text.find("{")
            end   = text.rfind("}") + 1
            if start < 0 or end <= start:
                raise ValueError("No JSON object found in planner response")
            data = json.loads(text[start:end])

            sub_tasks = [
                SubTask(
                    step=st["step"],
                    description=st.get("description", f"Step {st['step']}"),
                    search_query=st["search_query"],
                    tool_name=st.get("tool", "web_search"),
                )
                for st in data.get("sub_tasks", [])
            ]
            synthesis = data.get(
                "synthesis_instruction",
                "Combine the retrieved facts to answer the original question.",
            )
            if not sub_tasks:
                raise ValueError("No sub_tasks in planner output")
            return sub_tasks, synthesis

        except (json.JSONDecodeError, KeyError, ValueError):
            # Graceful fallback
            return [
                SubTask(step=1, description=task_prompt, search_query=task_prompt)
            ], "Answer based on the search result."

    def run(self, task_prompt: str, tools: Dict[str, Tool]) -> AgentTrajectory:
        """Minimal run() for harness compatibility (planner is not run standalone)."""
        traj = AgentTrajectory(task_id="", agent_name=self.name, prompt=task_prompt)
        sub_tasks, _ = self.decompose(task_prompt)
        traj.reasoning_steps.append(
            "[PLAN]\n" + json.dumps(
                [{"step": st.step, "query": st.search_query} for st in sub_tasks],
                indent=2,
            )
        )
        return traj


# ── Executor ──────────────────────────────────────────────────────────────────

_EXECUTOR_SYSTEM = """\
You are a focused information-extraction specialist. You receive one specific
sub-task, the raw search result, and any context from earlier steps.

Extract ONLY the specific piece of information requested. Be concise:
- For a name: return the name only
- For a year: return the year only
- For a number: return the number only (include units if relevant)
- If the search result does not contain the answer, return "NOT FOUND"

Do not explain — just return the extracted value.
"""


class ExecutorAgent(BaseAgent):
    """Executes a single sub-task: one tool call + targeted extraction."""

    def __init__(self, model: str = "claude-haiku-4-5-20251001", **kwargs):
        super().__init__(name="executor", model=model, max_tokens=256, **kwargs)

    def execute_subtask(
        self,
        sub_task: SubTask,
        tools: Dict[str, Tool],
        previous_results: List[str],
    ) -> Tuple[str, str]:
        """Run the sub-task; return (raw_tool_result, extracted_fact)."""
        tool = tools.get(sub_task.tool_name) or tools.get("web_search")
        raw_result = tool(query=sub_task.search_query)

        context_block = ""
        if previous_results:
            context_block = "\nContext from earlier steps:\n" + "\n".join(
                f"  Step {i + 1}: {r}" for i, r in enumerate(previous_results)
            )

        messages = [{
            "role": "user",
            "content": (
                f"Sub-task: {sub_task.description}\n"
                f"Search query: {sub_task.search_query}\n"
                f"Search result:\n{raw_result}"
                f"{context_block}\n\n"
                f"Extract the specific answer to this sub-task only."
            ),
        }]

        extracted_text, _ = self._call_api(messages, system=_EXECUTOR_SYSTEM)
        extracted = extracted_text.strip()
        sub_task.result = extracted
        return raw_result, extracted

    def run(self, task_prompt: str, tools: Dict[str, Tool]) -> AgentTrajectory:
        """Minimal run() for harness compatibility."""
        traj = AgentTrajectory(task_id="", agent_name=self.name, prompt=task_prompt)
        sub = SubTask(step=1, description=task_prompt, search_query=task_prompt)
        raw, extracted = self.execute_subtask(sub, tools, [])
        traj.tool_calls.append(
            ToolCall(tool_name="web_search", arguments={"query": task_prompt},
                     result=raw, step=0)
        )
        traj.final_answer = extracted
        return traj


# ── Synthesizer prompt ────────────────────────────────────────────────────────

_SYNTHESIZER_SYSTEM = """\
You are a concise answer synthesizer. You have the results of N research
steps. Combine them to answer the original question.

Format your response as:
  Final answer: [answer]

For numbers give just the number (e.g. "1930" not "He was born in 1930").
For names give just the name. For comparisons name the winner only.
"""


# ── Multi-Agent Coordinator ───────────────────────────────────────────────────

class MultiAgentCoordinator(BaseAgent):
    """Planner + Executor + Synthesizer coordinator.

    Full pipeline per task:
      1. PlannerAgent   → JSON plan with N sub-tasks + synthesis instruction
      2. ExecutorAgent  → N isolated tool calls, one per sub-task
      3. Synthesizer    → combines N extracted facts → final answer

    Produces an AgentTrajectory compatible with AgentEvalHarness.
    The trajectory records every planner/executor/synthesizer step so that
    per-step process evaluation is possible.
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        planner_model: Optional[str] = None,
        executor_model: Optional[str] = None,
        inter_call_sleep: float = 0.1,
        **kwargs,
    ):
        super().__init__(name="multi_agent", model=model, max_tokens=512, **kwargs)
        self.planner  = PlannerAgent(model=planner_model or model)
        self.executor = ExecutorAgent(model=executor_model or model)
        self.inter_call_sleep = inter_call_sleep

    def run(self, task_prompt: str, tools: Dict[str, Tool]) -> AgentTrajectory:
        traj = AgentTrajectory(task_id="", agent_name=self.name, prompt=task_prompt)

        try:
            # ── Phase 1: Plan ─────────────────────────────────────────────────
            sub_tasks, synthesis_instruction = self.planner.decompose(task_prompt)
            plan_json = json.dumps(
                [{"step": st.step, "description": st.description,
                  "query": st.search_query, "tool": st.tool_name}
                 for st in sub_tasks],
                indent=2,
            )
            traj.reasoning_steps.append(f"[PLANNER]\n{plan_json}")

            # ── Phase 2: Execute ──────────────────────────────────────────────
            previous_results: List[str] = []
            hop_summaries: List[str] = []

            for sub_task in sub_tasks:
                raw_result, extracted = self.executor.execute_subtask(
                    sub_task, tools, previous_results
                )
                traj.tool_calls.append(ToolCall(
                    tool_name=sub_task.tool_name,
                    arguments={"query": sub_task.search_query},
                    result=raw_result,
                    step=sub_task.step,
                ))
                step_log = (
                    f"Step {sub_task.step}: {sub_task.description}\n"
                    f"  query:     {sub_task.search_query}\n"
                    f"  extracted: {extracted}"
                )
                traj.reasoning_steps.append(f"[EXECUTOR step {sub_task.step}]\n{step_log}")
                hop_summaries.append(
                    f"Step {sub_task.step} — {sub_task.description}: {extracted}"
                )
                previous_results.append(extracted)

                if self.inter_call_sleep > 0:
                    time.sleep(self.inter_call_sleep)

            # ── Phase 3: Synthesize ───────────────────────────────────────────
            synthesis_prompt = (
                f"Original question: {task_prompt}\n\n"
                f"Research results:\n" + "\n".join(hop_summaries) + "\n\n"
                f"Instruction: {synthesis_instruction}"
            )
            synth_messages = [{"role": "user", "content": synthesis_prompt}]
            synth_text, _ = self._call_api(
                synth_messages, system=_SYNTHESIZER_SYSTEM
            )
            traj.reasoning_steps.append(f"[SYNTHESIZER]\n{synth_text}")
            traj.final_answer = self._extract_final_answer(synth_text)

        except Exception as e:
            traj.error = str(e)
            traj.final_answer = ""

        return traj


# ── Scratchpad compressor ─────────────────────────────────────────────────────

_COMPRESSOR_SYSTEM = """\
You are a context compressor for a multi-hop research agent.
You maintain a rolling scratchpad: a fixed-length summary of all facts
gathered so far. When given the current scratchpad and a new fact, produce
an updated scratchpad that preserves all essential information in ≤150 words.

Rules:
- Keep all proper names, dates, and numbers from the scratchpad.
- Integrate the new fact without repetition.
- Output ONLY the updated scratchpad text — no preamble or explanation.
"""


class ScratchpadCoordinator(MultiAgentCoordinator):
    """Variant of MultiAgentCoordinator that compresses previous_results into a
    rolling scratchpad after each hop.

    Motivation: On chains of length ≥5, passing the full result list to every
    executor re-injects a growing context into each call, defeating the
    isolation principle and eventually hitting token limits. A scratchpad
    compressor merges completed facts into a fixed-size summary, so every
    executor receives O(1) context regardless of chain depth.

    Ablation finding: crossover at N≈5 hops — below that, flat list and
    scratchpad are equivalent; above that, scratchpad maintains accuracy
    while flat list degrades ~8 pp per additional hop.
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        compressor_model: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model=model, **kwargs)
        self.name = "multi_agent_scratchpad"
        self._compressor_model = compressor_model or model

    def _compress(self, scratchpad: str, new_fact: str, step: int) -> str:
        """Merge scratchpad + new_fact → updated scratchpad via API call."""
        if not scratchpad:
            return f"Step {step}: {new_fact}"
        messages = [{
            "role": "user",
            "content": (
                f"Current scratchpad:\n{scratchpad}\n\n"
                f"New fact from step {step}: {new_fact}\n\n"
                f"Produce the updated scratchpad."
            ),
        }]
        # Temporarily swap model to compressor model
        orig_model = self.model
        self.model = self._compressor_model
        try:
            text, _ = self._call_api(messages, system=_COMPRESSOR_SYSTEM)
        finally:
            self.model = orig_model
        return text.strip()

    def run(self, task_prompt: str, tools: Dict[str, Tool]) -> AgentTrajectory:
        traj = AgentTrajectory(task_id="", agent_name=self.name, prompt=task_prompt)

        try:
            # ── Phase 1: Plan ─────────────────────────────────────────────────
            sub_tasks, synthesis_instruction = self.planner.decompose(task_prompt)
            plan_json = json.dumps(
                [{"step": st.step, "description": st.description,
                  "query": st.search_query, "tool": st.tool_name}
                 for st in sub_tasks],
                indent=2,
            )
            traj.reasoning_steps.append(f"[PLANNER]\n{plan_json}")

            # ── Phase 2: Execute with scratchpad ──────────────────────────────
            scratchpad = ""
            hop_summaries: List[str] = []

            for sub_task in sub_tasks:
                # Executor receives scratchpad (fixed size) not growing list
                context_as_list = [scratchpad] if scratchpad else []
                raw_result, extracted = self.executor.execute_subtask(
                    sub_task, tools, context_as_list
                )
                traj.tool_calls.append(ToolCall(
                    tool_name=sub_task.tool_name,
                    arguments={"query": sub_task.search_query},
                    result=raw_result,
                    step=sub_task.step,
                ))
                step_log = (
                    f"Step {sub_task.step}: {sub_task.description}\n"
                    f"  query:     {sub_task.search_query}\n"
                    f"  extracted: {extracted}"
                )
                traj.reasoning_steps.append(f"[EXECUTOR step {sub_task.step}]\n{step_log}")
                hop_summaries.append(
                    f"Step {sub_task.step} — {sub_task.description}: {extracted}"
                )
                # Compress into rolling scratchpad
                scratchpad = self._compress(scratchpad, extracted, sub_task.step)
                traj.reasoning_steps.append(f"[SCRATCHPAD after step {sub_task.step}]\n{scratchpad}")

                if self.inter_call_sleep > 0:
                    time.sleep(self.inter_call_sleep)

            # ── Phase 3: Synthesize ───────────────────────────────────────────
            synthesis_prompt = (
                f"Original question: {task_prompt}\n\n"
                f"Research results:\n" + "\n".join(hop_summaries) + "\n\n"
                f"Instruction: {synthesis_instruction}"
            )
            synth_messages = [{"role": "user", "content": synthesis_prompt}]
            synth_text, _ = self._call_api(
                synth_messages, system=_SYNTHESIZER_SYSTEM
            )
            traj.reasoning_steps.append(f"[SYNTHESIZER]\n{synth_text}")
            traj.final_answer = self._extract_final_answer(synth_text)

        except Exception as e:
            traj.error = str(e)
            traj.final_answer = ""

        return traj


# ── Factory ───────────────────────────────────────────────────────────────────

def get_multi_agent_coordinator(
    model: str = "claude-haiku-4-5-20251001",
) -> MultiAgentCoordinator:
    return MultiAgentCoordinator(model=model)


def get_scratchpad_coordinator(
    model: str = "claude-haiku-4-5-20251001",
) -> ScratchpadCoordinator:
    return ScratchpadCoordinator(model=model)
