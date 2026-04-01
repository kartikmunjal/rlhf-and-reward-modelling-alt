"""
Agent implementations for AgentBench-Mini.

Three configurations that can be run on the same task set:

  1. ZeroShotAgent      — Claude with no tools; answers from parametric memory only.
                          Baseline that shows what model knowledge alone achieves.

  2. ReActAgent         — Claude with tools and ReAct prompting (Yao et al. 2022):
                          Thought → Action → Observation → Thought → …
                          Each step is explicit in the reasoning chain.

  3. PlanAndExecuteAgent — Claude first writes a step-by-step plan for which tools
                           to call and in what order, then executes that plan.
                           Improves on ReAct on multi-step tasks by reducing
                           mid-trajectory drift.

All agents follow the same interface:
    agent.run(prompt: str, tools: Dict[str, Tool]) -> AgentTrajectory

Requires ANTHROPIC_API_KEY environment variable.
"""

from __future__ import annotations

import os
import re
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from eval.tasks.base import AgentTrajectory, ToolCall
from eval.tools import Tool


# ── Base agent ─────────────────────────────────────────────────────────────────

class BaseAgent(ABC):
    """Abstract agent that can run a task and return a full trajectory."""

    def __init__(
        self,
        name: str,
        model: str = "claude-haiku-4-5-20251001",
        max_tokens: int = 1024,
        max_tool_calls: int = 5,
    ):
        self.name = name
        self.model = model
        self.max_tokens = max_tokens
        self.max_tool_calls = max_tool_calls
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    @abstractmethod
    def run(self, task_prompt: str, tools: Dict[str, Tool]) -> AgentTrajectory:
        ...

    def _call_api(
        self,
        messages: List[Dict],
        system: str = "",
        tool_specs: Optional[List[Dict]] = None,
    ) -> Tuple[str, Optional[List[Dict]]]:
        """Call the Anthropic API and return (text_response, tool_use_blocks)."""
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if tool_specs:
            kwargs["tools"] = tool_specs

        for attempt in range(3):
            try:
                response = self.client.messages.create(**kwargs)
                break
            except Exception as e:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)

        text = ""
        tool_uses = []
        for block in response.content:
            if block.type == "text":
                text = block.text
            elif block.type == "tool_use":
                tool_uses.append({
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })

        return text, tool_uses if tool_uses else None

    def _extract_final_answer(self, text: str) -> str:
        """Extract a clean final answer from the agent's text response."""
        # Look for explicit "Answer:" or "Final answer:" markers
        for marker in ["Final answer:", "Answer:", "ANSWER:", "Result:", "RESULT:"]:
            if marker.lower() in text.lower():
                idx = text.lower().find(marker.lower())
                return text[idx + len(marker):].strip().split("\n")[0].strip()
        # Fall back to last non-empty line
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        return lines[-1] if lines else text.strip()


# ── Zero-shot agent ────────────────────────────────────────────────────────────

class ZeroShotAgent(BaseAgent):
    """Claude with no tools — answers from parametric memory only.

    Baseline: shows what the model knows without any retrieval augmentation.
    Expected to fail on recent factual tasks but succeed on stable knowledge.
    """

    def __init__(self, model: str = "claude-haiku-4-5-20251001", **kwargs):
        super().__init__(name="zero_shot", model=model, **kwargs)

    def run(self, task_prompt: str, tools: Dict[str, Tool]) -> AgentTrajectory:
        traj = AgentTrajectory(
            task_id="",
            agent_name=self.name,
            prompt=task_prompt,
        )

        system = (
            "You are a helpful AI assistant. Answer questions directly and concisely. "
            "If you are not confident in your answer, say so clearly."
        )
        messages = [{"role": "user", "content": task_prompt}]

        try:
            text, _ = self._call_api(messages, system=system)
            traj.final_answer = self._extract_final_answer(text)
            traj.reasoning_steps = [text]
        except Exception as e:
            traj.error = str(e)
            traj.final_answer = ""

        return traj


# ── ReAct agent ────────────────────────────────────────────────────────────────

_REACT_SYSTEM = """You are a helpful AI assistant that solves tasks step by step.

Follow this format for each step:
  Thought: [what you know and what you need to find out]
  Action: [tool_name with query]
  Observation: [tool result]

Continue until you have enough information, then write:
  Final answer: [your answer]

Be concise. Call tools only when necessary — do not search for things you already know."""


class ReActAgent(BaseAgent):
    """Claude with ReAct prompting (Yao et al. 2022).

    Interleaves Thought → Action → Observation steps until the agent
    produces a final answer.  The full chain-of-thought is captured in
    the trajectory for process-level evaluation.
    """

    def __init__(self, model: str = "claude-haiku-4-5-20251001", **kwargs):
        super().__init__(name="react", model=model, **kwargs)

    def run(self, task_prompt: str, tools: Dict[str, Tool]) -> AgentTrajectory:
        traj = AgentTrajectory(
            task_id="",
            agent_name=self.name,
            prompt=task_prompt,
        )

        tool_specs = [t.to_anthropic_tool_spec() for t in tools.values()]
        messages = [{"role": "user", "content": task_prompt}]
        step = 0

        try:
            while step < self.max_tool_calls + 1:
                text, tool_uses = self._call_api(
                    messages,
                    system=_REACT_SYSTEM,
                    tool_specs=tool_specs,
                )
                traj.reasoning_steps.append(text)

                if not tool_uses:
                    # Agent chose not to call a tool → final answer
                    traj.final_answer = self._extract_final_answer(text)
                    break

                # Process each tool call
                tool_results = []
                for tu in tool_uses:
                    tool_name = tu["name"]
                    query = tu["input"].get("query", "")
                    if tool_name in tools:
                        result = tools[tool_name](query=query)
                    else:
                        result = f"Unknown tool: {tool_name}"

                    tc = ToolCall(
                        tool_name=tool_name,
                        arguments=tu["input"],
                        result=result,
                        step=step,
                    )
                    traj.tool_calls.append(tc)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tu["id"],
                        "content": result if result else "No results found.",
                    })

                # Add assistant turn and tool results to messages
                messages.append({"role": "assistant", "content": [
                    {"type": "text", "text": text} if text else None,
                    *[{"type": "tool_use", "id": tu["id"], "name": tu["name"],
                       "input": tu["input"]} for tu in tool_uses],
                ]})
                messages[-1]["content"] = [b for b in messages[-1]["content"] if b]
                messages.append({"role": "user", "content": tool_results})
                step += 1

            else:
                # Hit max_tool_calls — extract best answer from last response
                traj.final_answer = self._extract_final_answer(
                    traj.reasoning_steps[-1] if traj.reasoning_steps else ""
                )

        except Exception as e:
            traj.error = str(e)
            traj.final_answer = ""

        return traj


# ── Plan-and-Execute agent ─────────────────────────────────────────────────────

_PLAN_SYSTEM = """You are a careful AI assistant. When given a task that requires
information retrieval, first write a numbered plan of exactly which tool calls
you will make and why, then execute that plan step by step.

Format:
  PLAN:
  1. [tool_name]: [what to search for and why]
  2. [tool_name]: [what to search for and why]
  ...

  EXECUTION:
  (proceed to use the tools as planned)

  Final answer: [your answer]

Planning first reduces errors on multi-step tasks because you commit to a
strategy before executing it."""


class PlanAndExecuteAgent(BaseAgent):
    """Claude with an explicit planning step before tool use.

    Phase 1: The agent writes out a numbered plan of tool calls.
    Phase 2: The agent executes the plan step by step.

    Expected improvement over ReAct on multi-step tasks: the plan prevents
    mid-trajectory drift where the agent forgets earlier results.
    """

    def __init__(self, model: str = "claude-haiku-4-5-20251001", **kwargs):
        super().__init__(name="plan_and_execute", model=model, **kwargs)

    def run(self, task_prompt: str, tools: Dict[str, Tool]) -> AgentTrajectory:
        traj = AgentTrajectory(
            task_id="",
            agent_name=self.name,
            prompt=task_prompt,
        )

        tool_specs = [t.to_anthropic_tool_spec() for t in tools.values()]

        # Phase 1: Generate plan
        plan_messages = [{"role": "user", "content": (
            f"{task_prompt}\n\n"
            "First, write a numbered plan of the tool calls you will make to "
            "answer this question. Then I will ask you to execute it."
        )}]

        try:
            plan_text, _ = self._call_api(plan_messages, system=_PLAN_SYSTEM)
            traj.reasoning_steps.append(f"[PLAN]\n{plan_text}")
        except Exception as e:
            traj.error = str(e)
            return traj

        # Phase 2: Execute with tools
        exec_messages = [
            {"role": "user", "content": task_prompt},
            {"role": "assistant", "content": plan_text},
            {"role": "user", "content": "Now execute your plan using the available tools."},
        ]

        step = 0
        try:
            while step < self.max_tool_calls + 1:
                text, tool_uses = self._call_api(
                    exec_messages,
                    system=_PLAN_SYSTEM,
                    tool_specs=tool_specs,
                )
                traj.reasoning_steps.append(f"[EXEC step {step}]\n{text}")

                if not tool_uses:
                    traj.final_answer = self._extract_final_answer(text)
                    break

                tool_results = []
                for tu in tool_uses:
                    tool_name = tu["name"]
                    query = tu["input"].get("query", "")
                    result = tools[tool_name](query=query) if tool_name in tools else f"Unknown tool: {tool_name}"

                    traj.tool_calls.append(ToolCall(
                        tool_name=tool_name,
                        arguments=tu["input"],
                        result=result,
                        step=step,
                    ))
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tu["id"],
                        "content": result if result else "No results found.",
                    })

                exec_messages.append({"role": "assistant", "content": [
                    *([{"type": "text", "text": text}] if text else []),
                    *[{"type": "tool_use", "id": tu["id"], "name": tu["name"],
                       "input": tu["input"]} for tu in tool_uses],
                ]})
                exec_messages.append({"role": "user", "content": tool_results})
                step += 1

            else:
                traj.final_answer = self._extract_final_answer(
                    traj.reasoning_steps[-1]
                )

        except Exception as e:
            traj.error = str(e)

        return traj


# ── Factory ────────────────────────────────────────────────────────────────────

def get_all_agents(model: str = "claude-haiku-4-5-20251001") -> List[BaseAgent]:
    """Return all three agent configurations for a full benchmark run."""
    return [
        ZeroShotAgent(model=model),
        ReActAgent(model=model),
        PlanAndExecuteAgent(model=model),
    ]
