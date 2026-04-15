"""
Extension 14: CodeExecutorAgent.

A ReAct-style agent specialized for code debugging tasks. Given a broken
Python function and a test suite, it uses python_exec in a loop to:
  1. Diagnose the bug
  2. Write a fix and test it in the sandbox
  3. Iterate until tests pass or max_iterations is reached
  4. Return the corrected function code

Final answer extraction: looks for "FIXED CODE:" marker, falls back to
the last ```python ... ``` block in the response.

Follows the same agent.run(task_prompt, tools) → AgentTrajectory interface
as all other agents, so it drops into AgentEvalHarness and run_code_benchmark.py
without modification.
"""

from __future__ import annotations

import re
from typing import Dict, List

from eval.agents import BaseAgent
from eval.tasks.base import AgentTrajectory, ToolCall
from eval.tools import Tool


_CODE_SYSTEM = """\
You are a Python debugging specialist. You will be given a broken Python function
and a set of test assertions that must all pass.

Your process:
1. Read the broken code and identify the bug
2. Write a fix
3. Test your fix using python_exec — include your implementation AND the test
   assertions in the code you execute, like:
   ```
   def fixed_function(...):
       ...
   assert fixed_function(x) == expected
   ```
4. If tests fail, revise and try again
5. Once all tests pass, output the corrected function with the marker:

FIXED CODE:
```python
[your corrected function here]
```

Rules:
- Return ONLY the function definition (no test code in the final answer)
- Include imports inside the function only if the function itself requires them
- Be concise — identify the bug, fix it, verify it
"""


class CodeExecutorAgent(BaseAgent):
    """ReAct-style agent that debugs Python code using a sandboxed executor.

    Uses the python_exec tool to test candidate fixes iteratively. Returns
    the corrected function code as final_answer.
    """

    def __init__(self, model: str = "claude-haiku-4-5-20251001", **kwargs):
        super().__init__(
            name="code_executor",
            model=model,
            max_tokens=1024,
            max_tool_calls=6,
            **kwargs,
        )

    def run(self, task_prompt: str, tools: Dict[str, Tool]) -> AgentTrajectory:
        traj = AgentTrajectory(task_id="", agent_name=self.name, prompt=task_prompt)

        python_exec = tools.get("python_exec")
        if python_exec is None:
            traj.error = "python_exec tool not found"
            traj.final_answer = ""
            return traj

        tool_spec = python_exec.to_anthropic_tool_spec()
        messages: List[Dict] = [{"role": "user", "content": task_prompt}]

        for step in range(self.max_tool_calls + 1):
            text, tool_uses = self._call_api(
                messages, system=_CODE_SYSTEM, tool_specs=[tool_spec]
            )

            if text:
                traj.reasoning_steps.append(text)

            if tool_uses:
                # Execute the python_exec call
                tool_use = tool_uses[0]
                code = tool_use["input"].get("code", "")
                result = python_exec(code=code)

                traj.tool_calls.append(ToolCall(
                    tool_name="python_exec",
                    arguments={"code": code},
                    result=result,
                    step=step,
                ))

                # Feed result back as tool result message
                messages.append({
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": text} if text else None,
                        {
                            "type": "tool_use",
                            "id": tool_use["id"],
                            "name": "python_exec",
                            "input": tool_use["input"],
                        },
                    ],
                })
                # Filter out None blocks
                messages[-1]["content"] = [
                    b for b in messages[-1]["content"] if b is not None
                ]
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use["id"],
                            "content": result,
                        }
                    ],
                })
            else:
                # No tool call — agent is done
                traj.final_answer = self._extract_code_answer(text)
                break
        else:
            # Ran out of iterations — extract whatever the last text had
            traj.final_answer = self._extract_code_answer(
                traj.reasoning_steps[-1] if traj.reasoning_steps else ""
            )

        return traj

    @staticmethod
    def _extract_code_answer(text: str) -> str:
        """Extract the fixed function from agent response.

        Priority:
        1. Code block immediately after "FIXED CODE:" marker
        2. Last ```python ... ``` block in response
        3. Full text (fallback)
        """
        if not text:
            return ""

        # Look for FIXED CODE: marker followed by a code block
        marker_match = re.search(
            r"FIXED CODE:\s*```python\s*\n(.*?)```",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if marker_match:
            return marker_match.group(1).strip()

        # Fall back to last ```python block
        blocks = re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)
        if blocks:
            return blocks[-1].strip()

        # Last resort: return full text
        return text.strip()
