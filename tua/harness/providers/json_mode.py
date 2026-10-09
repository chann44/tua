"""JSON action mode: fallback for models without reliable native tool calling.

The tool list goes into the system prompt; the model replies with a single
fenced JSON block::

    {"tool": "press_keys", "args": {"keys": ["ok"]}}
    {"done": true, "summary": "YouTube is playing lofi beats"}

On a parse failure the harness re-prompts once with the error (handled by the
loop via `Turn.error`). Only usable on top of an OpenAI-compatible endpoint,
since local VLMs (Ollama / LM Studio / vLLM) are the target.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import replace

from tua.harness.messages import Message, ToolCall
from tua.harness.providers.base import Capabilities, Turn, sanitize_schema
from tua.harness.providers.openai_compat import OpenAICompatProvider

_FENCED = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)

JSON_RULES = """\

## How to reply (JSON action protocol)
You have NO native function calling. Every reply must contain exactly ONE action as a
fenced JSON block, and nothing else after it:

    {"tool": "<tool name>", "args": {<arguments>}}

or, when the task is fully done and verified:

    {"done": true, "summary": "<what you accomplished>"}

Rules:
- One action per reply. Wait for its result before deciding the next action.
- `args` must match the tool's schema. Omit optional arguments you don't need.
- Do not invent tool names or arguments.
"""


def _balanced_objects(text: str) -> list[str]:
    """All top-level balanced `{...}` substrings, in order."""
    objects, depth, start = [], 0, -1
    in_str, esc = False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start >= 0:
                objects.append(text[start : i + 1])
    return objects


def parse_action(text: str) -> dict | None:
    """Extract the action object: prefer a fenced block, else the last JSON object."""
    candidates = [m.group(1) for m in _FENCED.finditer(text)]
    candidates.extend(_balanced_objects(text))
    for cand in reversed(candidates):
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def build_system_prompt(base: str, tools) -> str:
    lines = [base, JSON_RULES, "## Available tools"]
    for t in tools:
        desc = " ".join((t.description or "").split())
        lines.append(f"- {t.name}: {desc}")
        lines.append(f"  args: {json.dumps(sanitize_schema(t.schema))}")
    return "\n".join(lines)


class JSONActionProvider:
    """Wraps an OpenAI-compatible provider; speaks the JSON action protocol."""

    def __init__(self, inner: OpenAICompatProvider):
        if not isinstance(inner, OpenAICompatProvider):
            raise SystemExit("--json-mode is only supported for OpenAI-compatible endpoints "
                             "(ollama, lmstudio, vllm, openai, openrouter, ...)")
        self.inner = inner
        self.name = getattr(inner, "name", "json")
        base_caps = getattr(inner, "capabilities", None)
        self.capabilities = replace(base_caps, native_tools=False) if base_caps else Capabilities(native_tools=False)

    async def complete(self, system: str, messages: list[Message], tools) -> Turn:
        prompt = build_system_prompt(system, tools)
        text, usage = await self.inner.raw_chat(prompt, messages)

        action = parse_action(text)
        if action is None:
            return Turn(text=text, usage=usage, error=(
                "Could not parse a JSON action from your reply. Answer with a single fenced "
                '```json block: {"tool": "...", "args": {...}} or {"done": true, "summary": "..."}.'
            ))

        if action.get("done"):
            return Turn(text=text, usage=usage, tool_calls=[
                ToolCall(id=f"done-{uuid.uuid4().hex[:6]}", name="done",
                         arguments={"summary": str(action.get("summary") or "")})
            ])

        tool, args = action.get("tool"), action.get("args") or {}
        known = {t.name for t in tools}
        if not isinstance(tool, str) or tool not in known:
            return Turn(text=text, usage=usage, error=(
                f"Unknown tool {tool!r}. Available tools: {', '.join(sorted(known))}."
            ))
        if not isinstance(args, dict):
            return Turn(text=text, usage=usage, error='"args" must be a JSON object of tool arguments.')

        call = ToolCall(id=f"json-{uuid.uuid4().hex[:8]}", name=tool, arguments=args)
        return Turn(text=text, tool_calls=[call], usage=usage)
