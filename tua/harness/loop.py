"""The observe -> act -> verify loop.

Stops on `done(summary)`, a turn with no tool calls, `--max-steps`, or Ctrl-C.
Tool calls from one model turn run strictly in order (the TV is one device).
Destructive tools (per `destructive_hint`, plus `power off|toggle`) need a y/n
confirmation unless the CLI passed `--yes`.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Callable

import anyio

from tua.harness.context import prune
from tua.harness.mcp_client import MCPClient
from tua.harness.messages import AssistantMessage, Message, ToolResult, Usage, UserMessage
from tua.harness.providers.base import Provider, ToolSpec
from tua.harness.trace import Trace

DONE = ToolSpec(
    name="done",
    description=(
        "Call when the task is complete and verified, or when you are confident it cannot be done. "
        "This ends the run."
    ),
    schema={
        "type": "object",
        "properties": {"summary": {"type": "string", "description": "What was accomplished, or why it failed."}},
        "required": ["summary"],
    },
)

HARNESS_RULES = """\

## Harness rules
- You are fully autonomous: decide, act, and verify without asking a human.
- After every meaningful action, VERIFY the result (screenshot / get_ui / now_playing) before continuing; if it failed, retry a different way.
- Prefer the fewest, most reliable actions: deep links (play_youtube, open_url, netflix, launch_app, search_tv) and select_element beat many D-pad presses.
- Older screenshots are pruned from context — rely on your own earlier notes, not on old images.
- The run only ends when you call `done(summary)`. If the task proves impossible, call done with a summary starting with "FAILED:".
"""


@dataclass
class RunConfig:
    task: str
    model: str = ""
    max_steps: int = 40
    keep_images: int = 3
    max_text_chars: int = 6000
    auto_approve: bool = False  # --yes: skip destructive-tool confirmations


@dataclass
class RunResult:
    status: str = "error"  # done | no_tool_calls | max_steps | parse_error | interrupted | error
    summary: str = ""
    steps: int = 0
    usage: Usage = field(default_factory=Usage)
    run_dir: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "done" and not self.summary.lstrip().upper().startswith("FAILED")


def _is_risky(tool: ToolSpec, args: dict) -> bool:
    if tool.destructive:
        return True
    # `power` isn't flagged destructive by the server, but off/toggle kills ADB-visibility.
    return tool.name == "power" and args.get("action") in ("off", "toggle")


async def _confirm(tool_name: str, args: dict) -> bool:
    prompt = f"  run destructive tool {tool_name} {json.dumps(args)}? [y/N] "

    def ask() -> str:
        try:
            return input(prompt)
        except EOFError:  # non-interactive stdin: treat as "no"
            print("n (no tty — pass --yes to auto-approve)")
            return "n"

    answer = await anyio.to_thread.run_sync(ask)
    return answer.strip().lower() in ("y", "yes")


async def run_task(
    provider: Provider,
    client: MCPClient,
    cfg: RunConfig,
    trace: Trace,
    echo: Callable[[str], None] = lambda line: None,
) -> RunResult:
    server_tools = await client.list_tools()
    tools = server_tools + [DONE]
    system = "\n\n".join(s for s in (client.instructions.strip(), HARNESS_RULES.strip()) if s)
    messages: list[Message] = [UserMessage.text(cfg.task)]

    result = RunResult()
    usage_total = Usage()
    parse_failures = 0
    trace.log({"event": "run_start", "task": cfg.task, "model": cfg.model,
               "max_steps": cfg.max_steps, "tools": [t.name for t in tools]})
    echo(f"◆ {len(server_tools)} tools from server · model {cfg.model or provider.name}")

    try:
        for step in range(1, cfg.max_steps + 1):
            result.steps = step
            t0 = time.monotonic()
            turn = await provider.complete(system, prune(messages, cfg.keep_images, cfg.max_text_chars), tools)
            latency = time.monotonic() - t0
            usage_total.add(turn.usage)
            trace.log({
                "event": "model_turn", "step": step, "latency_s": round(latency, 2),
                "text": turn.text, "error": turn.error,
                "tool_calls": [{"name": c.name, "arguments": c.arguments} for c in turn.tool_calls],
                "usage": [turn.usage.input_tokens, turn.usage.output_tokens] if turn.usage else None,
            })

            if turn.error is not None:
                parse_failures += 1
                echo(f"[{step}] ✗ unparseable reply ({turn.error[:120]})")
                if parse_failures > 1:
                    result.status, result.summary = "parse_error", turn.error
                    break
                messages.append(AssistantMessage(text=turn.text or "(unparseable reply)"))
                messages.append(UserMessage.text(f"ERROR: {turn.error} Reply again with a single action."))
                continue
            parse_failures = 0

            messages.append(AssistantMessage(text=turn.text, tool_calls=turn.tool_calls, usage=turn.usage))
            if turn.text and not turn.tool_calls:
                echo(f"[{step}] model: {turn.text.strip()[:160]}")

            if not turn.tool_calls:
                result.status, result.summary = "no_tool_calls", turn.text
                break

            for call in turn.tool_calls:
                if call.name == "done":
                    result.status = "done"
                    result.summary = str(call.arguments.get("summary") or turn.text or "")
                    echo(f"[{step}] ✔ done: {result.summary[:200]}")
                    trace.log({"event": "done", "step": step, "summary": result.summary})
                    break

                echo(f"[{step}] {call.name}({json.dumps(call.arguments)})")
                tool = next((t for t in server_tools if t.name == call.name), None)
                if tool is None:
                    res = ToolResult(call_id=call.id, tool_name=call.name,
                                     text=f"Unknown tool '{call.name}'. Available: "
                                          f"{', '.join(t.name for t in tools)}", is_error=True)
                elif _is_risky(tool, call.arguments) and not cfg.auto_approve:
                    if await _confirm(call.name, call.arguments):
                        res = await _execute(client, call)
                    else:
                        res = ToolResult(call_id=call.id, tool_name=call.name,
                                         text="User declined to run this tool. Continue differently.", is_error=True)
                else:
                    res = await _execute(client, call)

                trace.log({
                    "event": "tool_result", "step": step, "tool": call.name, "arguments": call.arguments,
                    "text": res.text, "is_error": res.is_error, "images": [trace.save_image(i) for i in res.images],
                })
                echo(_result_line(step, res))
                messages.append(res)
            if result.status == "done":
                break
        else:
            result.status = "max_steps"
            result.summary = f"gave up after {cfg.max_steps} steps"
    except (KeyboardInterrupt, asyncio.CancelledError):
        result.status = "interrupted"
        result.summary = "interrupted by user"
    except Exception as e:  # provider or transport blew up
        result.status, result.error = "error", f"{type(e).__name__}: {e}"

    result.usage = usage_total
    result.run_dir = str(trace.dir)
    trace.log({"event": "run_end", "status": result.status, "summary": result.summary,
               "steps": result.steps, "usage": [usage_total.input_tokens, usage_total.output_tokens],
               "error": result.error})
    return result


async def _execute(client: MCPClient, call) -> ToolResult:
    t0 = time.monotonic()
    try:
        text, images, is_error = await client.call(call.name, call.arguments)
        return ToolResult(call_id=call.id, tool_name=call.name, text=text, images=images, is_error=is_error)
    except Exception as e:  # MCP-level failure (timeout, transport, ...)
        return ToolResult(call_id=call.id, tool_name=call.name,
                          text=f"{type(e).__name__}: {e}", is_error=True)
    finally:
        del t0  # latency logged from the trace timestamps


def _result_line(step: int, res: ToolResult) -> str:
    first = res.text.splitlines()[0] if res.text else ""
    mark = "✗" if res.is_error else "→"
    shots = f" [+{len(res.images)} shot]" if res.images else ""
    line = f"[{step}] {mark} {first[:140]}{shots}"
    if res.is_error and res.text.count("\n") > 0:
        more = res.text.splitlines()
        line = f"[{step}] ✗ {' / '.join(m[:80] for m in more[:2])[:160]}"
    return line
