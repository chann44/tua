# Plan: model-agnostic computer-use harness

## Goal

Ship `tua run "<task>" --model <provider>/<model>`: an agent loop that drives the TV with **any** vision model (Claude, GPT, Gemini, or local models through Ollama / LM Studio / vLLM), without needing Claude Code or another MCP client.

## Key decision: the harness is an MCP client

The harness launches `tv-mcp` as a subprocess and talks to it over stdio, the way Claude Code does.

- **No refactor of `tv_mcp.py`.** `list_tools` already returns a JSON schema for each tool, and tool results already include images (`screenshot`) and text.
- **One source of truth.** A tool added to the server shows up in the harness automatically.
- **Works with other servers too.** Pointing the harness at a different MCP server (for example a desktop or browser one) turns it into a general computer-use harness.
- Stdio overhead is a few milliseconds, which is nothing next to ADB screenshots (about 300 ms) and model latency.

The other option would be to import the tool functions directly. That needs a split into `device.py` and `tools.py` first, and it ties the harness to this one server. Skip it unless profiling shows stdio is a bottleneck.

## Architecture

```
tua/harness/
  cli.py          # `tua run`, flags, model string parsing
  loop.py         # observe → act → verify loop, step budget, stop conditions
  mcp_client.py   # spawn server, list_tools, call_tool → normalized results
  messages.py     # internal message format (text, image, tool_call, tool_result)
  context.py      # screenshot pruning, history trimming
  providers/
    base.py       # Provider.complete(system, messages, tools) -> Turn
    anthropic.py
    openai_compat.py   # OpenAI, OpenRouter, Ollama, LM Studio, vLLM, Groq, Gemini (OpenAI endpoint)
    json_mode.py       # fallback for models without native tool calling
  trace.py        # per-run JSONL + screenshots in runs/<timestamp>/
```

### Provider layer

Two native adapters cover almost everything:

| Adapter          | Covers                                                                     |
| ---------------- | -------------------------------------------------------------------------- |
| `anthropic`      | Claude models                                                              |
| `openai_compat`  | OpenAI, Gemini (OpenAI-compatible endpoint), OpenRouter, Ollama, LM Studio, vLLM |

Model string format: `anthropic/claude-sonnet-5-5`, `openai/gpt-…`, `ollama/qwen2.5vl`, `openrouter/<id>`. Base URL and key are resolved from env vars per provider.

Translation rules that need care:

- **Images in tool results.** Anthropic accepts images inside `tool_result`. OpenAI Chat Completions does not accept images in `tool` messages, so send the text part as the tool result and the image as a follow-up `user` message.
- **Tool schemas.** Convert MCP `inputSchema` to each provider's format and strip keywords that some providers reject (`$defs`, `anyOf` with `null`; flatten `int | None` to optional).
- **Parallel tool calls.** Run them in order, never concurrently, because the TV is a single device.

Each model gets capability flags: `vision`, `native_tools`, `max_images`, `context_tokens`. They come from a small built-in table that the user can override with CLI flags.

### Fallback for weak or local models: JSON action mode

Models without reliable function calling get the tool list in the system prompt and reply with a single fenced JSON block, `{"tool": "...", "args": {...}}`. The harness parses it and, on a parse failure, re-prompts once with the error. This is what makes small local VLMs usable.

### Grounding help: set-of-marks

`get_ui` already returns element bounds. Add an optional `--marks` mode: draw numbered boxes on the screenshot and expose a `select_mark(n)` tool that maps to `select_element` / `tap`. Small models are much better at "pick box 7" than at producing pixel coordinates. This doesn't help in YouTube or Netflix, which expose no UI tree, so those fall back to plain screenshots.

### Loop

1. The system prompt is the server's `instructions` plus the harness rules: finish by calling `done(summary)`, and verify after each action.
2. Each step: model turn → execute tool calls → append results → prune context.
3. The run stops on `done`, on a turn with no tool calls, at `--max-steps` (default 40), or on Ctrl-C.
4. Context pruning keeps the last 3 screenshots and replaces older ones with `[screenshot omitted]`. Long text results are truncated.
5. Safety: tools marked `destructive_hint` (`adb_shell`, `power off`) need a y/n confirmation unless `--yes` is passed.

### Tracing

Each run writes `runs/<ts>/trace.jsonl` (prompts, tool calls, results, token usage, latency) plus the screenshots. A later `tua replay runs/<ts>` can render this as an HTML timeline. This is essential for comparing models.

## Phases

| # | Deliverable | Done when |
| - | ----------- | --------- |
| 1 | `mcp_client` + `loop` + `anthropic` adapter + CLI | `tua run "open YouTube" --model anthropic/claude-sonnet-5-5` works end to end |
| 2 | `openai_compat` adapter + image-in-tool workaround | The same task works on GPT, Gemini, and one Ollama VLM |
| 3 | Context pruning, step budget, confirmations, tracing | A 30-step task finishes without overflowing context, and the trace is readable |
| 4 | JSON action mode + capability table | A local model without tool calling finishes a simple task |
| 5 | Set-of-marks grounding | Measurable success-rate gain for small models in native apps |
| 6 | Eval suite: `tua eval` | About 10 tasks with automatic checks, and a per-model success, steps, and cost table |

Eval task ideas, each checked with `now_playing`, `tv_status`, or the foreground app: play a YouTube video by name, set the volume to 10, open Settings → Display, launch Netflix title X, pause and seek +30s.

## Dependencies

`mcp` (already a dependency, ships the client), `anthropic`, `openai`. Declare them as the `harness` extra (`pip install tv-mcp[harness]`) so the MCP server install stays light. Skip LiteLLM: two thin adapters are less code to debug than a large abstraction, and they make image handling explicit.

## Open questions

- Should the harness also be exposed as an MCP tool (`run_task`), so one agent can hand off whole TV tasks to a cheaper model?
- Do we want streaming output in the CLI, or a step-by-step summary only?
- How should budgets work: a step cap only, or also a token or dollar cap per run?
