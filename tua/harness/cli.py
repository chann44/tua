"""`tua` — model-agnostic harness CLI.

    tua run "open YouTube and play lofi beats" --model anthropic/claude-sonnet-4-5
    tua run "set volume to 10" --model ollama/qwen2.5vl --json-mode
    tua tools                       # inspect what the server exposes
"""

from __future__ import annotations

import argparse
import asyncio
import shlex
import sys
from dataclasses import replace
from pathlib import Path

from tua.harness.loop import RunConfig, run_task
from tua.harness.mcp_client import DEFAULT_SERVER, MCPClient
from tua.harness.providers.base import Provider, builtin_caps
from tua.harness.trace import Trace


def _die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(2)


def _parse_model(spec: str) -> tuple[str, str]:
    if "/" not in spec:
        _die(f"--model must look like <provider>/<model>, got '{spec}'. "
             f"Providers: anthropic, openai, openrouter, gemini, groq, deepseek, together, "
             f"fireworks, ollama, lmstudio, vllm, compat.")
    provider, model = spec.split("/", 1)
    return provider.strip(), model.strip()


def _make_provider(provider: str, model: str, args) -> Provider:
    if provider == "anthropic":
        from tua.harness.providers.anthropic import AnthropicProvider
        try:
            prov = AnthropicProvider(model=model, max_tokens=args.max_tokens)
        except SystemExit:
            raise
        except Exception as e:
            _die(str(e))
    elif provider in _endpoints():
        from tua.harness.providers.openai_compat import OpenAICompatProvider
        try:
            prov = OpenAICompatProvider(provider, model, base_url=args.base_url,
                                        api_key=args.api_key, max_tokens=args.max_tokens)
        except SystemExit as e:
            raise SystemExit(f"error: {e}") from None
    else:
        _die(f"unknown provider '{provider}'. Known: anthropic, {', '.join(sorted(_endpoints()))}")
        return None  # unreachable

    caps = builtin_caps(provider, model)
    if args.vision is not None:
        caps = replace(caps, vision=args.vision)
    if args.max_images is not None:
        caps = replace(caps, max_images=args.max_images)
    if args.context_tokens:
        caps = replace(caps, context_tokens=args.context_tokens)
    prov.capabilities = caps

    if args.json_mode:
        from tua.harness.providers.json_mode import JSONActionProvider
        try:
            prov = JSONActionProvider(prov)
        except SystemExit as e:
            raise SystemExit(f"error: {e}") from None
    return prov


def _endpoints() -> dict:
    from tua.harness.providers.openai_compat import ENDPOINTS
    return ENDPOINTS


def _server_command(value: str) -> list[str]:
    return shlex.split(value) if value else list(DEFAULT_SERVER)


async def _cmd_run(args) -> int:
    provider_name, model = _parse_model(args.model)
    provider = _make_provider(provider_name, model, args)

    cfg = RunConfig(
        task=args.task,
        model=f"{provider_name}/{model}",
        max_steps=args.max_steps,
        keep_images=args.keep_images,
        max_text_chars=args.max_text_chars,
        auto_approve=args.yes,
    )
    trace = Trace(Path(args.runs_dir), args.task)
    print(f"◆ run {cfg.model!r}: {args.task}")
    print(f"◆ trace {trace.dir}")

    async with MCPClient(command=_server_command(args.server)) as client:
        result = await run_task(provider, client, cfg, trace, echo=print)

    trace.close()
    ok = result.ok
    icon = "✔" if ok else "✗"
    print(f"{icon} {result.status} · {result.steps} steps · {result.usage} · {result.run_dir}")
    if result.summary:
        print(f"  {result.summary[:400]}")
    if result.error:
        print(f"  error: {result.error[:400]}", file=sys.stderr)
    return 0 if ok else 1


async def _cmd_tools(args) -> int:
    async with MCPClient(command=_server_command(args.server)) as client:
        print("instructions:")
        for line in client.instructions.strip().splitlines():
            print(f"  {line}")
        print("\ntools:")
        for t in await client.list_tools():
            flags = []
            if t.destructive:
                flags.append("destructive")
            print(f"  {t.name}{' (' + ', '.join(flags) + ')' if flags else ''}")
            print(f"    {t.description[:150]}")
    return 0


def _add_server_flag(p: argparse.ArgumentParser) -> None:
    p.add_argument("--server", metavar="CMD",
                   default=" ".join(DEFAULT_SERVER),
                   help="command that starts the MCP server (default: %(default)s)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tua", description="model-agnostic computer-use harness for Android TV")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="run a task with a model driving the TV")
    run.add_argument("task", help="natural-language task, e.g. 'open YouTube and play lofi beats'")
    run.add_argument("--model", required=True, help="<provider>/<model>, e.g. anthropic/claude-sonnet-4-5, ollama/qwen2.5vl")
    _add_server_flag(run)
    run.add_argument("--max-steps", type=int, default=40, help="model-turn budget (default: %(default)s)")
    run.add_argument("--max-tokens", type=int, default=4096, help="max output tokens per model turn")
    run.add_argument("--yes", action="store_true", help="auto-approve destructive tools (no y/n prompt)")
    run.add_argument("--json-mode", action="store_true",
                     help="JSON action protocol instead of native tool calling (weak/local models)")
    run.add_argument("--base-url", help="override the provider's API base URL")
    run.add_argument("--api-key", help="override the provider's API key")
    run.add_argument("--vision", dest="vision", action="store_true", default=None,
                     help="force vision on (override capability table)")
    run.add_argument("--no-vision", dest="vision", action="store_false", help="force vision off")
    run.add_argument("--max-images", type=int, default=None, help="images per request (capability override)")
    run.add_argument("--context-tokens", type=int, default=0, help="context window (capability override)")
    run.add_argument("--keep-images", type=int, default=3, help="screenshots kept in context (default: %(default)s)")
    run.add_argument("--max-text-chars", type=int, default=6000, help="tool-result truncation threshold")
    run.add_argument("--runs-dir", default="runs", help="where run traces go (default: %(default)s)")
    run.set_defaults(func=_cmd_run)

    tools = sub.add_parser("tools", help="list what the MCP server exposes")
    _add_server_flag(tools)
    tools.set_defaults(func=_cmd_tools)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        code = asyncio.run(args.func(args))
    except KeyboardInterrupt:
        print("\ninterrupted")
        raise SystemExit(130) from None
    raise SystemExit(code)


if __name__ == "__main__":
    main()
