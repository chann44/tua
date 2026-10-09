"""Provider interface, capability flags and tool-schema normalization."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Protocol

from tua.harness.messages import Message, ToolCall, Usage


@dataclass(frozen=True)
class Capabilities:
    """Per-model feature flags. Built-in defaults, overridable from the CLI."""

    vision: bool = True
    native_tools: bool = True
    max_images: int = 4  # images sent per request, after pruning
    context_tokens: int = 128_000


# Built-in overrides, keyed by "<provider>/" or "<provider>/<model>" prefix (longest match wins).
BUILTIN_CAPS: dict[str, Capabilities] = {
    "ollama/": Capabilities(vision=True, native_tools=False, max_images=2, context_tokens=16_000),
    "lmstudio/": Capabilities(vision=True, native_tools=False, max_images=2, context_tokens=16_000),
    "vllm/": Capabilities(vision=True, native_tools=True, max_images=4, context_tokens=32_000),
    "gemini/": Capabilities(vision=True, native_tools=True, max_images=8, context_tokens=200_000),
    "groq/": Capabilities(vision=False, native_tools=True, max_images=0, context_tokens=64_000),
}


def builtin_caps(provider: str, model: str) -> Capabilities:
    spec = f"{provider}/{model}"
    matches = [k for k in BUILTIN_CAPS if spec.startswith(k) or provider + "/" == k]
    return replace(BUILTIN_CAPS[max(matches, key=len)]) if matches else Capabilities()


@dataclass
class ToolSpec:
    """A tool as the harness sees it (normalized from MCP `Tool`)."""

    name: str
    description: str
    schema: dict
    destructive: bool = False


@dataclass
class Turn:
    """One model reply."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage | None = None
    stop_reason: str = ""
    # Set when the reply could not be turned into actions (JSON-action mode).
    # The loop re-prompts once with this error, then gives up.
    error: str | None = None


class Provider(Protocol):
    name: str
    capabilities: Capabilities

    async def complete(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> Turn: ...


# --------------------------------------------------------------------- schema normalization

_NULL = lambda b: isinstance(b, dict) and (b.get("type") == "null" or b.get("enum") == [None])  # noqa: E731

_STRIP = {"$defs", "definitions", "$id", "$schema", "title", "$comment", "examples"}


def sanitize_schema(schema: dict) -> dict:
    """Normalize a JSON Schema for providers that reject exotic keywords.

    - resolves ``$ref`` against ``$defs`` (then drops them)
    - collapses ``anyOf: [T, null]`` (Python's ``int | None``) into ``T``
    - strips noise some APIs reject: ``title``, ``$defs``, ``$id``, ...
    """

    defs: dict = {}
    for key in ("$defs", "definitions"):
        if isinstance(schema.get(key), dict):
            defs.update(schema[key])

    def walk(node: object) -> object:
        if isinstance(node, list):
            return [walk(n) for n in node]
        if not isinstance(node, dict):
            return node
        node = dict(node)
        if "$ref" in node:
            target = defs.get(str(node["$ref"]).rsplit("/", 1)[-1])
            if isinstance(target, dict):
                node = {**{k: v for k, v in target.items() if k != "title"},
                        **{k: v for k, v in node.items() if k != "$ref"}}
        if "anyOf" in node:
            non_null = [b for b in node["anyOf"] if not _NULL(b)]
            if len(non_null) == 1 and isinstance(non_null[0], dict):
                # Re-walk the merged node: the surviving branch may carry its own `$ref`.
                return walk({**non_null[0], **{k: v for k, v in node.items() if k != "anyOf"}})
        out = {}
        for k, v in node.items():
            if k in _STRIP:
                continue
            out[k] = walk(v) if isinstance(v, (dict, list)) else v
        return out

    return walk(schema)  # type: ignore[return-value]
