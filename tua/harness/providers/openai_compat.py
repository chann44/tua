"""OpenAI-compatible chat adapter: OpenAI, Gemini (OpenAI endpoint), OpenRouter,
Groq, DeepSeek, Together, Ollama, LM Studio, vLLM and generic `compat` endpoints.

Translation rules that need care:
- Images can NOT ride inside `tool` messages (Chat Completions rejects them), so tool
  results send only their text and all images from that turn follow as one `user`
  message *after* the whole batch of `tool` messages.
- The newest `capabilities.max_images` images are kept; older ones become placeholders.
- Tool schemas are sanitized (`$defs`, `anyOf[...null]` flattened) in `base.sanitize_schema`.
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass

from tua.harness.messages import (
    AssistantMessage,
    Image,
    Message,
    TextPart,
    ToolCall,
    ToolResult,
    Usage,
    UserMessage,
)
from tua.harness.providers.base import Turn, sanitize_schema


@dataclass(frozen=True)
class Endpoint:
    base_url: str
    key_env: str
    key_optional: bool = False
    base_env: str = ""  # env var that overrides base_url


ENDPOINTS: dict[str, Endpoint] = {
    "openai": Endpoint("https://api.openai.com/v1", "OPENAI_API_KEY", base_env="OPENAI_BASE_URL"),
    "openrouter": Endpoint("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "groq": Endpoint("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "gemini": Endpoint("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY"),
    "deepseek": Endpoint("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
    "together": Endpoint("https://api.together.xyz/v1", "TOGETHER_API_KEY"),
    "fireworks": Endpoint("https://api.fireworks.ai/inference/v1", "FIREWORKS_API_KEY"),
    "ollama": Endpoint("http://localhost:11434/v1", "OLLAMA_API_KEY", key_optional=True, base_env="OLLAMA_BASE_URL"),
    "lmstudio": Endpoint("http://localhost:1234/v1", "LMSTUDIO_API_KEY", key_optional=True),
    "vllm": Endpoint("http://localhost:8000/v1", "VLLM_API_KEY", key_optional=True, base_env="VLLM_BASE_URL"),
    "compat": Endpoint("", "OPENAI_COMPAT_API_KEY", key_optional=True, base_env="OPENAI_COMPAT_BASE_URL"),
}


class OpenAICompatProvider:
    def __init__(self, provider: str, model: str, base_url: str | None = None,
                 api_key: str | None = None, max_tokens: int = 4096):
        try:
            from openai import AsyncOpenAI
        except ImportError as e:  # pragma: no cover
            raise SystemExit("openai SDK missing — install with: pip install 'tv-mcp[harness]'") from e

        spec = ENDPOINTS[provider]
        self.name = provider
        self.model = model
        self.max_tokens = max_tokens
        self.capabilities = None  # set by the CLI from the capability table

        base = base_url or (os.environ.get(spec.base_env) if spec.base_env else None) or spec.base_url
        if not base:
            raise SystemExit(
                f"'{provider}' has no default base URL. Pass --base-url or set {spec.base_env}."
            )
        key = api_key or os.environ.get(spec.key_env)
        if not key:
            if spec.key_optional:
                key = "not-needed"  # local servers; the SDK requires a non-empty string
            else:
                raise SystemExit(f"Set {spec.key_env} (or pass --api-key) to use {provider}.")
        self.client = AsyncOpenAI(base_url=base, api_key=key)

    async def complete(self, system: str, messages: list[Message], tools) -> Turn:
        payload: dict = {
            "model": self.model,
            "messages": self._convert_all(system, messages),
        }
        if tools:
            payload["tools"] = [
                {"type": "function", "function": {"name": t.name, "description": t.description or "",
                                                  "parameters": sanitize_schema(t.schema)}}
                for t in tools
            ]
        payload.update(self._max_tokens_param())
        resp = await self.client.chat.completions.create(**payload)
        msg = resp.choices[0].message
        calls = []
        for i, c in enumerate(msg.tool_calls or []):
            try:
                args = json.loads(c.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw": c.function.arguments}
            calls.append(ToolCall(id=c.id or f"call_{i}", name=c.function.name, arguments=args))
        usage = Usage(resp.usage.prompt_tokens, resp.usage.completion_tokens) if resp.usage else None
        return Turn(text=msg.content or "", tool_calls=calls, usage=usage,
                    stop_reason=resp.choices[0].finish_reason or "")

    async def raw_chat(self, system: str, messages: list[Message]) -> tuple[str, Usage | None]:
        """Plain completion without tools — used by JSON-action mode."""
        payload = {"model": self.model, "messages": self._convert_all(system, messages)}
        payload.update(self._max_tokens_param())
        resp = await self.client.chat.completions.create(**payload)
        usage = Usage(resp.usage.prompt_tokens, resp.usage.completion_tokens) if resp.usage else None
        return resp.choices[0].message.content or "", usage

    # ----------------------------------------------------------------- helpers

    def _max_tokens_param(self) -> dict:
        # OpenAI's newer models (o*, gpt-5*) reject `max_tokens` outright.
        if self.name == "openai" and (self.model.startswith("o") or self.model.startswith("gpt-5")):
            return {"max_completion_tokens": self.max_tokens}
        return {"max_tokens": self.max_tokens}

    def _keep_set(self, messages: list[Message]) -> set[int]:
        """`id()`s of the newest images to keep (bounded by capabilities)."""
        cap = self.capabilities.max_images if self.capabilities else 4
        if not (self.capabilities and self.capabilities.vision):
            cap = 0
        keep, seen = set(), 0
        for m in reversed(messages):
            if isinstance(m, ToolResult):
                for img in reversed(m.images):
                    if seen < cap:
                        keep.add(id(img))
                        seen += 1
        return keep

    def _convert_all(self, system: str, messages: list[Message]) -> list[dict]:
        keep = self._keep_set(messages)
        out: list[dict] = [{"role": "system", "content": system}]
        pending_images: list[Image] = []

        def flush_images() -> None:
            nonlocal pending_images
            if not pending_images:
                return
            content: list[dict] = [
                {"type": "text", "text": "Tool-result screenshots from the TV (oldest first):"}
            ]
            content.extend(self._image_part(i) for i in pending_images)
            out.append({"role": "user", "content": content})
            pending_images = []

        for m in messages:
            if isinstance(m, AssistantMessage):
                flush_images()
                if not m.text and not m.tool_calls:
                    continue  # nothing to say
                entry: dict = {"role": "assistant", "content": m.text or None}
                if m.tool_calls:
                    entry["tool_calls"] = [
                        {"id": c.id, "type": "function",
                         "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                        for c in m.tool_calls
                    ]
                out.append(entry)
            elif isinstance(m, ToolResult):
                # Text-only here; images flush as a user message after the tool batch.
                kept = [i for i in m.images if id(i) in keep]
                text = m.text or "(no output)"
                if len(kept) < len(m.images):
                    text += f"\n[{len(m.images) - len(kept)} screenshot(s) omitted]"
                out.append({"role": "tool", "tool_call_id": m.call_id, "content": text})
                pending_images.extend(kept)
            else:  # UserMessage
                flush_images()
                content = []
                for p in m.parts:
                    if isinstance(p, TextPart):
                        content.append({"type": "text", "text": p.text})
                    elif id(p.image) in keep:
                        content.append(self._image_part(p.image))
                    else:
                        content.append({"type": "text", "text": "[screenshot omitted]"})
                out.append({"role": "user", "content": content or [{"type": "text", "text": "(empty)"}]})

        flush_images()  # screenshots from the latest tool results
        return out

    def _image_part(self, image: Image) -> dict:
        if not self.capabilities or not self.capabilities.vision:
            return {"type": "text", "text": "[screenshot omitted: model has no vision]"}
        b64 = base64.b64encode(image.data).decode()
        return {"type": "image_url", "image_url": {"url": f"data:{image.mime};base64,{b64}"}}
