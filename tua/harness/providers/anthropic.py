"""Claude via the anthropic SDK: native tool use, images inside `tool_result`."""

from __future__ import annotations

import base64
import os

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
from tua.harness.providers.base import Capabilities, Turn, sanitize_schema


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str, max_tokens: int = 4096, api_key: str | None = None):
        try:
            from anthropic import AsyncAnthropic
        except ImportError as e:  # pragma: no cover
            raise SystemExit("anthropic SDK missing — install with: pip install 'tv-mcp[harness]'") from e
        self.model = model
        self.max_tokens = max_tokens
        self.capabilities = Capabilities()
        self.client = AsyncAnthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    async def complete(self, system: str, messages: list[Message], tools) -> Turn:
        resp = await self.client.messages.create(
            model=self.model,
            system=system,
            messages=self._convert_all(messages),
            max_tokens=self.max_tokens,
            tools=[
                {"name": t.name, "description": t.description or "", "input_schema": sanitize_schema(t.schema)}
                for t in tools
            ],
        )
        text = "\n".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        calls = [
            ToolCall(id=b.id, name=b.name, arguments=dict(b.input or {}))
            for b in resp.content
            if getattr(b, "type", "") == "tool_use"
        ]
        usage = Usage(getattr(resp.usage, "input_tokens", 0), getattr(resp.usage, "output_tokens", 0))
        return Turn(text=text, tool_calls=calls, usage=usage, stop_reason=resp.stop_reason or "")

    # ----------------------------------------------------------------- conversion

    def _keep_set(self, messages: list[Message]) -> set[int]:
        """`id()`s of the newest images to keep, bounded by capabilities."""
        cap = self.capabilities.max_images if self.capabilities else 4
        if not (self.capabilities and self.capabilities.vision):
            cap = 0
        keep: set[int] = set()
        seen = 0
        for m in reversed(messages):
            if isinstance(m, ToolResult):
                for img in reversed(m.images):
                    if seen < cap:
                        keep.add(id(img))
                        seen += 1
        return keep

    def _convert_all(self, messages: list[Message]) -> list[dict]:
        keep = self._keep_set(messages)
        out: list[dict] = []
        for m in messages:
            converted = self._convert(m, keep)
            # Anthropic wants alternating roles; merge consecutive user turns.
            if converted["role"] == "user" and out and out[-1]["role"] == "user":
                out[-1]["content"].extend(converted["content"])
            else:
                out.append(converted)
        return out or [{"role": "user", "content": [{"type": "text", "text": "(empty conversation)"}]}]

    def _convert(self, m: Message, keep: set[int]) -> dict:
        if isinstance(m, AssistantMessage):
            content: list[dict] = []
            if m.text:
                content.append({"type": "text", "text": m.text})
            content.extend(
                {"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments} for c in m.tool_calls
            )
            return {"role": "assistant", "content": content or [{"type": "text", "text": "(no reply)"}]}

        if isinstance(m, ToolResult):
            content = []
            if m.text:
                content.append({"type": "text", "text": m.text})
            kept = 0
            for img in m.images:
                if id(img) in keep:
                    content.append(self._image_block(img))
                    kept += 1
            if kept < len(m.images):
                content.append({"type": "text", "text": f"[{len(m.images) - kept} screenshot(s) omitted]"})
            return {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": m.call_id, "content": content or [{"type": "text", "text": "(no output)"}],
                     "is_error": m.is_error}
                ],
            }

        assert isinstance(m, UserMessage)
        content = []
        for p in m.parts:
            if isinstance(p, TextPart):
                content.append({"type": "text", "text": p.text})
            elif id(p.image) in keep:
                content.append(self._image_block(p.image))
            else:
                content.append({"type": "text", "text": "[screenshot omitted]"})
        return {"role": "user", "content": content or [{"type": "text", "text": "(empty)"}]}

    def _image_block(self, image: Image) -> dict:
        if not self.capabilities.vision:
            return {"type": "text", "text": "[screenshot omitted: model has no vision]"}
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": image.mime,
                "data": base64.b64encode(image.data).decode(),
            },
        }
