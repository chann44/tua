"""Internal message format, shared by the loop and every provider adapter.

The loop only ever sees these types; each provider translates to/from its own
wire format (Anthropic content blocks, OpenAI chat messages, ...).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Image:
    """Raw image bytes (usually a JPEG screenshot)."""

    data: bytes
    mime: str = "image/jpeg"


@dataclass
class TextPart:
    text: str


@dataclass
class ImagePart:
    image: Image


Part = TextPart | ImagePart


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, other: "Usage | None") -> None:
        if other:
            self.input_tokens += other.input_tokens
            self.output_tokens += other.output_tokens

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    def __str__(self) -> str:
        return f"{self.input_tokens:,} in / {self.output_tokens:,} out"


@dataclass
class AssistantMessage:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage | None = None


@dataclass
class UserMessage:
    """Plain user turn: the task itself, or a JSON-mode re-prompt."""

    parts: list[Part] = field(default_factory=list)

    @classmethod
    def text(cls, text: str) -> "UserMessage":
        return cls([TextPart(text)])


@dataclass
class ToolResult:
    call_id: str  # matches ToolCall.id
    tool_name: str
    text: str = ""
    images: list[Image] = field(default_factory=list)
    is_error: bool = False


Message = AssistantMessage | UserMessage | ToolResult
