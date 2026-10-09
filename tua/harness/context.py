"""Context pruning: keep the last few screenshots, truncate long tool results."""

from __future__ import annotations

from tua.harness.messages import Message, ToolResult

OMITTED = "[screenshot omitted to save context]"


def prune(messages: list[Message], keep_images: int = 3, max_text_chars: int = 6000) -> list[Message]:
    """Return a copy of `messages` fit for the model's context window.

    - images only survive in the newest `keep_images` image-bearing tool results;
      older ones are replaced by an `[screenshot omitted]` note
    - tool-result text longer than `max_text_chars` keeps its head and tail
    """
    keep_ids: set[int] = set()
    seen = 0
    for m in reversed(messages):
        if isinstance(m, ToolResult) and m.images and seen < keep_images:
            keep_ids.add(id(m))
            seen += 1

    out: list[Message] = []
    for m in messages:
        if isinstance(m, ToolResult):
            text, images = m.text, m.images
            if m.images and id(m) not in keep_ids:
                images = []
                text = f"{text}\n{OMITTED}" if text else OMITTED
            if len(text) > max_text_chars:
                head, tail = max_text_chars * 3 // 4, 1000
                cut = len(text) - head - tail
                text = f"{text[:head]}\n... [{cut} chars truncated] ...\n{text[-tail:]}"
            if text != m.text or images is not m.images:
                m = ToolResult(call_id=m.call_id, tool_name=m.tool_name, text=text,
                               images=images, is_error=m.is_error)
        out.append(m)
    return out
