"""The harness is an MCP client: spawn the server as a subprocess, talk stdio.

No refactor of `tv_mcp.py` needed — `list_tools` gives us JSON schemas and
`call_tool` returns text + images. Point `command` at any other MCP server and
this becomes a general computer-use harness.
"""

from __future__ import annotations

import base64
import os
import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass, field

from mcp import Client
from mcp.client.stdio import StdioServerParameters, get_default_environment

from tua.harness.messages import Image
from tua.harness.providers.base import ToolSpec

DEFAULT_SERVER = [sys.executable, "-m", "tv_mcp"]


@dataclass
class MCPClient:
    command: list[str] = field(default_factory=lambda: list(DEFAULT_SERVER))
    env: dict[str, str] = field(default_factory=dict)  # extra env for the server subprocess

    async def __aenter__(self) -> "MCPClient":
        self._stack = AsyncExitStack()
        env = get_default_environment()
        # Pass TV/ADB configuration through to the server.
        env.update({k: v for k, v in os.environ.items() if k.startswith(("TV_", "ADB_"))})
        env.update(self.env)
        params = StdioServerParameters(command=self.command[0], args=self.command[1:], env=env)
        self._client = await self._stack.enter_async_context(Client(params))
        return self

    async def __aexit__(self, *exc) -> None:
        await self._stack.__aexit__(*exc)

    # ------------------------------------------------------------------ API

    @property
    def instructions(self) -> str:
        return self._client.instructions or ""

    async def list_tools(self) -> list[ToolSpec]:
        result = await self._client.list_tools()
        return [
            ToolSpec(
                name=t.name,
                description=" ".join((t.description or "").split()),
                schema=t.input_schema if isinstance(t.input_schema, dict) else {"type": "object"},
                destructive=bool(t.annotations.destructive_hint) if t.annotations else False,
            )
            for t in result.tools
        ]

    async def call(self, name: str, arguments: dict) -> tuple[str, list[Image], bool]:
        """Run a tool; returns (text, images, is_error). Never raises for tool errors."""
        result = await self._client.call_tool(name, arguments)
        texts: list[str] = []
        images: list[Image] = []
        for block in result.content or []:
            btype = getattr(block, "type", None)
            if btype == "text":
                texts.append(block.text)
            elif btype == "image":
                images.append(Image(data=base64.b64decode(block.data), mime=block.mime_type or "image/jpeg"))
        return "\n".join(t for t in texts if t).strip(), images, bool(result.is_error)
