"""Provider-neutral client for the Task Manager Streamable HTTP MCP server."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from datetime import timedelta
from typing import Any, Protocol

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from app.errors import MCPToolError, MCPUnavailableError
from app.schemas.llm import ToolDefinition


class MCPConnection(Protocol):
    async def list_tools(self) -> list[ToolDefinition]: ...

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str: ...


class MCPClient(Protocol):
    def connect(self) -> AbstractAsyncContextManager[MCPConnection]: ...


class _TaskManagerConnection:
    def __init__(self, client: ClientSession) -> None:
        self._client = client

    async def list_tools(self) -> list[ToolDefinition]:
        tools: list[ToolDefinition] = []
        cursor: str | None = None
        try:
            while True:
                result = await self._client.list_tools(cursor=cursor)
                tools.extend(
                    ToolDefinition(
                        name=tool.name,
                        description=tool.description,
                        input_schema=dict(tool.inputSchema),
                    )
                    for tool in result.tools
                )
                cursor = result.nextCursor
                if cursor is None:
                    return tools
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise MCPUnavailableError("Could not list Task Manager MCP tools") from exc

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        try:
            result = await self._client.call_tool(name, arguments)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise MCPUnavailableError("Task Manager MCP tool call failed") from exc

        content = self._content_for_model(result)
        if result.isError:
            raise MCPToolError(f"Task Manager MCP tool '{name}' returned an error: {content}")
        return content

    @staticmethod
    def _content_for_model(result: Any) -> str:
        if result.structuredContent is not None:
            return json.dumps(result.structuredContent, ensure_ascii=False)

        text_blocks = [
            block.text for block in result.content if getattr(block, "type", None) == "text"
        ]
        if text_blocks:
            return "\n".join(text_blocks)

        blocks = [block.model_dump(mode="json", by_alias=True) for block in result.content]
        return json.dumps(blocks, ensure_ascii=False)


class TaskManagerMCPClient:
    def __init__(self, url: str, timeout_s: float) -> None:
        self._url = url
        self._timeout_s = timeout_s

    @asynccontextmanager
    async def connect(self) -> AsyncIterator[MCPConnection]:
        stack = AsyncExitStack()
        try:
            read_stream, write_stream, _ = await stack.enter_async_context(
                streamable_http_client(self._url)
            )
            session = await stack.enter_async_context(
                ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=timedelta(seconds=self._timeout_s),
                )
            )
            await session.initialize()
        except asyncio.CancelledError:
            await stack.aclose()
            raise
        except Exception as exc:
            await stack.aclose()
            raise MCPUnavailableError("Task Manager MCP endpoint is unavailable") from exc

        try:
            yield _TaskManagerConnection(session)
        finally:
            await stack.aclose()
