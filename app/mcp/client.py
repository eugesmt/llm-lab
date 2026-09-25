"""Provider-neutral client for the Task Manager Streamable HTTP MCP server."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from datetime import timedelta
from typing import Any, Protocol

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from app.errors import MCPToolError, MCPUnavailableError
from app.schemas.llm import ToolDefinition


class MCPConnection(Protocol):
    async def list_tools(self) -> list[ToolDefinition]: ...

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str: ...


class MCPClient(Protocol):
    def connect(self) -> AbstractAsyncContextManager[MCPConnection]: ...


class _StreamableHTTPConnection:
    def __init__(self, client: ClientSession, server_name: str) -> None:
        self._client = client
        self._server_name = server_name

    @property
    def _server_prefix(self) -> str:
        return "" if self._server_name == "MCP" else f"{self._server_name} "

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
            raise MCPUnavailableError(f"Could not list {self._server_prefix}MCP tools") from exc

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        try:
            result = await self._client.call_tool(name, arguments)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise MCPUnavailableError(f"{self._server_prefix}MCP tool call failed") from exc

        content = self._content_for_model(result)
        if result.isError:
            raise MCPToolError(
                f"{self._server_prefix}MCP tool '{name}' returned an error: {content}"
            )
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


class StreamableHTTPMCPClient:
    def __init__(
        self,
        url: str,
        timeout_s: float,
        headers: Mapping[str, str] | None = None,
        *,
        server_name: str = "MCP",
    ) -> None:
        self._url = url
        self._timeout_s = timeout_s
        self._headers = dict(headers) if headers else None
        self._server_name = server_name

    @property
    def _server_prefix(self) -> str:
        return "" if self._server_name == "MCP" else f"{self._server_name} "

    @asynccontextmanager
    async def connect(self) -> AsyncIterator[MCPConnection]:
        stack = AsyncExitStack()
        try:
            http_client = await stack.enter_async_context(
                httpx.AsyncClient(
                    headers=self._headers,
                    timeout=httpx.Timeout(self._timeout_s, read=None),
                )
            )
            read_stream, write_stream, _ = await stack.enter_async_context(
                streamable_http_client(self._url, http_client=http_client)
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
            await self._close_stack_best_effort(stack)
            raise
        except Exception as exc:
            await self._close_stack_best_effort(stack)
            raise MCPUnavailableError(f"{self._server_prefix}MCP endpoint is unavailable") from exc

        try:
            yield _StreamableHTTPConnection(session, self._server_name)
        except BaseException:
            await self._close_stack_best_effort(stack)
            raise
        else:
            try:
                await stack.aclose()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                raise MCPUnavailableError(
                    f"{self._server_prefix}MCP endpoint cleanup failed"
                ) from exc

    @staticmethod
    async def _close_stack_best_effort(stack: AsyncExitStack) -> None:
        try:
            await stack.aclose()
        except BaseException:
            pass


class TaskManagerMCPClient(StreamableHTTPMCPClient):
    """Backward-compatible Task Manager MCP client."""

    def __init__(
        self,
        url: str,
        timeout_s: float,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(url, timeout_s, headers=headers, server_name="Task Manager")
