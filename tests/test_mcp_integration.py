"""Opt-in live contract test against the Task Manager MCP endpoint."""

from __future__ import annotations

import json
import os
from urllib.parse import urlparse

import pytest

from app.mcp.client import StreamableHTTPMCPClient, TaskManagerMCPClient

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_live_taskmanager_mcp_lists_and_calls_os_tools() -> None:
    if os.getenv("RUN_REAL_MCP_TESTS") != "1":
        pytest.skip("set RUN_REAL_MCP_TESTS=1 to call the local MCP endpoint")

    url = os.getenv("TASKMANAGER_MCP_URL", "http://127.0.0.1:8090/mcp")
    client = TaskManagerMCPClient(url=url, timeout_s=10)

    async with client.connect() as connection:
        tools = await connection.list_tools()
        tool_names = {tool.name for tool in tools}
        assert {
            "get_system_overview",
            "get_top_processes",
            "get_resource_history",
        }.issubset(tool_names)

        overview = json.loads(await connection.call_tool("get_system_overview", {}))
        assert "captured_at" in overview
        assert "cpu_percent" in overview
        assert "memory_percent" in overview

        if "save_diagnostic_report" in tool_names:
            report = json.loads(
                await connection.call_tool(
                    "save_diagnostic_report",
                    {"content": "# MCP test\n", "format": "markdown"},
                )
            )
            parsed_url = urlparse(report.get("download_url", ""))
            assert parsed_url.scheme in {"http", "https"}
            assert parsed_url.netloc
            assert report.get("path")


@pytest.mark.asyncio
async def test_live_multi_mcp_lists_tools_without_mutating_notes() -> None:
    """Opt-in two-server contract check; intentionally performs no Obsidian writes."""
    if os.getenv("RUN_REAL_MULTI_MCP_TESTS") != "1":
        pytest.skip("set RUN_REAL_MULTI_MCP_TESTS=1 to call both local MCP endpoints")

    obsidian_url = os.getenv("OBSIDIAN_MCP_URL")
    api_key = os.getenv("OBSIDIAN_MCP_API_KEY")
    if not obsidian_url or not api_key:
        pytest.skip("set OBSIDIAN_MCP_URL and OBSIDIAN_MCP_API_KEY for the Obsidian endpoint")

    taskmanager = TaskManagerMCPClient(
        url=os.getenv("TASKMANAGER_MCP_URL", "http://127.0.0.1:8090/mcp"),
        timeout_s=10,
    )
    obsidian = StreamableHTTPMCPClient(
        url=obsidian_url,
        timeout_s=10,
        headers={"Authorization": f"Bearer {api_key}"},
        server_name="Obsidian",
    )

    async with taskmanager.connect() as task_connection:
        task_tools = await task_connection.list_tools()
    async with obsidian.connect() as obsidian_connection:
        obsidian_tools = await obsidian_connection.list_tools()

    assert {tool.name for tool in task_tools} >= {
        "get_system_overview",
        "get_top_processes",
        "get_resource_history",
    }
    assert {tool.name for tool in obsidian_tools} >= {
        "search_simple",
        "vault_read",
        "vault_write",
    }
