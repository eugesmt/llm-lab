"""Opt-in live contract test against the Task Manager MCP endpoint."""

from __future__ import annotations

import json
import os
from urllib.parse import urlparse

import pytest

from app.mcp.client import TaskManagerMCPClient

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
