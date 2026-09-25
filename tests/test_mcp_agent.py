"""Agent loop tests with an in-process fake MCP client and no network."""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest

from app.api.chat import chat
from app.errors import MCPAgentLoopError, MCPToolError
from app.schemas.chat import ChatRequest
from app.schemas.llm import (
    LLMRequest,
    LLMResponse,
    ToolCall,
    ToolCompletionRequest,
    ToolCompletionResponse,
    ToolDefinition,
    Usage,
)
from app.services.llm import LLMService
from tests.conftest import MOCK_CATALOG


class FakeMCPConnection:
    def __init__(self, tool_result: str | None = None) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.tool_result = tool_result or '{"cpu_percent": 17.5, "memory_percent": 42.0}'

    async def list_tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="get_system_overview",
                description="Return current OS metrics",
                input_schema={"type": "object", "properties": {}},
            )
        ]

    async def call_tool(self, name: str, arguments: dict[str, object]) -> str:
        self.calls.append((name, arguments))
        return self.tool_result


class FakeMCPClient:
    def __init__(self, tool_result: str | None = None) -> None:
        self.connection = FakeMCPConnection(tool_result)

    @asynccontextmanager
    async def connect(self):  # type: ignore[no-untyped-def]
        yield self.connection


class ScriptedProvider:
    name = "mock"

    def __init__(self, turns: list[ToolCompletionResponse]) -> None:
        self.turns = turns
        self.requests: list[ToolCompletionRequest] = []

    async def chat(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(provider=self.name, model=request.model, answer="plain", latency_ms=1)

    async def complete(self, request: ToolCompletionRequest) -> ToolCompletionResponse:
        self.requests.append(request)
        return self.turns.pop(0)


def turn(
    *,
    content: str | None = None,
    tool_name: str | None = None,
    finish_reason: str = "stop",
) -> ToolCompletionResponse:
    calls = []
    if tool_name is not None:
        calls = [ToolCall(id="call-1", name=tool_name, arguments={})]
    return ToolCompletionResponse(
        provider="mock",
        model="mock-fast",
        content=content,
        tool_calls=calls,
        usage=Usage(prompt_tokens=2, completion_tokens=3, total_tokens=5),
        latency_ms=1,
        finish_reason=finish_reason,
    )


def make_service(provider: ScriptedProvider, mcp: FakeMCPClient, rounds: int = 4) -> LLMService:
    return LLMService(
        providers={"mock": provider},
        catalog=MOCK_CATALOG,
        request_timeout_s=5,
        taskmanager_mcp=mcp,
        mcp_max_rounds=rounds,
    )


@pytest.mark.asyncio
async def test_agent_calls_mcp_and_returns_model_answer() -> None:
    provider = ScriptedProvider(
        [
            turn(tool_name="get_system_overview", finish_reason="tool_calls"),
            turn(content="CPU загружен на 17,5%."),
        ]
    )
    mcp = FakeMCPClient()

    response = await make_service(provider, mcp).chat(
        model="mock-fast",
        prompt="Как загружен процессор?",
        reasoning_effort="low",
        use_taskmanager_mcp=True,
    )

    assert response.answer == "CPU загружен на 17,5%."
    assert response.usage is not None
    assert response.usage.total_tokens == 10
    assert response.mcp_tools_used == ["get_system_overview"]
    assert mcp.connection.calls == [("get_system_overview", {})]
    second_messages = provider.requests[1].messages
    assert second_messages[-1].role == "tool"
    assert "17.5" in (second_messages[-1].content or "")
    assert {tool.name for tool in provider.requests[0].tools} == {"get_system_overview"}


@pytest.mark.asyncio
async def test_agent_preserves_downloadable_mcp_report() -> None:
    provider = ScriptedProvider(
        [turn(tool_name="get_system_overview", finish_reason="tool_calls"), turn(content="Готово.")]
    )
    mcp = FakeMCPClient(
        '{"report_id":"r-1","path":"diagnostic-r-1.md",'
        '"download_url":"http://127.0.0.1:8090/reports/diagnostic-r-1.md"}'
    )

    response = await make_service(provider, mcp).chat(
        model="mock-fast", prompt="Сохрани отчёт", use_taskmanager_mcp=True
    )

    assert response.artifacts[0].download_url == (
        "http://127.0.0.1:8090/reports/diagnostic-r-1.md"
    )
    assert response.artifacts[0].filename == "diagnostic-r-1.md"
    assert response.artifacts[0].report_id == "r-1"


@pytest.mark.asyncio
async def test_agent_ignores_non_http_download_url() -> None:
    provider = ScriptedProvider([turn(tool_name="get_system_overview"), turn(content="Готово.")])
    mcp = FakeMCPClient('{"download_url":"javascript:alert(1)","path":"bad.md"}')

    response = await make_service(provider, mcp).chat(
        model="mock-fast", prompt="Сохрани отчёт", use_taskmanager_mcp=True
    )

    assert response.artifacts == []


@pytest.mark.asyncio
async def test_agent_rejects_tool_not_advertised_by_mcp() -> None:
    provider = ScriptedProvider([turn(tool_name="delete_everything")])
    with pytest.raises(MCPToolError, match="unknown MCP tool"):
        await make_service(provider, FakeMCPClient()).chat(
            model="mock-fast", prompt="test", use_taskmanager_mcp=True
        )


@pytest.mark.asyncio
async def test_agent_stops_repeated_tool_loop() -> None:
    provider = ScriptedProvider([turn(tool_name="get_system_overview") for _ in range(2)])
    mcp = FakeMCPClient()
    with pytest.raises(MCPAgentLoopError, match="after 2 MCP rounds"):
        await make_service(provider, mcp, rounds=2).chat(
            model="mock-fast", prompt="loop", use_taskmanager_mcp=True
        )
    assert len(mcp.connection.calls) == 1


@pytest.mark.asyncio
async def test_chat_api_flag_enables_mcp_agent() -> None:
    provider = ScriptedProvider(
        [turn(tool_name="get_system_overview"), turn(content="Система работает нормально.")]
    )
    mcp = FakeMCPClient()
    service = make_service(provider, mcp)

    response = await chat(
        ChatRequest(
            model="mock-fast",
            prompt="Состояние системы?",
            use_taskmanager_mcp=True,
        ),
        service,
    )
    assert response.answer == "Система работает нормально."
    assert mcp.connection.calls == [("get_system_overview", {})]
