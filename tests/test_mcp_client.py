from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from inspect import signature

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from app.config import Settings
from app.errors import MCPToolError, MCPUnavailableError
from app.main import create_app
from app.mcp import client as client_module
from app.mcp.client import StreamableHTTPMCPClient, TaskManagerMCPClient


def test_settings_require_obsidian_url_and_key_together() -> None:
    assert Settings(_env_file=None).mcp_max_rounds == 10
    with pytest.raises(ValidationError):
        Settings(_env_file=None, obsidian_mcp_url="https://obsidian.example/mcp")
    with pytest.raises(ValidationError):
        Settings(_env_file=None, obsidian_mcp_api_key=SecretStr("secret"))


def test_streamable_http_client_keeps_headers_private() -> None:
    secret = "dummy-secret"
    client = StreamableHTTPMCPClient(
        "https://example.test/mcp",
        timeout_s=12,
        headers={"Authorization": f"Bearer {secret}"},
        server_name="Obsidian",
    )
    assert client._headers == {"Authorization": f"Bearer {secret}"}
    assert client._server_name == "Obsidian"
    assert secret not in repr(client)

    settings = Settings(
        _env_file=None,
        obsidian_mcp_url="https://example.test/mcp",
        obsidian_mcp_api_key=SecretStr(secret),
    )
    assert secret not in repr(settings)


def test_task_manager_client_constructor_remains_compatible() -> None:
    client = TaskManagerMCPClient("http://127.0.0.1:8090/mcp", timeout_s=30)
    assert client._headers is None
    assert client._server_name == "Task Manager"


@pytest.mark.asyncio
async def test_connect_passes_headers_and_closes_session(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    @asynccontextmanager
    async def fake_transport(url: str, *, http_client: httpx.AsyncClient):
        seen["url"] = url
        seen["http_client"] = http_client
        yield object(), object(), None

    class FakeSession:
        def __init__(self, read_stream, write_stream, *, read_timeout_seconds):
            seen["timeout"] = read_timeout_seconds
            self.closed = False
            seen["session"] = self

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            self.closed = True

        async def initialize(self):
            seen["initialized"] = True

    monkeypatch.setattr(client_module, "streamable_http_client", fake_transport)
    monkeypatch.setattr(client_module, "ClientSession", FakeSession)

    client = StreamableHTTPMCPClient(
        "https://obsidian.example/mcp",
        timeout_s=12,
        headers={"Authorization": "Bearer secret"},
        server_name="Obsidian",
    )
    async with client.connect():
        assert seen["initialized"] is True

    assert seen["url"] == "https://obsidian.example/mcp"
    http_client = seen["http_client"]
    assert isinstance(http_client, httpx.AsyncClient)
    assert http_client.headers["authorization"] == "Bearer secret"
    assert http_client.timeout.read is None
    assert http_client.is_closed
    assert seen["session"].closed
    assert seen["timeout"].total_seconds() == 12


def test_mcp_sdk_transport_uses_public_http_client_parameter() -> None:
    parameters = signature(client_module.streamable_http_client).parameters
    assert "http_client" in parameters
    assert "headers" not in parameters


@pytest.mark.asyncio
async def test_setup_error_survives_failing_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    primary = ValueError("initialize failed")

    @asynccontextmanager
    async def failing_transport(url: str, *, http_client: httpx.AsyncClient):
        try:
            yield object(), object(), None
        finally:
            raise RuntimeError("transport cleanup failed")

    class FailingSession:
        def __init__(self, read_stream, write_stream, *, read_timeout_seconds):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            raise RuntimeError("session cleanup failed")

        async def initialize(self):
            raise primary

    monkeypatch.setattr(client_module, "streamable_http_client", failing_transport)
    monkeypatch.setattr(client_module, "ClientSession", FailingSession)

    with pytest.raises(MCPUnavailableError) as caught:
        async with StreamableHTTPMCPClient("https://example.test/mcp", timeout_s=1).connect():
            pass
    assert caught.value.__cause__ is primary


@pytest.mark.asyncio
async def test_body_error_and_cancellation_survive_failing_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @asynccontextmanager
    async def transport(url: str, *, http_client: httpx.AsyncClient):
        yield object(), object(), None

    class FailingSession:
        def __init__(self, read_stream, write_stream, *, read_timeout_seconds):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            raise RuntimeError("session cleanup failed")

        async def initialize(self):
            pass

    monkeypatch.setattr(client_module, "streamable_http_client", transport)
    monkeypatch.setattr(client_module, "ClientSession", FailingSession)
    client = StreamableHTTPMCPClient("https://example.test/mcp", timeout_s=1)

    with pytest.raises(MCPToolError, match="body failure"):
        async with client.connect():
            raise MCPToolError("body failure")

    with pytest.raises(asyncio.CancelledError):
        async with client.connect():
            raise asyncio.CancelledError


def test_create_app_registers_servers_and_preserves_injections() -> None:
    class FakeMCP:
        pass

    taskmanager = FakeMCP()
    obsidian = FakeMCP()
    settings = Settings(_env_file=None)
    app = create_app(
        settings=settings,
        providers={},
        catalog=[],
        taskmanager_mcp=taskmanager,
        mcp_servers={"obsidian": obsidian},
    )
    assert app.state.mcp_servers["taskmanager"] is taskmanager
    assert app.state.mcp_servers["obsidian"] is obsidian

    configured = Settings(
        _env_file=None,
        obsidian_mcp_url="https://example.test/mcp",
        obsidian_mcp_api_key=SecretStr("dummy-secret"),
    )
    configured_app = create_app(settings=configured, providers={}, catalog=[])
    assert set(configured_app.state.mcp_servers) == {"taskmanager", "obsidian"}
