"""POST /api/chat: success, validation, errors, latency and usage reporting."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.providers.mock import MockProvider
from app.schemas.llm import LLMResponse
from tests.conftest import MOCK_CATALOG


class EmptyAnswerProvider:
    """Adapter misbehaviour simulation: returns a whitespace-only answer."""

    name = "mock"

    async def chat(self, request) -> LLMResponse:  # type: ignore[no-untyped-def]
        return LLMResponse(provider="mock", model=request.model, answer="   ", latency_ms=1.0)


def test_chat_success(client: TestClient, provider_calls) -> None:
    response = client.post(
        "/api/chat", json={"model": "mock-fast", "prompt": "Привет", "reasoning_effort": "low"}
    )
    assert response.status_code == 200

    data = response.json()
    assert data["provider"] == "mock"
    assert data["model"] == "mock-fast"
    assert data["answer"] == "[mock:mock-fast] Привет"
    assert data["finish_reason"] == "stop"
    assert data["latency_ms"] >= 0

    usage = data["usage"]
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    assert usage["reasoning_tokens"] == 3  # effort was requested

    # The prompt reached the provider untouched (benchmark methodology relies on this).
    assert len(provider_calls) == 1
    assert provider_calls[0].prompt == "Привет"
    assert provider_calls[0].reasoning_effort == "low"


def test_chat_without_reasoning_effort(client: TestClient) -> None:
    response = client.post("/api/chat", json={"model": "mock-fast", "prompt": "hi"})
    assert response.status_code == 200
    assert response.json()["usage"]["reasoning_tokens"] is None


def test_chat_unknown_model_is_404(client: TestClient) -> None:
    response = client.post("/api/chat", json={"model": "nope", "prompt": "hi"})
    assert response.status_code == 404
    assert "nope" in response.json()["error"]["message"]


def test_chat_unsupported_reasoning_effort_is_422(client: TestClient) -> None:
    response = client.post(
        "/api/chat",
        json={"model": "mock-plain", "prompt": "hi", "reasoning_effort": "low"},
    )
    assert response.status_code == 422
    assert "mock-plain" in response.json()["error"]["message"]


def test_chat_effort_outside_model_capabilities_is_422(client: TestClient) -> None:
    # mock-fast supports low/high; anything else must be rejected by the backend
    # even if the UI somehow offered it.
    response = client.post(
        "/api/chat",
        json={"model": "mock-fast", "prompt": "hi", "reasoning_effort": "max"},
    )
    assert response.status_code == 422


def test_chat_invalid_request_body_is_422(client: TestClient) -> None:
    assert client.post("/api/chat", json={"model": "mock-fast"}).status_code == 422
    assert client.post("/api/chat", json={"model": "mock-fast", "prompt": ""}).status_code == 422


def test_chat_provider_error_is_502(mock_provider: MockProvider) -> None:
    mock_provider._fail_models.add("mock-fast")
    app = create_app(providers={"mock": mock_provider}, catalog=MOCK_CATALOG)
    response = TestClient(app).post("/api/chat", json={"model": "mock-fast", "prompt": "hi"})
    assert response.status_code == 502
    assert "mock failure" in response.json()["error"]["message"]


def test_chat_timeout_is_504() -> None:
    slow_provider = MockProvider(delay_s=1.0)
    app = create_app(
        providers={"mock": slow_provider},
        catalog=MOCK_CATALOG,
        settings=Settings(request_timeout_s=0.05),
    )
    response = TestClient(app).post("/api/chat", json={"model": "mock-fast", "prompt": "hi"})
    assert response.status_code == 504
    assert "timed out" in response.json()["error"]["message"]


def test_chat_unregistered_provider_is_503() -> None:
    catalog = [MOCK_CATALOG[0].model_copy(update={"provider": "ghost"})]
    app = create_app(providers={}, catalog=catalog)
    response = TestClient(app).post("/api/chat", json={"model": "mock-fast", "prompt": "hi"})
    assert response.status_code == 503


def test_chat_empty_answer_is_502() -> None:
    app = create_app(providers={"mock": EmptyAnswerProvider()}, catalog=MOCK_CATALOG)
    response = TestClient(app).post("/api/chat", json={"model": "mock-fast", "prompt": "hi"})
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "EmptyResponseError"
