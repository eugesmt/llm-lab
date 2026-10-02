"""API tests for persisted conversations and task-aware RAG orchestration."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app
from app.providers.mock import MockProvider
from app.schemas.llm import LLMResponse, RAGSource
from app.storage import ChatStore
from tests.conftest import MOCK_CATALOG


class RecordingLLMService:
    """Small service double that records the orchestration contract."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def chat(self, **kwargs: object) -> LLMResponse:
        self.calls.append(kwargs)
        return LLMResponse(
            provider="mock",
            model=str(kwargs["model"]),
            answer=f"answer: {kwargs['prompt']}",
            latency_ms=1.0,
            finish_reason="stop",
            rag_mode=str(kwargs.get("rag_mode", "off")),
            rag_sources=[
                RAGSource(
                    chunk_id="chunk-1",
                    source="docs/example.md",
                    section="Example",
                    similarity=0.9,
                )
            ],
        )


def make_client(tmp_path: Path) -> tuple[TestClient, RecordingLLMService]:
    service = RecordingLLMService()
    app = create_app(
        providers={"mock": MockProvider()},
        catalog=MOCK_CATALOG,
        chat_store=ChatStore(tmp_path / "chat.sqlite"),
    )
    app.state.llm_service = service
    return TestClient(app), service


def create_conversation(client: TestClient) -> str:
    response = client.post("/api/conversations")
    assert response.status_code == 201
    return response.json()["id"]


def test_create_and_get_conversation(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path)
    conversation_id = create_conversation(client)

    response = client.get(f"/api/conversations/{conversation_id}")

    assert response.status_code == 200
    data = response.json()
    assert data["conversation"]["id"] == conversation_id
    assert data["messages"] == []
    assert data["task_state"]["goal"] is None


def test_multi_turn_persistence_and_rewritten_retrieval_query(tmp_path: Path) -> None:
    client, service = make_client(tmp_path)
    conversation_id = create_conversation(client)

    first = client.post(
        "/api/chat",
        json={
            "model": "mock-fast",
            "prompt": "Цель: настроить SIP service. Ограничение: только Docker.",
            "conversation_id": conversation_id,
            "rag_mode": "on",
        },
    )
    second = client.post(
        "/api/chat",
        json={
            "model": "mock-fast",
            "prompt": "А как это запустить?",
            "conversation_id": conversation_id,
            "rag_mode": "on",
        },
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["conversation_id"] == conversation_id
    assert second.json()["task_state"]["goal"] == "настроить SIP service"
    assert second.json()["task_state"]["constraints"] == ["только Docker"]

    retrieval_query = service.calls[0]["retrieval_query"]
    assert isinstance(retrieval_query, str)
    assert "настроить SIP service" in retrieval_query
    assert "только Docker" in retrieval_query
    conversation_context = str(service.calls[0]["conversation_context"])
    assert "ПАМЯТЬ ЗАДАЧИ" in conversation_context
    assert "Цель: настроить SIP service" in conversation_context
    assert "answer:" not in conversation_context

    stored = client.get(f"/api/conversations/{conversation_id}")
    assert stored.status_code == 200
    messages = stored.json()["messages"]
    assert [message["role"] for message in messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert messages[-1]["response_json"]["rag_sources"][0]["chunk_id"] == "chunk-1"


def test_conversation_isolation(tmp_path: Path) -> None:
    client, service = make_client(tmp_path)
    first_id = create_conversation(client)
    second_id = create_conversation(client)

    response = client.post(
        "/api/chat",
        json={
            "model": "mock-fast",
            "prompt": "Цель: первая задача.",
            "conversation_id": first_id,
        },
    )
    assert response.status_code == 200

    other = client.get(f"/api/conversations/{second_id}").json()
    assert other["messages"] == []
    assert other["task_state"]["goal"] is None
    assert len(service.calls) == 0


def test_task_state_recall_uses_memory_as_source_without_llm(tmp_path: Path) -> None:
    client, service = make_client(tmp_path)
    conversation_id = create_conversation(client)
    client.post(
        "/api/chat",
        json={
            "model": "mock-fast",
            "prompt": "Цель: локально развернуть LiveKit. Ограничение: только Docker.",
            "conversation_id": conversation_id,
        },
    )
    calls_before_recall = len(service.calls)

    response = client.post(
        "/api/chat",
        json={
            "model": "mock-fast",
            "prompt": "Напомни мою цель и все ограничения.",
            "conversation_id": conversation_id,
            "rag_mode": "enhanced",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert len(service.calls) == calls_before_recall
    assert data["finish_reason"] == "task_state_recall"
    assert data["rag_sources"][0]["chunk_id"] == "task_state"
    assert data["rag_sources"][0]["source"] == "Память задачи"
    assert data["citations"][0]["chunk_id"] == "task_state"
    assert "локально развернуть LiveKit" in data["answer"]
    assert "только Docker" in data["answer"]


def test_task_state_update_uses_memory_source_without_llm(tmp_path: Path) -> None:
    client, service = make_client(tmp_path)
    conversation_id = create_conversation(client)

    response = client.post(
        "/api/chat",
        json={
            "model": "mock-fast",
            "prompt": "Цель: локально развернуть LiveKit.",
            "conversation_id": conversation_id,
            "rag_mode": "enhanced",
        },
    )

    assert response.status_code == 200
    assert service.calls == []
    assert response.json()["finish_reason"] == "task_state_updated"
    assert response.json()["rag_sources"][0]["source"] == "Память задачи"


def test_stateless_backward_compatibility(tmp_path: Path) -> None:
    client, service = make_client(tmp_path)

    response = client.post(
        "/api/chat",
        json={"model": "mock-fast", "prompt": "Привет"},
    )

    assert response.status_code == 200
    assert "conversation_id" not in response.json()
    assert "conversation_context" not in service.calls[0]
    assert "retrieval_query" not in service.calls[0]


def test_unknown_conversation_is_404(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path)

    response = client.post(
        "/api/chat",
        json={
            "model": "mock-fast",
            "prompt": "Привет",
            "conversation_id": "missing",
        },
    )
    lookup = client.get("/api/conversations/missing")

    assert response.status_code == 404
    assert lookup.status_code == 404
