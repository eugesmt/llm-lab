"""Acceptance tests for long RAG conversations with deterministic task memory.

The tests deliberately replace the real LLM service with ``RecordingRAGService``.
This makes the checks about HTTP orchestration, persistence and memory rather
than about a particular provider's answer quality.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from app.main import create_app
from app.providers.mock import MockProvider
from app.schemas.llm import LLMResponse, RAGCitation, RAGSource
from app.storage import ChatStore
from tests.conftest import MOCK_CATALOG

SCENARIOS_PATH = Path(__file__).parents[1] / "evaluation" / "chat_memory_scenarios.json"


class RecordingRAGService:
    """Deterministic service double returning one source and one citation."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> LLMResponse:
        self.calls.append(kwargs)
        number = len(self.calls)
        source = RAGSource(
            chunk_id=f"recording_chunk_{number:03d}",
            source="recording/livekit.md",
            section="Acceptance fixture",
            similarity=0.91,
            rerank_score=0.91,
        )
        citation = RAGCitation(
            source=source.source,
            section=source.section,
            chunk_id=source.chunk_id,
            quote=f"Deterministic citation for turn {number}.",
        )
        return LLMResponse(
            provider="recording",
            model=str(kwargs["model"]),
            answer=f"Deterministic answer for turn {number}.",
            latency_ms=0.0,
            finish_reason="stop",
            rag_mode=kwargs.get("rag_mode", "on"),
            rag_sources=[source],
            citations=[citation],
            grounded=True,
        )


def load_scenarios() -> list[dict[str, Any]]:
    payload = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))
    return payload["scenarios"]


def make_client(
    database_path: Path,
) -> tuple[TestClient, RecordingRAGService]:
    service = RecordingRAGService()
    app = create_app(
        providers={"mock": MockProvider()},
        catalog=MOCK_CATALOG,
        chat_store=ChatStore(database_path),
    )
    app.state.llm_service = service
    return TestClient(app), service


def create_conversation(client: TestClient) -> str:
    response = client.post("/api/conversations")
    assert response.status_code == 201, response.text
    return response.json()["id"]


def run_scenario(
    client: TestClient,
    service: RecordingRAGService,
    scenario: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    conversation_id = create_conversation(client)
    responses: list[dict[str, Any]] = []

    for turn in scenario["messages"]:
        response = client.post(
            "/api/chat",
            json={
                "model": "mock-fast",
                "prompt": turn["user"],
                "conversation_id": conversation_id,
                "rag_mode": "enhanced",
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        responses.append(body)

        assert body["conversation_id"] == conversation_id
        assert body["rag_sources"], f"turn {turn['turn']} has no RAG sources"
        assert body["citations"], f"turn {turn['turn']} has no citations"

        if body["finish_reason"] == "task_state_updated":
            assert body["rag_sources"][0]["source"] == "Память задачи"
            continue

        call = service.calls[-1]
        assert call.get("conversation_context"), (
            f"turn {turn['turn']} did not pass conversation_context"
        )
        assert call.get("retrieval_query"), (
            f"turn {turn['turn']} did not pass retrieval_query"
        )

        if turn.get("follow_up"):
            retrieval_query = str(call["retrieval_query"])
            previous_users = [
                item["user"]
                for item in scenario["messages"]
                if item["turn"] < turn["turn"]
            ]
            expected_context_fragments = [
                scenario["goal"],
                *scenario["constraints"][: turn["turn"]],
                *previous_users[-1:],
            ]
            assert any(
                fragment in retrieval_query for fragment in expected_context_fragments
            ), (
                f"turn {turn['turn']} follow-up was not expanded with task context: "
                f"{retrieval_query!r}"
            )

    return conversation_id, responses


def assert_full_persisted_scenario(
    client: TestClient,
    conversation_id: str,
    scenario: dict[str, Any],
) -> None:
    # The store itself must contain all turns, even if a presentation endpoint
    # accidentally truncates the history.
    store = client.app.state.chat_store
    messages = store.list_messages(conversation_id, limit=100)
    assert len(messages) == 24
    assert [message.role for message in messages] == [
        role for _ in scenario["messages"] for role in ("user", "assistant")
    ]
    assistant_messages = messages[1::2]
    assert len(assistant_messages) == 12
    for message in assistant_messages:
        assert message.response_json is not None
        assert message.response_json["rag_sources"]
        assert message.response_json["citations"]


def test_two_long_scenarios_orchestrate_memory_rag_and_isolation(tmp_path: Path) -> None:
    scenarios = load_scenarios()
    database_path = tmp_path / "chat-memory.sqlite"
    client, service = make_client(database_path)

    first_id, first_responses = run_scenario(client, service, scenarios[0])
    second_id, second_responses = run_scenario(client, service, scenarios[1])

    assert len(first_responses) == len(second_responses) == 12
    assert len(service.calls) == sum(
        1
        for scenario in scenarios
        for turn in scenario["messages"]
        if not turn["user"].startswith(("Цель:", "Ограничение:", "Термин:"))
    )
    assert_full_persisted_scenario(client, first_id, scenarios[0])
    assert_full_persisted_scenario(client, second_id, scenarios[1])

    first_state = client.get(f"/api/conversations/{first_id}").json()["task_state"]
    second_state = client.get(f"/api/conversations/{second_id}").json()["task_state"]
    for state, scenario in zip((first_state, second_state), scenarios, strict=True):
        assert state["goal"] == scenario["expected_final_state"]["goal"]
        assert state["constraints"] == scenario["expected_final_state"]["constraints"]
        assert state["terms"] == scenario["expected_final_state"]["terms"]

    first_view = client.get(f"/api/conversations/{first_id}").json()
    second_view = client.get(f"/api/conversations/{second_id}").json()
    assert all(message["conversation_id"] == first_id for message in first_view["messages"])
    assert all(message["conversation_id"] == second_id for message in second_view["messages"])
    assert first_state["goal"] not in second_view["task_state"]["goal"]
    assert second_state["goal"] not in first_view["task_state"]["goal"]


def test_history_and_task_state_survive_application_recreation(tmp_path: Path) -> None:
    scenario = load_scenarios()[0]
    database_path = tmp_path / "recreated-chat.sqlite"
    client, service = make_client(database_path)
    conversation_id, _ = run_scenario(client, service, scenario)

    client.app.state.chat_store.close()
    restored_client, restored_service = make_client(database_path)
    response = restored_client.get(f"/api/conversations/{conversation_id}")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["task_state"]["goal"] == scenario["expected_final_state"]["goal"]
    assert body["task_state"]["terms"] == scenario["expected_final_state"]["terms"]
    assert body["messages"]
    assert restored_service.calls == []


def test_get_history_returns_all_messages_for_long_scenario(tmp_path: Path) -> None:
    scenario = load_scenarios()[0]
    client, service = make_client(tmp_path / "get-history.sqlite")
    conversation_id, _ = run_scenario(client, service, scenario)

    response = client.get(f"/api/conversations/{conversation_id}")

    assert response.status_code == 200, response.text
    messages = response.json()["messages"]
    assert len(messages) == 24, (
        "GET /api/conversations/{id} must return the complete 12-turn history; "
        f"received {len(messages)} messages"
    )
