"""Query-expansion checks for common local LiveKit questions."""

from __future__ import annotations

import pytest

from app.services.rag import RAGService


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Как установить LiveKit Server?", "get.livekit.io"),
        ("Как установить LiveKit Server на Linux?", "curl -sSL"),
        ("Какие значения ключей используются в dev mode?", "devkey"),
        ("На каком порту запускается сервер?", "localhost:7880"),
        ("Как подключить клиент к локальному серверу?", "Room access token"),
        ("Можно ли запустить без Kubernetes?", "self-hosting"),
    ],
)
def test_rewrite_expands_livekit_runtime_terms(question: str, expected: str) -> None:
    assert expected in RAGService.rewrite_query(question)
