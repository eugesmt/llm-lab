"""Unit-level contract tests for grounded answers and citation safety.

These tests describe the citations/anti-hallucination API used by the next
RAG iteration.  They intentionally use an in-memory retrieval double, so the
tests never need FAISS, SQLite, Ollama, or a real provider.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.providers.mock import MockProvider
from app.schemas.llm import LLMRequest, LLMResponse, RAGSource
from app.services.llm import LLMService
from app.services.rag import RetrievedChunk
from tests.conftest import MOCK_CATALOG

CHUNK_TEXT = (
    "Accepting SIP traffic requires two resources be created: a SIP Bridge "
    "and a SIP Dispatch Rule."
)


def _field(value: object, name: str) -> object:
    """Read a field from either a Pydantic citation or a plain mapping."""
    if isinstance(value, dict):
        return value[name]
    return getattr(value, name)


@dataclass
class RetrievalDouble:
    chunks: list[RetrievedChunk]
    relevant: bool
    calls: int = 0

    async def retrieve(self, question: str, enhanced: bool = False) -> list[RetrievedChunk]:
        self.calls += 1
        return self.chunks

    def context_is_relevant(self, chunks: list[RetrievedChunk]) -> bool:
        return self.relevant and bool(chunks)


class CitationProvider:
    name = "mock"

    def __init__(self, citation: dict[str, str]) -> None:
        self.requests: list[LLMRequest] = []
        self.citation = citation

    async def chat(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(
            provider=self.name,
            model=request.model,
            answer="Для приёма SIP-трафика нужны SIP Bridge и SIP Dispatch Rule.",
            latency_ms=1.0,
            citations=[
                {
                    "source": "server-oss/sip/README.md",
                    "section": "Creating Bridge and Dispatch Rule",
                    **self.citation,
                }
            ],
            grounded=True,
        )


class ForeignChunkProvider:
    name = "mock"

    async def chat(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(
            provider=self.name,
            model=request.model,
            answer="По данным structural_000492 нужно включить host networking.",
            latency_ms=1.0,
        )


def _chunk() -> RetrievedChunk:
    return RetrievedChunk(
        source=RAGSource(
            chunk_id="structural_000486",
            source="server-oss/sip/README.md",
            section="Creating Bridge and Dispatch Rule",
            similarity=0.91,
        ),
        text=CHUNK_TEXT,
    )


def _service(provider: object, rag: RetrievalDouble) -> LLMService:
    return LLMService(
        providers={"mock": provider},
        catalog=MOCK_CATALOG,
        request_timeout_s=5.0,
        rag_service=rag,
    )


@pytest.mark.asyncio
async def test_strong_rag_context_returns_sources_and_citations() -> None:
    chunk = _chunk()
    quote = "a SIP Bridge and a SIP Dispatch Rule"
    provider = CitationProvider(
        {
            "chunk_id": chunk.source.chunk_id,
            "quote": quote,
        }
    )
    rag = RetrievalDouble(chunks=[chunk], relevant=True)

    response = await _service(provider, rag).chat(
        model="mock-fast",
        prompt="Какие ресурсы нужны для входящего SIP-трафика?",
        rag_mode="on",
    )

    assert response.grounded is True
    assert response.unknown_reason is None
    assert response.rag_sources
    assert response.rag_sources[0].chunk_id == chunk.source.chunk_id
    assert response.citations
    assert _field(response.citations[0], "chunk_id") == chunk.source.chunk_id
    assert _field(response.citations[0], "quote") in chunk.text
    assert len(provider.requests) == 1
    assert rag.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("chunks", [[], [_chunk()]])
async def test_empty_or_weak_context_returns_unknown_without_provider_call(
    chunks: list[RetrievedChunk],
) -> None:
    provider = MockProvider()
    rag = RetrievalDouble(chunks=chunks, relevant=False)

    response = await _service(provider, rag).chat(
        model="mock-fast",
        prompt="Какая неизвестная настройка нужна для SIP?",
        rag_mode="on",
    )

    assert response.grounded is False
    assert response.citations == []
    assert response.rag_sources == []
    assert response.unknown_reason
    assert "не" in response.answer.lower()
    assert provider.requests == []
    assert rag.calls == 1


@pytest.mark.asyncio
async def test_citation_must_be_a_substring_of_retrieved_chunk() -> None:
    chunk = _chunk()
    provider = CitationProvider(
        {
            "chunk_id": chunk.source.chunk_id,
            "quote": "This sentence is not in the retrieved chunk.",
        }
    )
    rag = RetrievalDouble(chunks=[chunk], relevant=True)

    response = await _service(provider, rag).chat(
        model="mock-fast",
        prompt="Какие ресурсы нужны для входящего SIP-трафика?",
        rag_mode="on",
    )

    invalid_citations = [
        citation
        for citation in response.citations
        if _field(citation, "quote") not in chunk.text
    ]
    assert invalid_citations == []


@pytest.mark.asyncio
async def test_answer_cannot_reference_chunk_outside_current_results() -> None:
    rag = RetrievalDouble(chunks=[_chunk()], relevant=True)

    response = await _service(ForeignChunkProvider(), rag).chat(
        model="mock-fast",
        prompt="Что делать дальше?",
        rag_mode="enhanced",
    )

    assert response.unknown is True
    assert response.grounded is False
    assert response.rag_sources == []
    assert response.citations == []
    assert "вне текущего RAG-контекста" in (response.unknown_reason or "")


@pytest.mark.asyncio
async def test_off_mode_remains_compatible_and_does_not_call_rag() -> None:
    provider = MockProvider()
    rag = RetrievalDouble(chunks=[_chunk()], relevant=True)

    response = await _service(provider, rag).chat(
        model="mock-fast",
        prompt="Привет",
        rag_mode="off",
    )

    assert response.rag_mode == "off"
    assert response.citations == []
    assert response.grounded is False
    assert response.unknown_reason is None
    assert response.answer == "[mock:mock-fast] Привет"
    assert len(provider.requests) == 1
    assert rag.calls == 0
