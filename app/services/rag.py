"""Runtime retrieval over the locally built FAISS + SQLite index."""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import httpx

from app.errors import ProviderRequestError, ProviderUnavailableError
from app.schemas.llm import RAGSource


@dataclass(frozen=True)
class RetrievedChunk:
    source: RAGSource
    text: str


class RAGService:
    def __init__(
        self,
        index_dir: str,
        ollama_base_url: str,
        embedding_model: str,
        top_k: int,
        initial_top_k: int,
        similarity_threshold: float,
        context_threshold: float,
    ) -> None:
        self._index_path = Path(index_dir) / "index.faiss"
        self._database_path = Path(index_dir) / "chunks.sqlite"
        self._embedding_url = f"{ollama_base_url.rstrip('/')}/api/embed"
        self._embedding_model = embedding_model
        self._top_k = top_k
        self._initial_top_k = initial_top_k
        self._similarity_threshold = similarity_threshold
        self._context_threshold = context_threshold
        self._index: object | None = None

    async def retrieve(self, question: str, enhanced: bool = False) -> list[RetrievedChunk]:
        index = self._load_index()
        search_question = self.rewrite_query(question) if enhanced else question
        question_vector = await self._embed(search_question)
        import faiss
        import numpy as np

        vector = np.asarray([question_vector], dtype="float32")
        faiss.normalize_L2(vector)
        search_k = self._initial_top_k if enhanced else self._top_k
        similarities, vector_ids = index.search(vector, search_k)

        results: list[RetrievedChunk] = []
        with sqlite3.connect(self._database_path) as connection:
            for similarity, vector_id in zip(similarities[0], vector_ids[0], strict=True):
                if vector_id == -1:
                    continue

                row = connection.execute(
                    """
                    SELECT chunk_id, source, section, text
                    FROM chunks
                    WHERE vector_id = ?
                    """,
                    (int(vector_id),),
                ).fetchone()
                if row is None:
                    raise ProviderRequestError(
                        f"RAG metadata is missing for vector_id={int(vector_id)}"
                    )

                results.append(self._make_chunk(row, float(similarity)))

        if not enhanced:
            return results

        filtered = [
            chunk
            for chunk in results
            if chunk.source.similarity >= self._similarity_threshold
        ]

        query_terms = self._terms(search_question)
        reranked = sorted(
            filtered,
            key=lambda chunk: self._rerank_score(chunk, query_terms),
            reverse=True,
        )
        return [
            RetrievedChunk(
                source=chunk.source.model_copy(
                    update={"rerank_score": self._rerank_score(chunk, query_terms)}
                ),
                text=chunk.text,
            )
            for chunk in reranked[: self._top_k]
        ]

    def context_is_relevant(self, chunks: list[RetrievedChunk]) -> bool:
        """Return whether the best retrieved chunk clears the answer threshold.

        Candidate filtering and context sufficiency are deliberately separate:
        ``_similarity_threshold`` controls which enhanced-search candidates stay
        in the pool, while ``_context_threshold`` decides whether the resulting
        context is strong enough to support an answer.  Enhanced results use
        their computed rerank score; regular results use FAISS cosine similarity.
        An empty result is never considered sufficient.
        """
        if not chunks:
            return False

        best_score = max(self._context_score(chunk) for chunk in chunks)
        return best_score >= self._context_threshold

    @staticmethod
    def _context_score(chunk: RetrievedChunk) -> float:
        """Use the score that actually determined the final result ordering."""
        return (
            chunk.source.rerank_score
            if chunk.source.rerank_score is not None
            else chunk.source.similarity
        )

    @staticmethod
    def rewrite_query(question: str) -> str:
        """Добавляет к вопросу технические термины для более точного поиска."""
        lower_question = question.lower()
        expansions: list[str] = []
        mappings = {
            "входящ": "inbound SIP trunk SIP dispatch rule CreateSIPTrunk CreateSIPDispatchRule",
            "docker": "Docker host.docker.internal 172.17.0.1 Redis",
            "зависим": "Go libopus pkg-config opusfile libsoxr",
            "запис": "Egress room audio video output format",
            "агент": "AgentSession voice agent lifecycle room tools",
            "комнат": "LiveKit room WebSocket token identity",
            "redis": "Redis session state LiveKit server",
            "webhook": "Webhook event URL secret",
            "токен": "access token room identity grant",
        }
        for marker, terms in mappings.items():
            if marker in lower_question:
                expansions.append(terms)
        return f"{question} {' '.join(expansions)}".strip()

    @staticmethod
    def _terms(text: str) -> set[str]:
        stopwords = {
            "как", "что", "для", "нужен", "нужно", "какие", "какой", "из", "в", "и",
            "the", "how", "what", "for", "from", "with", "are", "is", "to",
        }
        return {
            term
            for term in re.findall(r"[a-zа-яё0-9_]+", text.lower())
            if len(term) > 2 and term not in stopwords
        }

    def _rerank_score(self, chunk: RetrievedChunk, query_terms: set[str]) -> float:
        chunk_terms = self._terms(f"{chunk.source.section or ''} {chunk.text}")
        overlap = len(query_terms & chunk_terms) / max(len(query_terms), 1)
        return 0.75 * chunk.source.similarity + 0.25 * overlap

    @staticmethod
    def _make_chunk(row: tuple[object, ...], similarity: float) -> RetrievedChunk:
        return RetrievedChunk(
            source=RAGSource(
                chunk_id=str(row[0]),
                source=str(row[1]),
                section=str(row[2]) if row[2] is not None else None,
                similarity=similarity,
            ),
            text=str(row[3]),
        )

    def _load_index(self) -> object:
        if self._index is None:
            if not self._index_path.exists() or not self._database_path.exists():
                raise ProviderUnavailableError(
                    f"RAG index is missing: {self._index_path}"
                )
            import faiss

            self._index = faiss.read_index(str(self._index_path))
        return self._index

    async def _embed(self, text: str) -> list[float]:
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    self._embedding_url,
                    json={
                        "model": self._embedding_model,
                        "input": text,
                        "keep_alive": 0,
                    },
                )
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError("Ollama embedding endpoint is unavailable") from exc

        try:
            payload = response.json()
            return payload["embeddings"][0]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise ProviderRequestError("Ollama returned an invalid embedding response") from exc
