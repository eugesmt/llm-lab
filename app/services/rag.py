"""Runtime retrieval over the locally built FAISS + SQLite index."""

from __future__ import annotations

import json
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
    ) -> None:
        self._index_path = Path(index_dir) / "index.faiss"
        self._database_path = Path(index_dir) / "chunks.sqlite"
        self._embedding_url = f"{ollama_base_url.rstrip('/')}/api/embed"
        self._embedding_model = embedding_model
        self._top_k = top_k
        self._index: object | None = None

    async def retrieve(self, question: str) -> list[RetrievedChunk]:
        index = self._load_index()
        question_vector = await self._embed(question)
        import faiss
        import numpy as np

        vector = np.asarray([question_vector], dtype="float32")
        faiss.normalize_L2(vector)
        similarities, vector_ids = index.search(vector, self._top_k)

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

                results.append(
                    RetrievedChunk(
                        source=RAGSource(
                            chunk_id=row[0],
                            source=row[1],
                            section=row[2],
                            similarity=float(similarity),
                        ),
                        text=row[3],
                    )
                )

        return results

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
