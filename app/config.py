"""Application configuration via Pydantic Settings.

API keys come only from environment variables (optionally a local ``.env``
file, which is gitignored) and are stored as ``SecretStr`` so they never leak
into logs or repr output.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    deepseek_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    openrouter_api_key: SecretStr | None = None
    # Reserved for a future Anthropic provider adapter; not used yet.
    anthropic_api_key: SecretStr | None = None

    # Single timeout for the whole chain: HTTP request -> application context
    # -> provider call (provider SDK clients receive the same value).
    request_timeout_s: float = 120.0

    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_embedding_model: str = "qwen3-embedding:0.6b"
    rag_index_dir: str = "/home/lab-dev/Projects/rag/rag_lab/indexes/structural"
    rag_initial_top_k: int = Field(default=8, ge=3, le=50)
    rag_top_k: int = Field(default=3, ge=1, le=20)
    # Candidate filter for enhanced RAG: chunks below this cosine similarity
    # are removed before heuristic reranking. It is not the answer threshold.
    rag_similarity_threshold: float = Field(default=0.55, ge=-1.0, le=1.0)
    # Context sufficiency threshold: the best final score must reach this value
    # for the retrieved context to be considered answer-worthy. Enhanced RAG
    # uses rerank_score; regular RAG uses cosine similarity. Empty context is
    # always insufficient. This value is intentionally separate from the filter.
    rag_context_threshold: float = Field(default=0.60, ge=-1.0, le=1.0)

    # Chat history is stored separately from the document index database.
    chat_db_path: Path = Path("data/chat.sqlite")

    taskmanager_mcp_url: str = "http://127.0.0.1:8090/mcp"
    mcp_timeout_s: float = 30.0
    mcp_max_rounds: int = Field(default=10, ge=2, le=10)

    obsidian_mcp_url: str | None = None
    obsidian_mcp_api_key: SecretStr | None = None

    @model_validator(mode="after")
    def validate_obsidian_mcp_settings(self) -> Settings:
        if (self.obsidian_mcp_url is None) != (self.obsidian_mcp_api_key is None):
            raise ValueError("obsidian_mcp_url and obsidian_mcp_api_key must be set together")
        return self

    host: str = "127.0.0.1"
    port: int = 8000
