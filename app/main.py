"""Application factory and entry point.

``create_app`` wires providers (protocol implementations), services and HTTP
error handlers together; tests can inject a custom catalog or providers
(e.g. MockProvider) through its parameters.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path
from threading import RLock

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app import catalog as catalog_module
from app.api import benchmark as benchmark_api
from app.api import chat as chat_api
from app.api import models as models_api
from app.config import Settings
from app.errors import AppError
from app.mcp.client import MCPClient, StreamableHTTPMCPClient, TaskManagerMCPClient
from app.providers.base import LLMProvider
from app.providers.deepseek import DeepSeekProvider
from app.providers.ollama import OllamaProvider
from app.providers.openai import OpenAIProvider
from app.providers.openrouter import OpenRouterProvider
from app.schemas.model import ModelInfo
from app.services.benchmark import BenchmarkService
from app.services.llm import LLMService
from app.services.memory import TaskMemoryService
from app.services.rag import RAGService
from app.storage import ChatStore

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


def _prepare_chat_store_for_web(store: ChatStore) -> ChatStore:
    """Make the synchronous store safe for FastAPI's worker/event-loop thread.

    ``ChatStore`` intentionally stays a small synchronous adapter.  FastAPI's
    TestClient and some ASGI servers can invoke the app on a thread different
    from the one that constructed the app, while sqlite3 defaults to strict
    thread affinity.  Rebind the already-initialized connection here without
    changing the storage component's public contract.
    """

    connection = getattr(store, "_connection", None)
    if connection is None:
        return store

    database = ":memory:" if str(store.path) == ":memory:" else str(store.path)
    replacement = sqlite3.connect(database, check_same_thread=False)
    if database == ":memory:":
        connection.backup(replacement)
    replacement.row_factory = sqlite3.Row
    replacement.execute("PRAGMA foreign_keys = ON")
    if database != ":memory:":
        replacement.execute("PRAGMA journal_mode = WAL")
    store._connection = replacement
    store._lock = RLock()
    connection.close()
    return store


def _error_response(exc: AppError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": type(exc).__name__, "message": exc.message}},
    )


def _register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        return _error_response(exc)

    @app.exception_handler(Exception)
    async def handle_unexpected_error(_: Request, exc: Exception) -> JSONResponse:
        # Never leak a traceback to the client.
        return JSONResponse(
            status_code=500, content={"error": {"message": "Internal server error"}}
        )


def create_app(
    settings: Settings | None = None,
    catalog: Sequence[ModelInfo] | None = None,
    providers: Mapping[str, LLMProvider] | None = None,
    taskmanager_mcp: MCPClient | None = None,
    mcp_servers: Mapping[str, MCPClient] | None = None,
    chat_store: ChatStore | None = None,
) -> FastAPI:
    settings = settings or Settings()
    catalog = catalog if catalog is not None else catalog_module.MODEL_CATALOG

    if providers is None:
        providers = {
            "deepseek": DeepSeekProvider(
                api_key=settings.deepseek_api_key.get_secret_value()
                if settings.deepseek_api_key
                else None,
                timeout_s=settings.request_timeout_s,
            ),
            "openai": OpenAIProvider(
                api_key=settings.openai_api_key.get_secret_value()
                if settings.openai_api_key
                else None,
                timeout_s=settings.request_timeout_s,
            ),
            "openrouter": OpenRouterProvider(
                api_key=settings.openrouter_api_key.get_secret_value()
                if settings.openrouter_api_key
                else None,
                timeout_s=settings.request_timeout_s,
            ),
            "ollama": OllamaProvider(
                base_url=settings.ollama_base_url,
                timeout_s=settings.request_timeout_s,
            ),
        }

    configured_servers = dict(mcp_servers or {})
    if taskmanager_mcp is None:
        taskmanager_mcp = configured_servers.get("taskmanager")
    if taskmanager_mcp is None:
        taskmanager_mcp = TaskManagerMCPClient(
            url=settings.taskmanager_mcp_url,
            timeout_s=settings.mcp_timeout_s,
        )
    configured_servers["taskmanager"] = taskmanager_mcp

    if (
        "obsidian" not in configured_servers
        and settings.obsidian_mcp_url is not None
        and settings.obsidian_mcp_api_key is not None
    ):
        configured_servers["obsidian"] = StreamableHTTPMCPClient(
            url=settings.obsidian_mcp_url,
            timeout_s=settings.mcp_timeout_s,
            headers={
                "Authorization": f"Bearer {settings.obsidian_mcp_api_key.get_secret_value()}"
            },
            server_name="Obsidian",
        )

    rag_service = RAGService(
        index_dir=settings.rag_index_dir,
        ollama_base_url=settings.ollama_base_url,
        embedding_model=settings.ollama_embedding_model,
        top_k=settings.rag_top_k,
        initial_top_k=settings.rag_initial_top_k,
        similarity_threshold=settings.rag_similarity_threshold,
        context_threshold=settings.rag_context_threshold,
    )

    llm_service = LLMService(
        providers=providers,
        catalog=catalog,
        request_timeout_s=settings.request_timeout_s,
        taskmanager_mcp=taskmanager_mcp,
        mcp_servers=configured_servers,
        mcp_max_rounds=settings.mcp_max_rounds,
        rag_service=rag_service,
    )

    chat_store = _prepare_chat_store_for_web(chat_store or ChatStore(settings.chat_db_path))
    task_memory_service = TaskMemoryService()

    app = FastAPI(title="LLM Lab", version="0.1.0")
    app.state.llm_service = llm_service
    app.state.benchmark_service = BenchmarkService(llm_service)
    app.state.mcp_servers = configured_servers
    app.state.chat_store = chat_store
    app.state.task_memory_service = task_memory_service

    _register_error_handlers(app)
    app.include_router(models_api.router)
    app.include_router(chat_api.router)
    app.include_router(benchmark_api.router)

    # Mounted last so /api, /docs and /openapi.json keep priority.
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
    return app


def main() -> None:
    settings = Settings()
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
    )


if __name__ == "__main__":
    main()
