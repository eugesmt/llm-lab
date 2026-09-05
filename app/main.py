"""Application factory and entry point.

``create_app`` wires providers (protocol implementations), services and HTTP
error handlers together; tests can inject a custom catalog or providers
(e.g. MockProvider) through its parameters.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

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
from app.providers.base import LLMProvider
from app.providers.deepseek import DeepSeekProvider
from app.providers.openai import OpenAIProvider
from app.schemas.model import ModelInfo
from app.services.benchmark import BenchmarkService
from app.services.llm import LLMService

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


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
        }

    llm_service = LLMService(
        providers=providers,
        catalog=catalog,
        request_timeout_s=settings.request_timeout_s,
    )

    app = FastAPI(title="LLM Lab", version="0.1.0")
    app.state.llm_service = llm_service
    app.state.benchmark_service = BenchmarkService(llm_service)

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
