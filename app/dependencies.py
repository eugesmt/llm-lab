"""FastAPI dependencies wiring routers to services stored on app.state."""

from __future__ import annotations

from fastapi import Request

from app.services.benchmark import BenchmarkService
from app.services.llm import LLMService


def get_llm_service(request: Request) -> LLMService:
    return request.app.state.llm_service


def get_benchmark_service(request: Request) -> BenchmarkService:
    return request.app.state.benchmark_service
