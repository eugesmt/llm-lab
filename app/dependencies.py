"""FastAPI dependencies wiring routers to services stored on app.state."""

from __future__ import annotations

from fastapi import Request

from app.services.benchmark import BenchmarkService
from app.services.llm import LLMService
from app.services.memory import TaskMemoryService
from app.storage import ChatStore


def get_llm_service(request: Request) -> LLMService:
    return request.app.state.llm_service


def get_benchmark_service(request: Request) -> BenchmarkService:
    return request.app.state.benchmark_service


def get_chat_store(request: Request) -> ChatStore:
    return request.app.state.chat_store


def get_task_memory_service(request: Request) -> TaskMemoryService:
    return request.app.state.task_memory_service
