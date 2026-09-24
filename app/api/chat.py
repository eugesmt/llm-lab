"""POST /api/chat — send one prompt to one model."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from app.dependencies import get_llm_service
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.llm import LLMService

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    service: Annotated[LLMService, Depends(get_llm_service)],
) -> ChatResponse:
    return await service.chat(
        model=request.model,
        prompt=request.prompt,
        reasoning_effort=request.reasoning_effort,
        use_taskmanager_mcp=request.use_taskmanager_mcp,
    )
