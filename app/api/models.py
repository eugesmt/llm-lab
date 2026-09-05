"""GET /api/models — the single source of model information for the frontend."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from app.dependencies import get_llm_service
from app.schemas.model import ModelInfo
from app.services.llm import LLMService

router = APIRouter(prefix="/api", tags=["models"])


@router.get("/models", response_model=list[ModelInfo])
def list_models(service: Annotated[LLMService, Depends(get_llm_service)]) -> list[ModelInfo]:
    return service.list_models()
