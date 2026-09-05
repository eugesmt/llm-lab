"""POST /api/benchmark — run one prompt against several models."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from app.dependencies import get_benchmark_service
from app.schemas.benchmark import BenchmarkRequest, BenchmarkResponse
from app.services.benchmark import BenchmarkService

router = APIRouter(prefix="/api", tags=["benchmark"])


@router.post("/benchmark", response_model=BenchmarkResponse)
async def benchmark(
    request: BenchmarkRequest,
    service: Annotated[BenchmarkService, Depends(get_benchmark_service)],
) -> BenchmarkResponse:
    return await service.run(
        prompt=request.prompt,
        models=request.models,
        reasoning_effort=request.reasoning_effort,
        repeats=request.repeats,
    )
