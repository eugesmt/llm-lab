"""API schemas for the benchmark endpoint."""

from pydantic import BaseModel, Field

from app.schemas.llm import Cost, LLMResponse


class BenchmarkRequest(BaseModel):
    prompt: str = Field(min_length=1)
    models: list[str] = Field(min_length=1)
    reasoning_effort: str | None = Field(default=None, examples=["low"])
    repeats: int = Field(default=1, ge=1, le=10)


class BenchmarkResult(LLMResponse):
    cost: Cost | None = None
    error: str | None = None


class BenchmarkResponse(BaseModel):
    prompt: str
    reasoning_effort: str | None
    repeats: int
    results: list[BenchmarkResult]
