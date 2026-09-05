"""Internal LLM domain models shared across the application.

These models are our own and intentionally independent of any provider SDK:
adapters convert provider-specific payloads into them and back.
"""

from pydantic import BaseModel


class Usage(BaseModel):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None


class Cost(BaseModel):
    input_cost: float | None = None
    output_cost: float | None = None
    total_cost: float | None = None


class LLMRequest(BaseModel):
    model: str
    prompt: str
    reasoning_effort: str | None = None


class LLMResponse(BaseModel):
    provider: str
    model: str
    answer: str
    usage: Usage | None = None
    latency_ms: float
    finish_reason: str | None = None
