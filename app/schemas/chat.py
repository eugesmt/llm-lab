"""API schemas for the chat endpoint."""

from pydantic import BaseModel, Field

from app.schemas.llm import LLMResponse

# The chat response is exactly our internal LLMResponse: provider-independent
# by design, so no separate projection model is needed.
ChatResponse = LLMResponse


class ChatRequest(BaseModel):
    model: str = Field(min_length=1, examples=["deepseek-v4-flash"])
    prompt: str = Field(min_length=1)
    reasoning_effort: str | None = Field(default=None, examples=["low"])
