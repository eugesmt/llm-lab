"""API schemas for chat and conversation endpoints."""

from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.conversation import ChatMessage, Conversation, TaskState
from app.schemas.llm import LLMResponse


class ChatResponse(LLMResponse):
    """LLM response with optional conversation state."""

    conversation_id: str | None = None
    task_state: TaskState | None = None


class ConversationResponse(BaseModel):
    """Complete persisted state returned by a conversation lookup."""

    conversation: Conversation
    messages: list[ChatMessage]
    task_state: TaskState


class ChatRequest(BaseModel):
    model: str = Field(min_length=1, examples=["deepseek-v4-flash"])
    prompt: str = Field(min_length=1)
    reasoning_effort: str | None = Field(default=None, examples=["low"])
    use_taskmanager_mcp: bool = False
    mcp_servers: list[str] | None = None
    rag_mode: Literal["off", "on", "enhanced"] = "off"
    conversation_id: str | None = Field(default=None, min_length=1)
