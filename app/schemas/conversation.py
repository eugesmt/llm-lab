"""Domain models for conversational history and task memory."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    """Return the current UTC time as an aware datetime."""

    return datetime.now(UTC)


class Conversation(BaseModel):
    """A persisted conversation identified by a UUID string."""

    id: str
    created_at: datetime
    updated_at: datetime


class ChatMessage(BaseModel):
    """One user or assistant message in a conversation."""

    id: str
    conversation_id: str
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime
    response_json: dict[str, Any] | None = None


class TaskState(BaseModel):
    """Compact memory of the goal and constraints of a conversation."""

    goal: str | None = None
    constraints: list[str] = Field(default_factory=list)
    terms: dict[str, str] = Field(default_factory=dict)
    clarifications: list[str] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=utc_now)
