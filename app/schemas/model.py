"""Model catalog entry: what the frontend needs to know about a model."""

from pydantic import BaseModel


class ModelInfo(BaseModel):
    id: str
    provider: str
    display_name: str
    thinking_supported: bool
    reasoning_effort: list[str]
