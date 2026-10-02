"""Internal LLM domain models shared across the application.

These models are our own and intentionally independent of any provider SDK:
adapters convert provider-specific payloads into them and back.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


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


class ToolDefinition(BaseModel):
    name: str
    description: str | None = None
    input_schema: dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class LLMMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    reasoning_content: str | None = None


class ToolCompletionRequest(BaseModel):
    model: str
    messages: list[LLMMessage]
    tools: list[ToolDefinition]
    reasoning_effort: str | None = None


class ToolCompletionResponse(BaseModel):
    provider: str
    model: str
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    reasoning_content: str | None = None
    usage: Usage | None = None
    latency_ms: float
    finish_reason: str | None = None


class ArtifactLink(BaseModel):
    download_url: str
    filename: str | None = None
    report_id: str | None = None


class RAGSource(BaseModel):
    chunk_id: str
    source: str
    section: str | None = None
    similarity: float
    rerank_score: float | None = None


class RAGCitation(BaseModel):
    source: str
    section: str | None = None
    chunk_id: str
    quote: str


class MCPTraceEntry(BaseModel):
    sequence: int
    round: int
    server: str
    tool: str
    alias: str
    status: Literal["success"] = "success"


class LLMResponse(BaseModel):
    provider: str
    model: str
    answer: str
    usage: Usage | None = None
    latency_ms: float
    finish_reason: str | None = None
    mcp_tools_used: list[str] = Field(default_factory=list)
    artifacts: list[ArtifactLink] = Field(default_factory=list)
    mcp_trace: list[MCPTraceEntry] = Field(default_factory=list)
    rag_mode: Literal["off", "on", "enhanced"] = "off"
    rag_sources: list[RAGSource] = Field(default_factory=list)
    citations: list[RAGCitation] = Field(default_factory=list)
    grounded: bool = False
    unknown: bool = False
    unknown_reason: str | None = None
