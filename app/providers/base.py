"""Minimal provider interface.

This is the only thing application code (LLMService, benchmark, API) depends
on — never a concrete SDK. Adapters implementing this protocol must:

* convert our ``LLMRequest`` into a provider-specific call,
* measure wall-clock latency of that call and fill ``LLMResponse.latency_ms``,
* translate provider/SDK exceptions into the normalized errors from
  ``app.errors`` (``asyncio.CancelledError`` must propagate unchanged so that
  parent-context cancellation works).
"""

from __future__ import annotations

from typing import Protocol

from app.schemas.llm import LLMRequest, LLMResponse


class LLMProvider(Protocol):
    """Common interface every provider adapter implements."""

    name: str

    async def chat(self, request: LLMRequest) -> LLMResponse: ...
