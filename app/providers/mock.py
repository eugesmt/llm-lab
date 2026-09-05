"""Deterministic in-process provider used by tests (and manual experiments).

Implements the same protocol as real adapters but never touches the network.
"""

from __future__ import annotations

import asyncio
from time import perf_counter

from app.errors import ProviderRequestError
from app.schemas.llm import LLMRequest, LLMResponse, Usage


class MockProvider:
    name = "mock"

    def __init__(
        self,
        delay_s: float = 0.01,
        fail_models: set[str] | None = None,
    ) -> None:
        self._delay_s = delay_s
        self._fail_models = fail_models or set()
        self.requests: list[LLMRequest] = []

    async def chat(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if request.model in self._fail_models:
            raise ProviderRequestError(f"mock failure for model {request.model}")

        start = perf_counter()
        await asyncio.sleep(self._delay_s)
        latency_ms = (perf_counter() - start) * 1000

        prompt_tokens = max(1, len(request.prompt.split()))
        answer = f"[mock:{request.model}] {request.prompt}"
        completion_tokens = max(1, len(answer.split()))
        return LLMResponse(
            provider=self.name,
            model=request.model,
            answer=answer,
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                reasoning_tokens=(3 if request.reasoning_effort not in (None, "none") else None),
            ),
            latency_ms=latency_ms,
            finish_reason="stop",
        )
