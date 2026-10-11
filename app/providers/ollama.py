"""Ollama adapter through its OpenAI-compatible chat API."""

from __future__ import annotations

from time import perf_counter

from openai import AsyncOpenAI

from app.schemas.llm import (
    LLMRequest,
    LLMResponse,
    ToolCompletionRequest,
    ToolCompletionResponse,
)

from ._openai_compat import (
    build_tool_completion_payload,
    parse_completion,
    parse_tool_completion,
    translate_openai_error,
)

PROVIDER_LABEL = "Ollama"


class OllamaProvider:
    name = "ollama"

    def __init__(
        self,
        base_url: str,
        timeout_s: float,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self._base_url = f"{base_url.rstrip('/')}/v1"
        self._timeout_s = timeout_s
        self._client = client

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key="ollama",
                base_url=self._base_url,
                timeout=self._timeout_s,
                max_retries=0,
            )
        return self._client

    async def chat(self, request: LLMRequest) -> LLMResponse:
        client = self._get_client()
        started = perf_counter()

        try:
            completion = await client.chat.completions.create(
                model=request.model,
                messages=[{"role": "user", "content": request.prompt}],
            )
        except Exception as exc:
            raise translate_openai_error(exc, PROVIDER_LABEL) from exc

        answer, usage, finish_reason = parse_completion(completion, PROVIDER_LABEL)
        return LLMResponse(
            provider=self.name,
            model=request.model,
            answer=answer,
            usage=usage,
            latency_ms=(perf_counter() - started) * 1000,
            finish_reason=finish_reason,
        )

    async def complete(self, request: ToolCompletionRequest) -> ToolCompletionResponse:
        client = self._get_client()
        started = perf_counter()

        try:
            completion = await client.chat.completions.create(
                model=request.model,
                **build_tool_completion_payload(request),
            )
        except Exception as exc:
            raise translate_openai_error(exc, PROVIDER_LABEL) from exc

        return parse_tool_completion(
            completion=completion,
            provider_name=self.name,
            provider_label=PROVIDER_LABEL,
            model=request.model,
            latency_ms=(perf_counter() - started) * 1000,
        )
