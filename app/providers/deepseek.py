"""DeepSeek provider adapter.

The only module in the application that knows DeepSeek specifics. DeepSeek's
API is OpenAI-compatible, so it is consumed through the official ``openai`` SDK
with a custom base URL (the integration path recommended by DeepSeek docs).
"""

from __future__ import annotations

from time import perf_counter

from openai import AsyncOpenAI

from app.errors import ProviderNotConfiguredError
from app.schemas.llm import LLMRequest, LLMResponse, ToolCompletionRequest, ToolCompletionResponse

from ._openai_compat import (
    build_tool_completion_payload,
    parse_completion,
    parse_tool_completion,
    translate_openai_error,
)

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
PROVIDER_LABEL = "DeepSeek"


class DeepSeekProvider:
    name = "deepseek"

    def __init__(
        self,
        api_key: str | None,
        timeout_s: float,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self._api_key = api_key
        self._timeout_s = timeout_s
        self._client = client

    def _get_client(self) -> AsyncOpenAI:
        # Created lazily so a missing API key surfaces as a clear application
        # error on first call instead of at import/startup time.
        if self._client is None:
            if not self._api_key:
                raise ProviderNotConfiguredError("DEEPSEEK_API_KEY is not set")
            self._client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=DEEPSEEK_BASE_URL,
                timeout=self._timeout_s,
                max_retries=0,  # SDK retries would distort latency measurements
            )
        return self._client

    @staticmethod
    def _reasoning_kwargs(reasoning_effort: str | None) -> dict[str, object]:
        if reasoning_effort == "none":
            return {"extra_body": {"thinking": {"type": "disabled"}}}
        if reasoning_effort is not None:
            return {"reasoning_effort": reasoning_effort}
        return {}

    async def chat(self, request: LLMRequest) -> LLMResponse:
        client = self._get_client()

        extra_kwargs = self._reasoning_kwargs(request.reasoning_effort)

        start = perf_counter()
        try:
            completion = await client.chat.completions.create(
                model=request.model,
                messages=[{"role": "user", "content": request.prompt}],
                **extra_kwargs,
            )
        except Exception as exc:
            raise translate_openai_error(exc, "DeepSeek") from exc
        latency_ms = (perf_counter() - start) * 1000

        answer, usage, finish_reason = parse_completion(completion, "DeepSeek")
        return LLMResponse(
            provider=self.name,
            model=request.model,
            answer=answer,
            usage=usage,
            latency_ms=latency_ms,
            finish_reason=finish_reason,
        )

    async def complete(self, request: ToolCompletionRequest) -> ToolCompletionResponse:
        client = self._get_client()
        extra_kwargs = build_tool_completion_payload(request)
        extra_kwargs.update(self._reasoning_kwargs(request.reasoning_effort))

        start = perf_counter()
        try:
            completion = await client.chat.completions.create(
                model=request.model,
                **extra_kwargs,
            )
        except Exception as exc:
            raise translate_openai_error(exc, PROVIDER_LABEL) from exc

        return parse_tool_completion(
            completion=completion,
            provider_name=self.name,
            provider_label=PROVIDER_LABEL,
            model=request.model,
            latency_ms=(perf_counter() - start) * 1000,
        )
