"""OpenAI provider adapter.

The only module in the application that knows OpenAI specifics (beyond the
OpenAI-compatible helpers shared with DeepSeek inside ``app.providers``).
"""

from __future__ import annotations

from time import perf_counter

from openai import AsyncOpenAI

from app.errors import ProviderNotConfiguredError
from app.schemas.llm import LLMRequest, LLMResponse

from ._openai_compat import parse_completion, translate_openai_error

PROVIDER_LABEL = "OpenAI"


class OpenAIProvider:
    name = "openai"

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
        if self._client is None:
            if not self._api_key:
                raise ProviderNotConfiguredError("OPENAI_API_KEY is not set")
            self._client = AsyncOpenAI(
                api_key=self._api_key,
                timeout=self._timeout_s,
                max_retries=0,  # SDK retries would distort latency measurements
            )
        return self._client

    async def chat(self, request: LLMRequest) -> LLMResponse:
        client = self._get_client()

        extra_kwargs: dict[str, str] = {}
        if request.reasoning_effort is not None:
            # Reasoning models accept reasoning_effort (none/low/medium/high/
            # xhigh/max depending on the model); non-reasoning models must not
            # receive it — the service layer already validated support.
            extra_kwargs["reasoning_effort"] = request.reasoning_effort

        start = perf_counter()
        try:
            completion = await client.chat.completions.create(
                model=request.model,
                messages=[{"role": "user", "content": request.prompt}],
                **extra_kwargs,
            )
        except Exception as exc:
            raise translate_openai_error(exc, "OpenAI") from exc
        latency_ms = (perf_counter() - start) * 1000

        answer, usage, finish_reason = parse_completion(completion, "OpenAI")
        return LLMResponse(
            provider=self.name,
            model=request.model,
            answer=answer,
            usage=usage,
            latency_ms=latency_ms,
            finish_reason=finish_reason,
        )
