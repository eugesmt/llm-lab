"""OpenRouter provider adapter.

OpenRouter aggregates many providers behind an OpenAI-compatible API, so it
consumes the same ``openai`` SDK with a custom base URL. Model IDs follow the
``vendor/model`` convention (e.g. ``inclusionai/ling-3.0-flash-sante:free``).
Free OpenRouter models have a rate limit instead of a price, so their catalog
entries carry zero pricing.
"""

from __future__ import annotations

from time import perf_counter

from openai import AsyncOpenAI

from app.errors import ProviderNotConfiguredError
from app.schemas.llm import LLMRequest, LLMResponse

from ._openai_compat import parse_completion, translate_openai_error

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
PROVIDER_LABEL = "OpenRouter"


class OpenRouterProvider:
    name = "openrouter"

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
                raise ProviderNotConfiguredError("OPENROUTER_API_KEY is not set")
            self._client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=OPENROUTER_BASE_URL,
                timeout=self._timeout_s,
                max_retries=0,  # SDK retries would distort latency measurements
            )
        return self._client

    async def chat(self, request: LLMRequest) -> LLMResponse:
        client = self._get_client()

        extra_kwargs: dict[str, str] = {}
        if request.reasoning_effort is not None and request.reasoning_effort != "none":
            # OpenRouter forwards reasoning_effort to models that support it;
            # "none" means "no thinking", which is the default for models
            # without reasoning, so it is simply not sent.
            extra_kwargs["reasoning_effort"] = request.reasoning_effort

        start = perf_counter()
        try:
            completion = await client.chat.completions.create(
                model=request.model,
                messages=[{"role": "user", "content": request.prompt}],
                **extra_kwargs,
            )
        except Exception as exc:
            raise translate_openai_error(exc, PROVIDER_LABEL) from exc
        latency_ms = (perf_counter() - start) * 1000

        answer, usage, finish_reason = parse_completion(completion, PROVIDER_LABEL)
        return LLMResponse(
            provider=self.name,
            model=request.model,
            answer=answer,
            usage=usage,
            latency_ms=latency_ms,
            finish_reason=finish_reason,
        )
