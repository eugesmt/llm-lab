"""LLM orchestration: catalog lookup, request validation, timeout, dispatch.

The service depends only on the provider protocol and the model catalog —
never on a concrete SDK (see app/providers/base.py).
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from app.errors import (
    EmptyResponseError,
    ModelNotFoundError,
    ProviderNotConfiguredError,
    ProviderTimeoutError,
    UnsupportedReasoningEffortError,
)
from app.schemas.llm import LLMRequest, LLMResponse
from app.schemas.model import ModelInfo

if TYPE_CHECKING:
    from app.providers.base import LLMProvider


class LLMService:
    def __init__(
        self,
        providers: Mapping[str, LLMProvider],
        catalog: Sequence[ModelInfo],
        request_timeout_s: float,
    ) -> None:
        self._providers = providers
        self._models: dict[str, ModelInfo] = {info.id: info for info in catalog}
        self._request_timeout_s = request_timeout_s

    def list_models(self) -> list[ModelInfo]:
        return list(self._models.values())

    def get_model(self, model_id: str) -> ModelInfo:
        info = self._models.get(model_id)
        if info is None:
            supported = ", ".join(sorted(self._models)) or "none"
            raise ModelNotFoundError(f"Unknown model '{model_id}'. Available models: {supported}")
        return info

    def _resolve(self, model_id: str, reasoning_effort: str | None) -> ModelInfo:
        info = self.get_model(model_id)
        if reasoning_effort is not None and reasoning_effort not in info.reasoning_effort:
            supported = ", ".join(info.reasoning_effort) or "none"
            raise UnsupportedReasoningEffortError(
                f"Model '{model_id}' does not support reasoning effort "
                f"'{reasoning_effort}'. Supported values: {supported}"
            )
        return info

    async def chat(
        self,
        model: str,
        prompt: str,
        reasoning_effort: str | None = None,
    ) -> LLMResponse:
        info = self._resolve(model, reasoning_effort)

        provider = self._providers.get(info.provider)
        if provider is None:
            raise ProviderNotConfiguredError(f"Provider '{info.provider}' is not registered")

        request = LLMRequest(model=model, prompt=prompt, reasoning_effort=reasoning_effort)
        try:
            async with asyncio.timeout(self._request_timeout_s):
                response = await provider.chat(request)
        except TimeoutError as exc:
            # asyncio.timeout turns cancellation into TimeoutError; provider
            # calls are cancelled through the same context. A CancelledError
            # from an HTTP client disconnect propagates unchanged.
            raise ProviderTimeoutError(
                f"Request to '{model}' timed out after {self._request_timeout_s} s"
            ) from exc

        if not response.answer.strip():
            raise EmptyResponseError(f"Provider returned an empty answer for '{model}'")
        return response
