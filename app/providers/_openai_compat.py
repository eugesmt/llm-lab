"""Shared helpers for OpenAI-compatible APIs (OpenAI and DeepSeek).

This is an internal utility of the provider layer, not application code: it
knows about the ``openai`` SDK types so the individual adapters stay thin, and
nothing outside ``app.providers`` imports it.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    RateLimitError,
)

from app.errors import (
    EmptyResponseError,
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from app.schemas.llm import Usage

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion


def translate_openai_error(exc: Exception, provider_label: str) -> Exception:
    """Map an ``openai`` SDK exception to a normalized application error."""
    if isinstance(exc, asyncio.CancelledError):
        raise exc
    if isinstance(exc, AuthenticationError):
        return ProviderAuthError(f"{provider_label} rejected the API key")
    if isinstance(exc, RateLimitError):
        return ProviderRateLimitError(f"{provider_label} rate limit exceeded")
    if isinstance(exc, APITimeoutError):
        return ProviderTimeoutError(f"{provider_label} request timed out")
    if isinstance(exc, APIConnectionError):
        return ProviderUnavailableError(f"{provider_label} API is unreachable")
    if isinstance(exc, NotFoundError | BadRequestError):
        return ProviderRequestError(f"{provider_label} rejected the request: {exc.message}")
    if isinstance(exc, APIStatusError):
        return ProviderUnavailableError(f"{provider_label} API error: HTTP {exc.status_code}")
    return ProviderRequestError(f"{provider_label} request failed: {exc}")


def parse_completion(
    completion: ChatCompletion, provider_label: str
) -> tuple[str, Usage | None, str | None]:
    """Extract answer, usage and finish reason from a chat completion."""
    if not completion.choices:
        raise EmptyResponseError(f"{provider_label} returned no choices")

    choice = completion.choices[0]
    answer = choice.message.content
    if answer is None or not answer.strip():
        raise EmptyResponseError(f"{provider_label} returned an empty answer")

    usage = None
    if completion.usage is not None:
        details = completion.usage.completion_tokens_details
        usage = Usage(
            prompt_tokens=completion.usage.prompt_tokens,
            completion_tokens=completion.usage.completion_tokens,
            total_tokens=completion.usage.total_tokens,
            reasoning_tokens=details.reasoning_tokens if details is not None else None,
        )

    return answer, usage, choice.finish_reason
