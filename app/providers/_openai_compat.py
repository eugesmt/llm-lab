"""Shared helpers for OpenAI-compatible APIs (OpenAI and DeepSeek).

This is an internal utility of the provider layer, not application code: it
knows about the ``openai`` SDK types so the individual adapters stay thin, and
nothing outside ``app.providers`` imports it.
"""

from __future__ import annotations

import asyncio
import json
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
from app.schemas.llm import (
    ToolCall,
    ToolCompletionRequest,
    ToolCompletionResponse,
    Usage,
)

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


def build_tool_completion_payload(request: ToolCompletionRequest) -> dict[str, object]:
    """Convert provider-neutral messages/tools into Chat Completions payloads."""
    messages: list[dict[str, object]] = []
    for message in request.messages:
        item: dict[str, object] = {"role": message.role}
        if message.content is not None:
            item["content"] = message.content
        if message.tool_call_id is not None:
            item["tool_call_id"] = message.tool_call_id
        if message.reasoning_content is not None:
            item["reasoning_content"] = message.reasoning_content
        if message.tool_calls:
            item["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments, ensure_ascii=False),
                    },
                }
                for call in message.tool_calls
            ]
        messages.append(item)

    tools = [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.input_schema,
            },
        }
        for tool in request.tools
    ]
    return {"messages": messages, "tools": tools, "tool_choice": "auto"}


def _parse_usage(completion: ChatCompletion) -> Usage | None:
    if completion.usage is None:
        return None
    details = completion.usage.completion_tokens_details
    return Usage(
        prompt_tokens=completion.usage.prompt_tokens,
        completion_tokens=completion.usage.completion_tokens,
        total_tokens=completion.usage.total_tokens,
        reasoning_tokens=details.reasoning_tokens if details is not None else None,
    )


def parse_tool_completion(
    completion: ChatCompletion,
    provider_name: str,
    provider_label: str,
    model: str,
    latency_ms: float,
) -> ToolCompletionResponse:
    if not completion.choices:
        raise EmptyResponseError(f"{provider_label} returned no choices")

    choice = completion.choices[0]
    message = choice.message
    calls: list[ToolCall] = []
    for raw_call in message.tool_calls or []:
        try:
            arguments = json.loads(raw_call.function.arguments)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ProviderRequestError(f"{provider_label} returned invalid tool arguments") from exc
        if not isinstance(arguments, dict):
            raise ProviderRequestError(f"{provider_label} returned non-object tool arguments")
        calls.append(ToolCall(id=raw_call.id, name=raw_call.function.name, arguments=arguments))

    content = message.content
    if not calls and (content is None or not content.strip()):
        raise EmptyResponseError(f"{provider_label} returned an empty answer")

    reasoning_content = getattr(message, "reasoning_content", None)
    if reasoning_content is None and message.model_extra:
        reasoning_content = message.model_extra.get("reasoning_content")

    return ToolCompletionResponse(
        provider=provider_name,
        model=model,
        content=content,
        tool_calls=calls,
        reasoning_content=reasoning_content,
        usage=_parse_usage(completion),
        latency_ms=latency_ms,
        finish_reason=choice.finish_reason,
    )
