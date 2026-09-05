"""Provider adapter tests with fake SDK clients — no network access."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    RateLimitError,
)
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionTokensDetails, CompletionUsage

from app.errors import (
    EmptyResponseError,
    ProviderAuthError,
    ProviderNotConfiguredError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from app.providers.deepseek import DeepSeekProvider
from app.providers.mock import MockProvider
from app.providers.openai import OpenAIProvider
from app.schemas.llm import LLMRequest

URL = "https://api.test/v1/x"


class FakeCompletions:
    def __init__(self, response: ChatCompletion | None = None, error: Exception | None = None):
        self._response = response
        self._error = error
        self.kwargs: dict[str, Any] | None = None

    async def create(self, **kwargs: Any) -> ChatCompletion:
        self.kwargs = kwargs
        if self._error is not None:
            raise self._error
        assert self._response is not None
        return self._response


class FakeClient:
    def __init__(self, completions: FakeCompletions):
        self.chat = type("FakeChat", (), {"completions": completions})()


def make_completion(
    content: str | None = "mock answer",
    finish_reason: str | None = "stop",
    with_usage: bool = True,
    reasoning_tokens: int | None = 7,
) -> ChatCompletion:
    usage = None
    if with_usage:
        details = (
            CompletionTokensDetails(reasoning_tokens=reasoning_tokens)
            if reasoning_tokens is not None
            else None
        )
        usage = CompletionUsage(
            prompt_tokens=11,
            completion_tokens=22,
            total_tokens=33,
            completion_tokens_details=details,
        )
    return ChatCompletion(
        id="chatcmpl-1",
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content=content),
                finish_reason=finish_reason,
            )
        ],
        created=1,
        model="test-model",
        object="chat.completion",
        usage=usage,
    )


def sdk_error(exc_cls: type[Exception]) -> Exception:
    if exc_cls is ValueError:
        return ValueError("boom")
    request = httpx.Request("POST", URL)
    if exc_cls is APITimeoutError:
        return APITimeoutError(request)
    if exc_cls is APIConnectionError:
        return APIConnectionError(request=request)
    status = {
        AuthenticationError: 401,
        RateLimitError: 429,
        BadRequestError: 400,
        NotFoundError: 404,
    }.get(exc_cls, 500)
    return exc_cls("boom", response=httpx.Response(status, request=request), body=None)


@pytest.mark.parametrize(
    ("provider_cls", "name"),
    [(DeepSeekProvider, "deepseek"), (OpenAIProvider, "openai")],
)
class TestAdapters:
    def test_request_shape_with_effort(self, provider_cls: type, name: str) -> None:
        completions = FakeCompletions(make_completion())
        provider = provider_cls(api_key="sk-test", timeout_s=5, client=FakeClient(completions))

        request = LLMRequest(model="test-model", prompt="hi there", reasoning_effort="low")
        response = asyncio.run(provider.chat(request))

        kwargs = completions.kwargs
        assert kwargs["model"] == "test-model"
        assert kwargs["messages"] == [{"role": "user", "content": "hi there"}]
        assert kwargs["reasoning_effort"] == "low"
        assert response.provider == name
        assert response.model == "test-model"

    def test_reasoning_effort_omitted_when_not_requested(
        self, provider_cls: type, name: str
    ) -> None:
        completions = FakeCompletions(make_completion())
        provider = provider_cls(api_key="sk-test", timeout_s=5, client=FakeClient(completions))

        asyncio.run(provider.chat(LLMRequest(model="test-model", prompt="hi")))
        assert "reasoning_effort" not in completions.kwargs

    def test_response_mapping(self, provider_cls: type, name: str) -> None:
        completions = FakeCompletions(make_completion())
        provider = provider_cls(api_key="sk-test", timeout_s=5, client=FakeClient(completions))

        response = asyncio.run(provider.chat(LLMRequest(model="test-model", prompt="hi")))

        assert response.answer == "mock answer"
        assert response.finish_reason == "stop"
        assert response.latency_ms >= 0
        usage = response.usage
        assert usage is not None
        assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (11, 22, 33)
        assert usage.reasoning_tokens == 7

    def test_usage_absent(self, provider_cls: type, name: str) -> None:
        completions = FakeCompletions(make_completion(with_usage=False))
        provider = provider_cls(api_key="sk-test", timeout_s=5, client=FakeClient(completions))

        response = asyncio.run(provider.chat(LLMRequest(model="test-model", prompt="hi")))
        assert response.usage is None
        assert response.answer == "mock answer"

    def test_usage_without_reasoning_details(self, provider_cls: type, name: str) -> None:
        completions = FakeCompletions(make_completion(reasoning_tokens=None))
        provider = provider_cls(api_key="sk-test", timeout_s=5, client=FakeClient(completions))

        response = asyncio.run(provider.chat(LLMRequest(model="test-model", prompt="hi")))
        assert response.usage is not None
        assert response.usage.reasoning_tokens is None

    @pytest.mark.parametrize(
        ("sdk_exc_cls", "expected"),
        [
            (AuthenticationError, ProviderAuthError),
            (RateLimitError, ProviderRateLimitError),
            (APITimeoutError, ProviderTimeoutError),
            (APIConnectionError, ProviderUnavailableError),
            (BadRequestError, ProviderRequestError),
            (NotFoundError, ProviderRequestError),
            (ValueError, ProviderRequestError),
        ],
    )
    def test_error_translation(
        self, provider_cls: type, name: str, sdk_exc_cls: type[Exception], expected: type[Exception]
    ) -> None:
        completions = FakeCompletions(error=sdk_error(sdk_exc_cls))
        provider = provider_cls(api_key="sk-test", timeout_s=5, client=FakeClient(completions))

        with pytest.raises(expected):
            asyncio.run(provider.chat(LLMRequest(model="test-model", prompt="hi")))

    def test_cancelled_error_propagates(self, provider_cls: type, name: str) -> None:
        completions = FakeCompletions(error=asyncio.CancelledError())
        provider = provider_cls(api_key="sk-test", timeout_s=5, client=FakeClient(completions))

        with pytest.raises(asyncio.CancelledError):
            asyncio.run(provider.chat(LLMRequest(model="test-model", prompt="hi")))


def test_deepseek_none_effort_disables_thinking() -> None:
    completions = FakeCompletions(make_completion())
    provider = DeepSeekProvider(api_key="sk-test", timeout_s=5, client=FakeClient(completions))

    response = asyncio.run(
        provider.chat(LLMRequest(model="test-model", prompt="hi", reasoning_effort="none"))
    )

    # "none" maps to the thinking toggle, not to an effort value.
    assert "reasoning_effort" not in completions.kwargs
    assert completions.kwargs["extra_body"] == {"thinking": {"type": "disabled"}}
    assert response.answer == "mock answer"


def test_missing_api_key_is_not_configured() -> None:
    with pytest.raises(ProviderNotConfiguredError):
        asyncio.run(
            DeepSeekProvider(api_key=None, timeout_s=5).chat(LLMRequest(model="m", prompt="p"))
        )
    with pytest.raises(ProviderNotConfiguredError):
        asyncio.run(
            OpenAIProvider(api_key=None, timeout_s=5).chat(LLMRequest(model="m", prompt="p"))
        )


@pytest.mark.parametrize("provider_cls", [DeepSeekProvider, OpenAIProvider])
def test_empty_choices_and_content(provider_cls: type[DeepSeekProvider | OpenAIProvider]) -> None:
    no_choices = make_completion()
    no_choices.choices = []
    provider = provider_cls(
        api_key="sk-test", timeout_s=5, client=FakeClient(FakeCompletions(no_choices))
    )
    with pytest.raises(EmptyResponseError):
        asyncio.run(provider.chat(LLMRequest(model="test-model", prompt="hi")))

    blank = make_completion(content="   ")
    provider = provider_cls(
        api_key="sk-test", timeout_s=5, client=FakeClient(FakeCompletions(blank))
    )
    with pytest.raises(EmptyResponseError):
        asyncio.run(provider.chat(LLMRequest(model="test-model", prompt="hi")))


def test_null_content_is_empty_response() -> None:
    completion = make_completion(content=None)
    provider = OpenAIProvider(
        api_key="sk-test", timeout_s=5, client=FakeClient(FakeCompletions(completion))
    )
    with pytest.raises(EmptyResponseError):
        asyncio.run(provider.chat(LLMRequest(model="test-model", prompt="hi")))


def test_mock_provider_latency_and_usage() -> None:
    provider = MockProvider(delay_s=0.05)
    response = asyncio.run(provider.chat(LLMRequest(model="m", prompt="a b c")))
    assert response.latency_ms >= 45
    assert response.usage is not None
    assert response.usage.prompt_tokens == 3
    assert response.finish_reason == "stop"
