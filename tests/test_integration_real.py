"""Integration tests against real provider APIs.

Deselected by default (addopts: -m "not integration"). Run explicitly:

    DEEPSEEK_API_KEY=... uv run pytest -m integration
"""

from __future__ import annotations

import asyncio
import os

import pytest

from app.schemas.llm import LLMRequest

pytestmark = pytest.mark.integration


def test_deepseek_real_api() -> None:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        pytest.skip("DEEPSEEK_API_KEY is not set")

    from app.providers.deepseek import DeepSeekProvider

    provider = DeepSeekProvider(api_key=api_key, timeout_s=120)
    response = asyncio.run(
        provider.chat(LLMRequest(model="deepseek-v4-flash", prompt="Reply with the single word OK"))
    )
    assert response.provider == "deepseek"
    assert response.answer.strip()
    assert response.latency_ms > 0
    assert response.usage is not None
    assert response.usage.total_tokens


def test_openai_real_api() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY is not set")

    from app.providers.openai import OpenAIProvider

    provider = OpenAIProvider(api_key=api_key, timeout_s=120)
    response = asyncio.run(
        provider.chat(LLMRequest(model="gpt-5.6-luna", prompt="Reply with the single word OK"))
    )
    assert response.provider == "openai"
    assert response.answer.strip()
    assert response.latency_ms > 0
    assert response.usage is not None
    assert response.usage.total_tokens
