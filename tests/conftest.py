"""Shared fixtures: mock catalog + MockProvider wired into create_app.

Unit tests never touch the network: the app is built with the MockProvider
injected behind the same LLMProvider protocol real adapters implement.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.providers.mock import MockProvider
from app.schemas.model import ModelInfo

MOCK_CATALOG = [
    ModelInfo(
        id="mock-fast",
        provider="mock",
        display_name="Mock Fast",
        thinking_supported=True,
        reasoning_effort=["low", "high"],
    ),
    ModelInfo(
        id="mock-plain",
        provider="mock",
        display_name="Mock Plain",
        thinking_supported=False,
        reasoning_effort=[],
    ),
]


@pytest.fixture
def mock_provider() -> MockProvider:
    return MockProvider(delay_s=0.01)


@pytest.fixture
def client(mock_provider: MockProvider) -> TestClient:
    app = create_app(providers={"mock": mock_provider}, catalog=MOCK_CATALOG)
    return TestClient(app)


@pytest.fixture
def provider_calls(mock_provider: MockProvider):
    return mock_provider.requests
