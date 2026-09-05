"""GET /api/models: the catalog endpoint the frontend depends on."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_models_endpoint_returns_catalog(client: TestClient) -> None:
    response = client.get("/api/models")
    assert response.status_code == 200

    models = response.json()
    assert [m["id"] for m in models] == ["mock-fast", "mock-plain"]
    for model in models:
        assert set(model) == {
            "id",
            "provider",
            "display_name",
            "thinking_supported",
            "reasoning_effort",
        }


def test_models_endpoint_exposes_capabilities(client: TestClient) -> None:
    models = {m["id"]: m for m in client.get("/api/models").json()}
    assert models["mock-fast"]["thinking_supported"] is True
    assert models["mock-fast"]["reasoning_effort"] == ["low", "high"]
    assert models["mock-plain"]["thinking_supported"] is False
    assert models["mock-plain"]["reasoning_effort"] == []
