"""POST /api/benchmark: identical prompt, repeats, error isolation, costs."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_benchmark_success_and_same_prompt(client: TestClient, provider_calls) -> None:
    response = client.post(
        "/api/benchmark",
        json={"prompt": "Один prompt для всех", "models": ["mock-fast", "mock-plain"]},
    )
    assert response.status_code == 200

    data = response.json()
    assert data["prompt"] == "Один prompt для всех"
    assert data["repeats"] == 1
    assert len(data["results"]) == 2

    for result in data["results"]:
        assert result["answer"] == f"[mock:{result['model']}] {data['prompt']}"
        assert result["error"] is None
        assert result["finish_reason"] == "stop"
        assert result["latency_ms"] >= 0

    # Every model received exactly the same prompt content.
    assert [call.prompt for call in provider_calls] == [
        "Один prompt для всех",
        "Один prompt для всех",
    ]


def test_benchmark_repeats(client: TestClient) -> None:
    response = client.post(
        "/api/benchmark",
        json={"prompt": "p", "models": ["mock-fast"], "repeats": 3},
    )
    data = response.json()
    assert data["repeats"] == 3
    assert len(data["results"]) == 3
    assert all(r["model"] == "mock-fast" for r in data["results"])


def test_benchmark_effort_applied_to_all(client: TestClient, provider_calls) -> None:
    response = client.post(
        "/api/benchmark",
        json={"prompt": "p", "models": ["mock-fast"], "reasoning_effort": "high"},
    )
    assert response.status_code == 200
    assert provider_calls[0].reasoning_effort == "high"


def test_benchmark_unknown_model_fails_fast(client: TestClient, provider_calls) -> None:
    response = client.post(
        "/api/benchmark",
        json={"prompt": "p", "models": ["mock-fast", "nope"]},
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "ModelNotFoundError"
    # Upfront validation: no model ran before the error.
    assert provider_calls == []


def test_benchmark_error_isolation(mock_provider, client) -> None:  # noqa: ARG001
    # Runtime failure of one model becomes a per-run error row; other models
    # still get their results (mock-plain is set to fail, mock-fast is not).
    from fastapi.testclient import TestClient as TC

    from app.main import create_app
    from tests.conftest import MOCK_CATALOG

    mock_provider._fail_models.add("mock-plain")
    app = create_app(providers={"mock": mock_provider}, catalog=MOCK_CATALOG)
    response = TC(app).post(
        "/api/benchmark",
        json={"prompt": "p", "models": ["mock-fast", "mock-plain"]},
    )
    assert response.status_code == 200

    by_model = {r["model"]: r for r in response.json()["results"]}
    assert by_model["mock-fast"]["error"] is None
    assert by_model["mock-fast"]["answer"]
    assert by_model["mock-plain"]["error"] is not None
    assert "mock failure" in by_model["mock-plain"]["error"]
    assert by_model["mock-plain"]["answer"] == ""


def test_benchmark_unsupported_effort_fails_fast(client: TestClient) -> None:
    # Capability violations are validation errors: nothing runs at all.
    response = client.post(
        "/api/benchmark",
        json={"prompt": "p", "models": ["mock-fast", "mock-plain"], "reasoning_effort": "high"},
    )
    assert response.status_code == 422
    assert "mock-plain" in response.json()["error"]["message"]


def test_benchmark_cost_is_null_for_unpriced_models(client: TestClient) -> None:
    response = client.post(
        "/api/benchmark",
        json={"prompt": "p", "models": ["mock-fast"]},
    )
    result = response.json()["results"][0]
    assert result["cost"] is None  # no invented prices for mock models


def test_benchmark_invalid_request(client: TestClient) -> None:
    assert client.post("/api/benchmark", json={"prompt": "p", "models": []}).status_code == 422
    assert (
        client.post("/api/benchmark", json={"prompt": "", "models": ["mock-fast"]}).status_code
        == 422
    )
    assert (
        client.post(
            "/api/benchmark",
            json={"prompt": "p", "models": ["mock-fast"], "repeats": 11},
        ).status_code
        == 422
    )
