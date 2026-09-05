"""Cost estimation: real prices applied correctly, None when unknown."""

from __future__ import annotations

from app.catalog import MODEL_CATALOG
from app.pricing import PRICING, estimate_cost
from app.schemas.llm import Usage


def test_cost_for_known_model() -> None:
    usage = Usage(prompt_tokens=1_000_000, completion_tokens=1_000_000, total_tokens=2_000_000)
    cost = estimate_cost("deepseek-v4-flash", usage)
    assert cost is not None
    assert cost.input_cost == 0.44
    assert cost.output_cost == 1.32
    assert cost.total_cost == 1.76


def test_cost_for_small_usage() -> None:
    usage = Usage(prompt_tokens=500, completion_tokens=900, total_tokens=1400)
    cost = estimate_cost("deepseek-v4-flash", usage)
    assert cost is not None
    assert cost.input_cost == round(500 / 1_000_000 * 0.44, 6)
    assert cost.output_cost == round(900 / 1_000_000 * 1.32, 6)
    assert cost.total_cost == round(cost.input_cost + cost.output_cost, 6)


def test_cost_partial_usage() -> None:
    usage = Usage(prompt_tokens=1_000_000, completion_tokens=None, total_tokens=1_000_000)
    cost = estimate_cost("deepseek-v4-pro", usage)
    assert cost is not None
    assert cost.input_cost == 1.32
    assert cost.output_cost is None
    assert cost.total_cost == 1.32


def test_cost_none_for_unknown_model_or_usage() -> None:
    usage = Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2)
    assert estimate_cost("unknown-model", usage) is None
    assert estimate_cost("deepseek-v4-flash", None) is None


def test_pricing_covers_every_catalog_model() -> None:
    for model in MODEL_CATALOG:
        assert model.id in PRICING, f"missing pricing for {model.id}"
