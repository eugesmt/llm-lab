"""Price table and cost estimation, kept separate from provider logic.

Prices are USD per 1M tokens, official peak rates as of 2026-09 (DeepSeek
off-peak is exactly half; cache-hit input discounts are not tracked because
Usage does not distinguish cached tokens). If a price is unknown or a usage
value is missing, the corresponding cost is None — never invented.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.schemas.llm import Cost, Usage


@dataclass(frozen=True)
class ModelPricing:
    input_per_1m: float
    output_per_1m: float


PRICING: dict[str, ModelPricing] = {
    "deepseek-v4-flash": ModelPricing(input_per_1m=0.44, output_per_1m=1.32),
    "deepseek-v4-pro": ModelPricing(input_per_1m=1.32, output_per_1m=3.96),
    # Free OpenRouter model: zero price, rate-limited instead.
    "inclusionai/ling-3.0-flash-sante:free": ModelPricing(input_per_1m=0.0, output_per_1m=0.0),
}


def estimate_cost(model_id: str, usage: Usage | None) -> Cost | None:
    """Estimate the cost of one response; None when pricing or usage is unknown."""
    pricing = PRICING.get(model_id)
    if pricing is None or usage is None:
        return None

    input_cost = (
        round(usage.prompt_tokens / 1_000_000 * pricing.input_per_1m, 6)
        if usage.prompt_tokens is not None
        else None
    )
    output_cost = (
        round(usage.completion_tokens / 1_000_000 * pricing.output_per_1m, 6)
        if usage.completion_tokens is not None
        else None
    )
    known = [value for value in (input_cost, output_cost) if value is not None]
    total_cost = round(sum(known), 6) if known else None

    return Cost(input_cost=input_cost, output_cost=output_cost, total_cost=total_cost)
