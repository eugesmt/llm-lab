"""Benchmark orchestration.

Methodology: every model receives the identical prompt with the identical
reasoning configuration; only the model (and provider) varies. Runs are
sequential so latency numbers are not distorted by concurrent load. A failure
of one model is reported in that model's result row instead of failing the
whole benchmark.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.errors import AppError
from app.pricing import estimate_cost
from app.schemas.benchmark import BenchmarkResponse, BenchmarkResult

if TYPE_CHECKING:
    from app.services.llm import LLMService


class BenchmarkService:
    def __init__(self, llm: LLMService) -> None:
        self._llm = llm

    async def run(
        self,
        prompt: str,
        models: list[str],
        reasoning_effort: str | None = None,
        repeats: int = 1,
    ) -> BenchmarkResponse:
        # Validate everything upfront so the benchmark fails fast with 404/422
        # instead of after spending budget on earlier models.
        infos = [self._llm._resolve(model_id, reasoning_effort) for model_id in models]

        results: list[BenchmarkResult] = []
        for info in infos:
            for _ in range(repeats):
                try:
                    response = await self._llm.chat(info.id, prompt, reasoning_effort)
                    results.append(
                        BenchmarkResult(
                            **response.model_dump(),
                            cost=estimate_cost(info.id, response.usage),
                        )
                    )
                except AppError as exc:
                    results.append(
                        BenchmarkResult(
                            provider=info.provider,
                            model=info.id,
                            answer="",
                            latency_ms=0.0,
                            error=exc.message,
                        )
                    )

        return BenchmarkResponse(
            prompt=prompt,
            reasoning_effort=reasoning_effort,
            repeats=repeats,
            results=results,
        )
