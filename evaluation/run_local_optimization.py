"""Reproducible baseline/optimized benchmark for local Ollama.

The script intentionally uses Ollama's native API: unlike the OpenAI-compatible
endpoint, it reports generation timings and accepts context/thinking controls.
No third-party judge is used; quality is a transparent task-specific rubric.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "evaluation" / "local_llm_cases.json"
DEFAULT_OUTPUT = ROOT / "evaluation" / "local_llm_results.json"

# The optimized model's Modelfile owns this compact system prompt. Keep the
# benchmark request identical to the deployed profile.
SRE_SYSTEM_PROMPT = """Ты SRE-ассистент первой линии. Анализируй только данные из запроса.
Не выдумывай метрики и не называй причину доказанной, если данных недостаточно.
Высокий iowait при низком user+system означает ожидание дискового I/O, а не
перегруженный CPU. Для iowait >= 50% и disk util >= 90% диагноз — насыщение
дискового I/O; используй только проверки `iostat -xz 1`, `pidstat -d 1`,
`iotop -oPa`. Для IUse=100% используй только `df -i` и поиск каталогов с
множеством файлов. Для пула PostgreSQL используй `pg_stat_activity` и метрики
pool active/idle/pending. Для NXDOMAIN используй `dig` и authoritative DNS.
Для memory pressure используй `free`, `/proc/meminfo`, memory PSI и журнал OOM.
Не смешивай категории и не добавляй `df`, `du`, `find` или `fstrim` к диагнозу
дискового I/O. Не предлагай remediation: удаление, перезапуск, изменение
конфигурации или порогов. Отвечай по-русски строго в формате:
Диагноз: <одна причина и уверенность>
Проверки: <2-3 конкретные команды или метрики>
Начинай сразу с `Диагноз:` и не показывай внутренние рассуждения."""


@dataclass(frozen=True)
class Profile:
    name: str
    model: str
    system: str | None
    think: bool | None
    options: dict[str, int | float]


def profiles(base_model: str, optimized_model: str) -> list[Profile]:
    return [
        # Mirrors the old adapter (base model, user message only, model
        # defaults). The 512-token cap is only a benchmark safety guard; the
        # old adapter itself had no cap and one trial ran for over 3 minutes.
        Profile("baseline", base_model, None, None, {"num_predict": 512}),
        Profile(
            "optimized",
            optimized_model,
            SRE_SYSTEM_PROMPT,
            False,
            {
                "temperature": 0.1,
                "num_predict": 192,
                "num_ctx": 4096,
                "seed": 42,
                "top_k": 20,
                "top_p": 0.8,
                "repeat_penalty": 1.1,
            },
        ),
    ]


def score_answer(answer: str, fact_groups: list[list[str]]) -> dict[str, Any]:
    """Score facts (8), required structure (1), and concision (1)."""
    normalized = answer.casefold()
    matched = [any(term.casefold() in normalized for term in group) for group in fact_groups]
    fact_score = 8.0 * sum(matched) / len(matched) if matched else 0.0
    headings = ["диагноз:", "проверки:"]
    structure_score = sum(heading in normalized for heading in headings) / len(headings)
    concision_score = 1.0 if 0 < len(answer) <= 1200 else 0.0
    return {
        "score": round(fact_score + structure_score + concision_score, 2),
        "matched_fact_groups": sum(matched),
        "total_fact_groups": len(matched),
        "has_required_structure": structure_score == 1.0,
        "within_1200_chars": concision_score == 1.0,
    }


async def model_memory(client: httpx.AsyncClient, model: str) -> dict[str, int | None]:
    response = await client.get("/api/ps")
    response.raise_for_status()
    wanted = model.split(":", 1)[0]
    entries = response.json().get("models", [])
    entry = next(
        (
            item
            for item in entries
            if item.get("name", item.get("model", "")).split(":", 1)[0] == wanted
        ),
        {},
    )
    # Ollama may report the base-model name for a derived Modelfile profile.
    if not entry and len(entries) == 1:
        entry = entries[0]
    return {
        "size_bytes": entry.get("size"),
        "size_vram_bytes": entry.get("size_vram"),
        "context_length": entry.get("context_length"),
    }


async def run_case(
    client: httpx.AsyncClient, profile: Profile, case: dict[str, Any]
) -> dict[str, Any]:
    messages: list[dict[str, str]] = []
    if profile.system:
        messages.append({"role": "system", "content": profile.system})
    user_prompt = case["prompt"]
    if profile.name == "optimized":
        user_prompt += '\n\nНачни ответ сразу с "Диагноз:".'
    messages.append({"role": "user", "content": user_prompt})
    payload: dict[str, Any] = {
        "model": profile.model,
        "messages": messages,
        "stream": False,
        "keep_alive": "5m",
    }
    if profile.think is not None:
        payload["think"] = profile.think
    if profile.options:
        payload["options"] = profile.options

    started = perf_counter()
    response = await client.post("/api/chat", json=payload)
    wall_ms = (perf_counter() - started) * 1000
    response.raise_for_status()
    data = response.json()
    answer = data.get("message", {}).get("content", "").strip()
    thinking = data.get("message", {}).get("thinking", "")
    eval_count = data.get("eval_count") or 0
    eval_duration_s = (data.get("eval_duration") or 0) / 1_000_000_000
    return {
        "case_id": case["id"],
        "answer": answer,
        "quality": score_answer(answer, case["fact_groups"]),
        "wall_latency_ms": round(wall_ms, 2),
        "ollama_total_ms": round((data.get("total_duration") or 0) / 1_000_000, 2),
        "load_ms": round((data.get("load_duration") or 0) / 1_000_000, 2),
        "prompt_tokens": data.get("prompt_eval_count"),
        "generated_tokens": eval_count,
        "thinking_chars": len(thinking),
        "tokens_per_second": round(eval_count / eval_duration_s, 2) if eval_duration_s else None,
        "done_reason": data.get("done_reason"),
    }


def summarize(runs: list[dict[str, Any]], memory: dict[str, Any]) -> dict[str, Any]:
    return {
        "quality_mean_0_10": round(statistics.mean(r["quality"]["score"] for r in runs), 2),
        "latency_mean_ms": round(statistics.mean(r["wall_latency_ms"] for r in runs), 2),
        "latency_median_ms": round(statistics.median(r["wall_latency_ms"] for r in runs), 2),
        "generated_tokens_mean": round(statistics.mean(r["generated_tokens"] for r in runs), 2),
        "tokens_per_second_mean": round(
            statistics.mean(r["tokens_per_second"] for r in runs if r["tokens_per_second"]), 2
        ),
        "model_memory": memory,
    }


async def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    if args.only and args.output.exists():
        result = json.loads(args.output.read_text(encoding="utf-8"))
    else:
        result = {
            "task": "SRE incident triage",
            "rubric": "8 points facts + 1 structure + 1 concision",
            "profiles": {},
        }
    cases_by_id = {case["id"]: case for case in cases}
    # Keep resumed results consistent if the deterministic rubric changed.
    for saved_profile in result["profiles"].values():
        for saved_run in saved_profile["runs"]:
            case = cases_by_id[saved_run["case_id"]]
            saved_run["quality"] = score_answer(saved_run["answer"], case["fact_groups"])
        saved_profile["summary"] = summarize(
            saved_profile["runs"], saved_profile["summary"]["model_memory"]
        )
    timeout = httpx.Timeout(args.timeout)
    async with httpx.AsyncClient(base_url=args.ollama_url, timeout=timeout) as client:
        for profile in profiles(args.base_model, args.optimized_model):
            if args.only and profile.name != args.only:
                continue
            runs = []
            for case in cases:
                print(f"[{profile.name}] {case['id']}", flush=True)
                runs.append(await run_case(client, profile, case))
            memory = await model_memory(client, profile.model)
            result["profiles"][profile.name] = {
                "model": profile.model,
                "temperature": profile.options.get("temperature", "model default"),
                "max_tokens": profile.options.get("num_predict", "model default"),
                "context_window": profile.options.get("num_ctx", "Ollama default"),
                "thinking": profile.think if profile.think is not None else "model default",
                "runs": runs,
                "summary": summarize(runs, memory),
            }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--base-model", default="qwen2.5:3b")
    parser.add_argument("--optimized-model", default="llm-lab-sre:latest")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--only",
        choices=["baseline", "optimized"],
        help="Rerun one profile and preserve the other profile already in --output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = asyncio.run(benchmark(args))
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
