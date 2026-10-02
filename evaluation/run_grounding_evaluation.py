"""Проверяет источники, цитаты и режим «не знаю» на контрольных вопросах.

Скрипт не подменяет отсутствующие цитаты искусственными значениями. Он
проверяет фактический JSON, который вернул API, поэтому отчёт показывает,
какие части контракта grounding уже реализованы.
"""

import json
import math
import os
import sqlite3
from pathlib import Path
from typing import Any

import httpx

API_URL = os.getenv("LLM_LAB_CHAT_URL", "http://127.0.0.1:8000/api/chat")
MODEL = os.getenv("LLM_LAB_MODEL", "liquid/lfm-2.5-2.6b:free")
QUESTIONS_PATH = Path(__file__).with_name("rag_questions.json")
RESULTS_PATH = Path(__file__).with_name("grounding_evaluation.json")
TIMEOUT_SECONDS = float(os.getenv("LLM_LAB_TIMEOUT", "180"))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "qwen3-embedding:0.6b")
SEMANTIC_THRESHOLD = float(os.getenv("GROUNDING_SEMANTIC_THRESHOLD", "0.45"))
CHUNKS_DATABASE = Path(
    os.getenv(
        "RAG_CHUNKS_DATABASE",
        "/home/lab-dev/Projects/rag/rag_lab/indexes/structural/chunks.sqlite",
    )
)


def ask(question: str) -> dict[str, Any]:
    """Отправляет вопрос в улучшенный RAG-режим."""

    response = httpx.post(
        API_URL,
        json={"model": MODEL, "prompt": question, "rag_mode": "enhanced"},
        timeout=TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def source_chunk_ids(response: dict[str, Any]) -> set[str]:
    """Возвращает chunk_id из источников ответа."""

    return {
        str(source["chunk_id"])
        for source in response.get("rag_sources", [])
        if source.get("chunk_id")
    }


def response_citations(response: dict[str, Any]) -> list[dict[str, Any]]:
    """Читает будущий/текущий формат citations без изменения API."""

    citations = response.get("citations")
    if citations is None:
        citations = response.get("rag_citations", [])
    return citations if isinstance(citations, list) else []


def unknown_mode_detected(response: dict[str, Any]) -> bool:
    """Проверяет флаг unknown или явный отказ в тексте ответа."""

    for key in ("unknown", "is_unknown", "unknown_mode", "rag_unknown"):
        if response.get(key) is True:
            return True

    answer = str(response.get("answer", "")).lower()
    markers = (
        "не знаю",
        "нет информации",
        "недостаточно информации",
        "не найдено в документации",
        "не могу ответить",
    )
    return any(marker in answer for marker in markers)


def quote_exists_in_index(citation: dict[str, Any]) -> bool:
    """Проверяет, что quote дословно находится в сохранённом chunk."""

    chunk_id = str(citation.get("chunk_id", ""))
    quote = str(citation.get("quote", "")).strip()
    if not chunk_id or not quote or not CHUNKS_DATABASE.exists():
        return False
    with sqlite3.connect(CHUNKS_DATABASE) as connection:
        row = connection.execute(
            "SELECT text FROM chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
    return row is not None and quote in str(row[0])


def semantic_similarity(answer: str, quotes: list[str]) -> float | None:
    """Сравнивает смысл ответа и цитат через multilingual embeddings."""

    context = "\n".join(quote for quote in quotes if quote.strip())
    if not answer.strip() or not context:
        return None
    try:
        response = httpx.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={"model": EMBEDDING_MODEL, "input": [answer, context], "keep_alive": 0},
            timeout=60.0,
        )
        response.raise_for_status()
        first, second = response.json()["embeddings"]
    except (httpx.HTTPError, KeyError, TypeError, ValueError):
        return None

    dot = sum(a * b for a, b in zip(first, second, strict=True))
    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))
    if first_norm == 0 or second_norm == 0:
        return None
    return dot / (first_norm * second_norm)


def evaluate_item(item: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    """Проверяет обязательные признаки grounded-ответа."""

    sources = response.get("rag_sources", [])
    citations = response_citations(response)
    known_chunk_ids = source_chunk_ids(response)
    valid_citation_chunks = [
        citation.get("chunk_id") in known_chunk_ids
        for citation in citations
        if isinstance(citation, dict)
    ]
    non_empty_quotes = [
        bool(str(citation.get("quote", "")).strip())
        for citation in citations
        if isinstance(citation, dict)
    ]
    unknown_detected = unknown_mode_detected(response)
    expected_sources = set(item["sources"])
    actual_sources = {
        str(source.get("source", ""))
        for source in sources
        if isinstance(source, dict)
    }
    quotes = [
        str(citation.get("quote", ""))
        for citation in citations
        if isinstance(citation, dict)
    ]
    quotes_in_index = [
        quote_exists_in_index(citation)
        for citation in citations
        if isinstance(citation, dict)
    ]
    semantic_score = (
        None
        if item["should_unknown"]
        else semantic_similarity(str(response.get("answer", "")), quotes)
    )

    source_ok = bool(sources) if not item["should_unknown"] else not sources
    citation_ok = bool(citations) if not item["should_unknown"] else not citations
    quote_ok = (
        True
        if item["should_unknown"]
        else bool(non_empty_quotes) and all(non_empty_quotes)
    )
    chunk_link_ok = (
        True
        if item["should_unknown"]
        else bool(valid_citation_chunks) and all(valid_citation_chunks)
    )
    unknown_ok = unknown_detected == item["should_unknown"]
    expected_source_ok = (
        not actual_sources
        if item["should_unknown"]
        else bool(expected_sources & actual_sources)
    )
    quote_verbatim_ok = (
        True
        if item["should_unknown"]
        else bool(quotes_in_index) and all(quotes_in_index)
    )
    semantic_support_ok = (
        True
        if item["should_unknown"]
        else semantic_score is not None and semantic_score >= SEMANTIC_THRESHOLD
    )

    return {
        "id": item["id"],
        "question": item["question"],
        "expectation": item["expectation"],
        "expected_facts": item["expected_facts"],
        "expected_sources": item["sources"],
        "should_unknown": item["should_unknown"],
        "answer": response.get("answer", ""),
        "rag_mode": response.get("rag_mode"),
        "rag_sources": sources,
        "citations": citations,
        "checks": {
            "source_present": source_ok,
            "citation_present": citation_ok,
            "quote_non_empty": quote_ok,
            "citation_chunk_id_in_rag_sources": chunk_link_ok,
            "expected_source_used": expected_source_ok,
            "quote_verbatim_in_chunk": quote_verbatim_ok,
            "answer_supported_by_citations": semantic_support_ok,
            "unknown_mode": unknown_ok,
        },
        "semantic_similarity": semantic_score,
        "all_checks_passed": all(
            (
                source_ok,
                citation_ok,
                quote_ok,
                chunk_link_ok,
                expected_source_ok,
                quote_verbatim_ok,
                semantic_support_ok,
                unknown_ok,
            )
        ),
    }


def main() -> None:
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    results = []

    for number, item in enumerate(questions, start=1):
        print(f"[{number}/{len(questions)}] {item['id']}")
        try:
            response = ask(item["question"])
            results.append(evaluate_item(item, response))
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as error:
            results.append(
                {
                    "id": item["id"],
                    "question": item["question"],
                    "expectation": item["expectation"],
                    "should_unknown": item["should_unknown"],
                    "error": str(error),
                    "checks": {
                        "source_present": False,
                        "citation_present": False,
                        "quote_non_empty": False,
                        "citation_chunk_id_in_rag_sources": False,
                        "expected_source_used": False,
                        "quote_verbatim_in_chunk": False,
                        "answer_supported_by_citations": False,
                        "unknown_mode": False,
                    },
                    "all_checks_passed": False,
                }
            )

    passed = sum(result["all_checks_passed"] for result in results)
    check_names = (
        "source_present",
        "citation_present",
        "quote_non_empty",
        "citation_chunk_id_in_rag_sources",
        "expected_source_used",
        "quote_verbatim_in_chunk",
        "answer_supported_by_citations",
        "unknown_mode",
    )
    summary = {
        "api_url": API_URL,
        "model": MODEL,
        "rag_mode": "enhanced",
        "questions_total": len(results),
        "questions_all_checks_passed": passed,
        "checks_passed": {
            name: sum(result.get("checks", {}).get(name, False) for result in results)
            for name in check_names
        },
        "results": results,
    }
    RESULTS_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Все проверки пройдены: {passed}/{len(results)}")
    for name, count in summary["checks_passed"].items():
        print(f"{name}: {count}/{len(results)}")
    print(f"Сохранено: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
