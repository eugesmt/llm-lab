"""Сравнивает ответы без RAG, обычного RAG и улучшенного RAG."""

import json
from pathlib import Path

import httpx

API_URL = "http://127.0.0.1:8000/api/chat"
MODEL = "liquid/lfm-2.5-2.6b:free"
QUESTIONS_PATH = Path(__file__).with_name("rag_questions.json")
RESULTS_PATH = Path(__file__).with_name("rag_comparison.json")


def ask(question: str, rag_mode: str) -> dict:
    response = httpx.post(
        API_URL,
        json={
            "model": MODEL,
            "prompt": question,
            "rag_mode": rag_mode,
        },
        timeout=180.0,
    )
    response.raise_for_status()
    return response.json()


def facts_found(answer: str, expected: list[str]) -> int:
    answer_lower = answer.lower()
    return sum(term.lower() in answer_lower for term in expected)


questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
results = []

for item in questions:
    print(f"Обработка: {item['id']}")
    without_rag = ask(item["question"], "off")
    baseline = ask(item["question"], "on")
    enhanced = ask(item["question"], "enhanced")

    results.append(
        {
            "id": item["id"],
            "question": item["question"],
            "expected": item["expected"],
            "sources_expected": item["sources"],
            "without_rag": {
                "answer": without_rag["answer"],
                "facts_found": facts_found(without_rag["answer"], item["expected"]),
                "sources": without_rag.get("rag_sources", []),
            },
            "baseline_rag": {
                "answer": baseline["answer"],
                "facts_found": facts_found(baseline["answer"], item["expected"]),
                "sources": baseline.get("rag_sources", []),
            },
            "enhanced_rag": {
                "answer": enhanced["answer"],
                "facts_found": facts_found(enhanced["answer"], item["expected"]),
                "sources": enhanced.get("rag_sources", []),
            },
        }
    )

RESULTS_PATH.write_text(
    json.dumps(results, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)

total_expected = sum(len(item["expected"]) for item in results)
without_total = sum(item["without_rag"]["facts_found"] for item in results)
baseline_total = sum(item["baseline_rag"]["facts_found"] for item in results)
enhanced_total = sum(item["enhanced_rag"]["facts_found"] for item in results)

print(f"Факты без RAG: {without_total}/{total_expected}")
print(f"Факты обычного RAG: {baseline_total}/{total_expected}")
print(f"Факты улучшенного RAG: {enhanced_total}/{total_expected}")
print(f"Сохранено: {RESULTS_PATH}")
