# Отчёт: цитаты, источники и анти-галлюцинации

## Цель

Проверить, что улучшенный RAG возвращает ответ вместе с проверяемым
обоснованием: источником документа, `section`, `chunk_id`, непустой цитатой и
режимом «не знаю» при отсутствии подходящего контекста.

## Контрольный набор

В `rag_questions.json` находятся 10 вопросов: 8 вопросов по документации
LiveKit и 2 заведомо неподходящих вопроса — про оплату и погоду.

Для каждого вопроса явно указаны `expectation`, `expected_facts`, `sources` и
`should_unknown`. Это позволяет отдельно проверять содержание ответа и его
привязку к найденным документам.

## Проверки

`run_grounding_evaluation.py` отправляет каждый вопрос в `rag_mode=enhanced` и
проверяет наличие источника, цитаты, непустого `quote`, принадлежность
`citation.chunk_id` к `rag_sources`, дословное наличие цитаты в сохранённом
chunk, смысловую близость ответа и цитат, а также режим «не знаю» для
неподходящих вопросов. Для неподходящих вопросов отсутствие источников и цитат
считается ожидаемым, но должен сработать флаг `unknown`.

Скрипт проверяет фактический JSON API и не создаёт цитаты сам. Смысловая
близость вычисляется теми же multilingual embeddings через локальную Ollama.

Ожидаемый контракт ответа:

```json
{
  "answer": "Для входящего SIP-трафика нужны SIP Bridge и SIP Dispatch Rule.",
  "rag_sources": [{"chunk_id": "structural_000485", "source": "server-oss/sip/README.md", "section": "Creating Bridge and Dispatch Rule", "similarity": 0.82}],
  "citations": [{"chunk_id": "structural_000485", "quote": "Accepting SIP traffic requires two resources be created."}],
  "grounded": true,
  "unknown": false
}
```

## Запуск

Из корня `llm-lab`:

```bash
uv run python -m app.main
```

В другом терминале:

```bash
uv run python evaluation/run_grounding_evaluation.py
```

Результат сохраняется в `evaluation/grounding_evaluation.json`.

## Фактический результат

Контрольный прогон выполнен через `deepseek-v4-flash` на 10 вопросах: 8 по
документации и 2 вне её области.

```text
Все проверки пройдены: 10/10
source_present: 10/10
citation_present: 10/10
quote_non_empty: 10/10
citation_chunk_id_in_rag_sources: 10/10
expected_source_used: 10/10
quote_verbatim_in_chunk: 10/10
answer_supported_by_citations: 10/10
unknown_mode: 10/10
```

Для вопросов об оплате и погоде LLM не вызывалась: API вернул
`finish_reason="rag_unknown"`, `unknown=true`, пустые источники и цитаты.

Можно изменить адрес API и модель:

```bash
LLM_LAB_CHAT_URL=http://127.0.0.1:8000/api/chat \
LLM_LAB_MODEL=liquid/lfm-2.5-2.6b:free \
uv run python evaluation/run_grounding_evaluation.py
```
