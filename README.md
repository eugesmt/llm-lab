# LLM Lab

Веб-приложение для работы с несколькими LLM-провайдерами (DeepSeek, OpenAI) через единый интерфейс: выбор модели, настройка reasoning effort, измерение latency и token usage, а также benchmark одного prompt на нескольких моделях.

Проект подходит как для обучения работе с LLM API, так и как основа для дальнейшего расширения.

## Архитектура

```text
             WEB UI (vanilla JS)
                │
                ▼
             FastAPI
                │
                ▼
            Pydantic schemas
                │
           LLMService
                │
         LLMProvider (Protocol)
                │
        ┌───────┴────────┐
        ▼                ▼
 DeepSeekProvider   OpenAIProvider
        │                │
        ▼                ▼
   DeepSeek API      OpenAI API
```

Главный принцип: **application code не зависит от SDK провайдера**. Импорты `openai` (используется и для DeepSeek — их API OpenAI-совместим) находятся только внутри `app/providers/`. Новый провайдер добавляется одним файлом в `app/providers/`, без изменений в API, UI, benchmark или бизнес-логике.

## Требования

* [uv](https://docs.astral.sh/uv/) (управляет версией Python сам, глобальный Python не используется)
* Git

## Установка

### 1. Установка uv

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Синхронизация окружения и зависимостей

```bash
uv sync
```

Команда создаст `.venv`, установит Python (если нужно) и все зависимости строго по `uv.lock`.

### 3. Environment variables

API keys берутся только из переменных окружения (или локального файла `.env`, который не коммитится):

```bash
cp .env.example .env
# и заполните ключи в .env
```

| Переменная | Назначение |
|---|---|
| `DEEPSEEK_API_KEY` | ключ DeepSeek (модели `deepseek-v4-*`) |
| `OPENAI_API_KEY` | ключ OpenAI (модели `gpt-6-astra`, `gpt-5.6-*`) |
| `ANTHROPIC_API_KEY` | зарезервировано на будущее |
| `REQUEST_TIMEOUT_S` | общий timeout запроса, сек (по умолчанию `120`) |
| `HOST`, `PORT` | адрес сервера (`127.0.0.1`, `8000`) |

Ключи не хранятся в коде/Git и не логируются (читаются как `SecretStr`).

## Запуск

```bash
uv run llm-lab
```

* **Web UI:** <http://127.0.0.1:8000/>
* **API documentation (Swagger):** <http://127.0.0.1:8000/docs>
* **OpenAPI schema:** <http://127.0.0.1:8000/openapi.json>

## Endpoints

| Endpoint | Описание |
|---|---|
| `GET /api/models` | каталог моделей с capabilities (`thinking_supported`, допустимый `reasoning_effort`) |
| `POST /api/chat` | prompt → одной модели; ответ содержит `answer`, `latency_ms`, `usage`, `finish_reason` |
| `POST /api/benchmark` | один prompt → нескольким моделям (+ `repeats` 1–10); ошибки isolate по запускам |

Пример:

```bash
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"model": "deepseek-v4-flash", "prompt": "Привет!", "reasoning_effort": "low"}'
```

Ответ:

```json
{
  "provider": "deepseek",
  "model": "deepseek-v4-flash",
  "answer": "...",
  "usage": {"prompt_tokens": 10, "completion_tokens": 25, "total_tokens": 35, "reasoning_tokens": null},
  "latency_ms": 812.4,
  "finish_reason": "stop"
}
```

Модели и их цены: `app/catalog.py` и `app/pricing.py` (цены официальные, на 2026-09; если цены нет — cost будет `null`, значения не выдумываются).

## Как добавить нового провайдера

1. Создайте `app/providers/new_provider.py` с классом, реализующим протокол из `app/providers/base.py`:

   ```python
   class NewProvider:
       name = "new"
       async def chat(self, request: LLMRequest) -> LLMResponse: ...
   ```

   Адаптер только формирует provider-specific запрос, вызывает SDK и преобразует ответ/ошибки в наши модели (`app/schemas/llm.py`, `app/errors.py`). SDK-импорт — только здесь.
2. Зарегистрируйте провайдера в `create_app` (`app/main.py`) и добавьте его модели в `app/catalog.py` (и цены в `app/pricing.py`).

API, UI, benchmark и тесты менять не нужно. См. `AnthropicProvider` как упражнение: ключ `ANTHROPIC_API_KEY` уже читается в конфиге.

## Как добавить новую модель

1. Добавьте `ModelInfo` в `app/catalog.py` (`id`, `provider`, `display_name`, `thinking_supported`, `reasoning_effort` — возможности модели описывает backend, UI ничего не угадывает).
2. Если известны цены — добавьте запись в `app/pricing.py`, иначе cost будет `null`.

## Тесты

```bash
uv run pytest                  # unit-тесты (MockProvider, без сети)
uv run pytest -m integration   # реальные API (нужны DEEPSEEK_API_KEY / OPENAI_API_KEY)
```

Покрыто: каталог, chat (успех / 404 / 422 / 502 / 503 / 504 / пустой ответ / usage / latency), benchmark (одинаковый prompt, repeats, изоляция ошибок, cost), адаптеры (форма запроса, мапинг ответа, перевод ошибок SDK), pricing.

## Benchmark

Через UI — вкладка **Benchmark**: выберите модели, введите общий prompt, при необходимости задайте reasoning effort и число повторов.

Через API:

```bash
curl -X POST http://127.0.0.1:8000/api/benchmark \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Ваш вопрос", "models": ["deepseek-v4-flash", "gpt-5.6-luna"], "reasoning_effort": "low", "repeats": 3}'
```

Методология: все модели получают **идентичный prompt** и одинаковую конфигурацию reasoning; меняются только provider и model. Прогоны выполняются последовательно, чтобы latency не искажалась параллельной нагрузкой. Ошибка одной модели не отменяет benchmark — она фиксируется в строке результата этой модели.

## Структура проекта

```text
llm-lab/
├── pyproject.toml / uv.lock   # зависимости (uv), скрипт llm-lab
├── .env.example               # шаблон переменных окружения
├── app/
│   ├── main.py                # create_app(), обработчики ошибок, статика UI
│   ├── config.py              # Pydantic Settings (ключи — SecretStr)
│   ├── catalog.py             # единый каталог моделей
│   ├── pricing.py             # цены и расчёт стоимости
│   ├── errors.py              # application-level ошибки (→ HTTP-коды)
│   ├── api/                   # тонкие роутеры: chat, models, benchmark
│   ├── schemas/               # внутренние и API-модели (Pydantic)
│   ├── services/              # llm.py (валидация, timeout), benchmark.py
│   └── providers/             # base.py (Protocol), deepseek.py, openai.py, mock.py
├── frontend/                  # vanilla JS UI (index.html, app.js, style.css)
└── tests/                     # pytest: unit + integration (env-gated)
```
