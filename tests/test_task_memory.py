"""Unit tests for deterministic task memory and conversational query rewrite."""

from datetime import UTC, datetime

from app.schemas.conversation import TaskState
from app.services.memory import TaskMemoryService


def make_state(**overrides) -> TaskState:
    values = {
        "goal": None,
        "constraints": [],
        "terms": {},
        "clarifications": [],
        "updated_at": datetime.now(UTC),
    }
    values.update(overrides)
    return TaskState(**values)


def test_update_extracts_russian_markers_without_mutating_input() -> None:
    service = TaskMemoryService()
    state = make_state()

    updated = service.update(
        state,
        "Цель: настроить SIP-звонки. Ограничение: только Docker. "
        "Термин: RAG = поиск по документам. Уточнение: Redis локальный.",
    )

    assert updated is not state
    assert updated.goal == "настроить SIP-звонки"
    assert updated.constraints == ["только Docker"]
    assert updated.terms == {"RAG": "поиск по документам"}
    assert updated.clarifications == ["Redis локальный"]
    assert state.goal is None
    assert state.constraints == []
    assert state.terms == {}


def test_update_supports_goal_constraint_and_term_variants() -> None:
    service = TaskMemoryService()
    state = service.update(None, "Моя цель — собрать чат")
    state = service.update(state, "Хочу сохранить историю")
    state = service.update(state, "нужно отвечать с источниками")
    state = service.update(state, "без платных моделей; не использовать облако")
    state = service.update(state, "обязательно показывать цитаты")
    state = service.update(state, "Под памятью я понимаю SQLite")
    state = service.update(state, "эмбеддинг означает числовой вектор")

    assert state.goal == "отвечать с источниками"
    assert state.constraints == [
        "без платных моделей",
        "не использовать облако",
        "обязательно показывать цитаты",
    ]
    assert state.terms == {
        "памятью": "SQLite",
        "эмбеддинг": "числовой вектор",
    }


def test_update_deduplicates_and_keeps_order_and_limits_collections() -> None:
    service = TaskMemoryService()
    state = make_state(constraints=["Docker", "только SQLite"], clarifications=["первое"])

    updated = service.update(
        state,
        "ограничение: Docker. ограничение: Redis. "
        "уточнение: первое. уточнение: второе.",
    )

    assert updated.constraints == ["Docker", "только SQLite", "Redis"]
    assert updated.clarifications == ["первое", "второе"]

    many = make_state(constraints=[f"c{i}" for i in range(20)])
    limited = service.update(many, "ограничение: c20. ограничение: c21.")
    assert len(limited.constraints) == 20
    assert limited.constraints[0] == "c0"
    assert limited.constraints[-1] == "c19"


def test_overlapping_constraint_keeps_more_specific_wording() -> None:
    service = TaskMemoryService()

    state = service.update(make_state(), "Ограничение: запускать только локально.")

    assert state.constraints == ["запускать только локально"]


def test_detects_task_state_recall_request() -> None:
    assert TaskMemoryService.is_task_state_recall(
        "Напомни мою цель, ограничения и термины."
    )
    assert not TaskMemoryService.is_task_state_recall("Как запустить сервер?")


def test_detects_task_state_update_without_treating_questions_as_updates() -> None:
    assert TaskMemoryService.is_task_state_update("Ограничение: только Docker.")
    assert TaskMemoryService.is_task_state_update("Термин: сервер = SIP Service.")
    assert not TaskMemoryService.is_task_state_update("Как запустить сервер?")


def test_build_context_has_sections_and_only_last_ten_messages() -> None:
    service = TaskMemoryService()
    state = make_state(
        goal="настроить чат",
        constraints=["только локальные модели"],
        terms={"RAG": "поиск по документам"},
        clarifications=["нужны источники"],
    )
    messages = [
        {"role": "user", "content": f"старое сообщение {i}"} for i in range(3)
    ] + [
        {"role": "assistant", "content": f"сообщение {i}"} for i in range(3, 13)
    ]

    context = service.build_context(state, messages)

    assert "Цель: настроить чат" in context
    assert "Ограничения:" in context
    assert "- только локальные модели" in context
    assert "Термины:" in context
    assert "- RAG: поиск по документам" in context
    assert "Уточнения:" in context
    assert "- нужны источники" in context
    assert "старое сообщение 0" not in context
    assert context.count("Ассистент:") + context.count("Пользователь:") == 10


def test_rewrite_adds_goal_to_short_question_inside_active_task() -> None:
    service = TaskMemoryService()
    state = make_state(goal="настроить SIP")
    question = "Как настроить входящий SIP-звонок?"

    rewritten = service.rewrite_query(question, state, [])

    assert rewritten.startswith(question)
    assert "Цель задачи: настроить SIP" in rewritten


def test_rewrite_expands_short_follow_up_using_only_user_context() -> None:
    service = TaskMemoryService()
    state = make_state(
        goal="запустить SIP service",
        constraints=["только Docker"],
        terms={"SIP Bridge": "мост для телефонного трафика"},
    )
    question = "А как это запустить?"
    recent = [
        {"role": "user", "content": "Я настраиваю входящие SIP-звонки."},
        {"role": "assistant", "content": "Ответ ассистента, его нельзя считать фактом."},
    ]

    rewritten = service.rewrite_query(question, state, recent)

    assert rewritten.startswith(question)
    assert "запустить SIP service" in rewritten
    assert "только Docker" in rewritten
    assert "SIP Bridge = мост для телефонного трафика" in rewritten
    assert "Я настраиваю входящие SIP-звонки." in rewritten
    assert "Ответ ассистента" not in rewritten


def test_rewrite_without_state_or_history_returns_follow_up() -> None:
    service = TaskMemoryService()

    assert service.rewrite_query("А в Docker?", None, []) == "А в Docker?"
