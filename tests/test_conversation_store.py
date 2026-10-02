"""Isolated tests for SQLite chat history and task memory."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.schemas.conversation import TaskState
from app.storage.chat_store import ChatStore, ConversationNotFoundError


def test_database_is_created_and_survives_restart(tmp_path: Path) -> None:
    database = tmp_path / "chat.sqlite"

    first_store = ChatStore(database)
    conversation = first_store.create_conversation()
    first_store.append_message(conversation.id, "user", "Привет")
    first_store.close()

    second_store = ChatStore(database)
    restored = second_store.get_conversation(conversation.id)
    messages = second_store.list_messages(conversation.id)
    second_store.close()

    assert restored.id == conversation.id
    assert [message.content for message in messages] == ["Привет"]


def test_messages_keep_chronological_order_and_limit(tmp_path: Path) -> None:
    store = ChatStore(tmp_path / "chat.sqlite")
    conversation = store.create_conversation()
    for number in range(1, 5):
        store.append_message(conversation.id, "user", f"message {number}")

    messages = store.list_messages(conversation.id, limit=2)

    assert [message.content for message in messages] == ["message 3", "message 4"]


def test_task_state_round_trip_preserves_russian_json(tmp_path: Path) -> None:
    store = ChatStore(tmp_path / "chat.sqlite")
    conversation = store.create_conversation()
    state = TaskState(
        goal="Настроить RAG-чат",
        constraints=["Использовать SQLite"],
        terms={"RAG": "поиск контекста перед ответом"},
        clarifications=["Нужны источники в каждом ответе"],
    )

    saved = store.save_task_state(conversation.id, state)
    restored = store.get_task_state(conversation.id)
    raw_database = (tmp_path / "chat.sqlite").read_bytes()

    assert restored.goal == "Настроить RAG-чат"
    assert restored.constraints == ["Использовать SQLite"]
    assert restored.terms == {"RAG": "поиск контекста перед ответом"}
    assert restored.clarifications == ["Нужны источники в каждом ответе"]
    assert saved.updated_at == restored.updated_at
    assert b"SQLite" in raw_database


def test_two_conversations_are_isolated(tmp_path: Path) -> None:
    store = ChatStore(tmp_path / "chat.sqlite")
    first = store.create_conversation()
    second = store.create_conversation()
    store.append_message(first.id, "user", "Только первый диалог")
    store.save_task_state(second.id, TaskState(goal="Только вторая цель"))

    assert [message.content for message in store.list_messages(first.id)] == [
        "Только первый диалог"
    ]
    assert store.list_messages(second.id) == []
    assert store.get_task_state(first.id).goal is None
    assert store.get_task_state(second.id).goal == "Только вторая цель"


def test_unknown_conversation_id_raises_domain_exception(tmp_path: Path) -> None:
    store = ChatStore(tmp_path / "chat.sqlite")
    unknown_id = "00000000-0000-0000-0000-000000000000"

    with pytest.raises(ConversationNotFoundError, match=unknown_id):
        store.get_conversation(unknown_id)
    with pytest.raises(ConversationNotFoundError):
        store.list_messages(unknown_id)
    with pytest.raises(ConversationNotFoundError):
        store.append_message(unknown_id, "user", "не создавать диалог молча")
    with pytest.raises(ConversationNotFoundError):
        store.get_task_state(unknown_id)
    with pytest.raises(ConversationNotFoundError):
        store.save_task_state(unknown_id, TaskState())
