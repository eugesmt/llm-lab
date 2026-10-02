"""SQLite persistence for chat history and task memory.

The store deliberately uses the standard-library ``sqlite3`` module.  It is
small enough for a local service and keeps the persistence contract explicit:
the document index has its own SQLite database, while this module stores only
conversation state and messages.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from app.schemas.conversation import ChatMessage, Conversation, TaskState


class ConversationNotFoundError(LookupError):
    """Raised when a conversation-scoped operation receives an unknown ID."""

    def __init__(self, conversation_id: str) -> None:
        super().__init__(f"Conversation not found: {conversation_id}")
        self.conversation_id = conversation_id


class ChatStore:
    """Synchronous SQLite store for conversations, messages and task state."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)

        database = ":memory:" if str(self.path) == ":memory:" else str(self.path)
        self._connection = sqlite3.connect(database)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        if str(self.path) != ":memory:":
            self._connection.execute("PRAGMA journal_mode = WAL")
        self._create_schema()

    def close(self) -> None:
        """Close the database connection."""

        self._connection.close()

    def __enter__(self) -> ChatStore:
        return self

    def __exit__(self, _exc_type: Any, _exc_value: Any, _traceback: Any) -> None:
        self.close()

    def create_conversation(self) -> Conversation:
        """Create and persist an empty conversation with a UUID string ID."""

        now = _utc_now()
        conversation = Conversation(id=str(uuid4()), created_at=now, updated_at=now)
        self._connection.execute(
            "INSERT INTO conversations (id, created_at, updated_at) VALUES (?, ?, ?)",
            (conversation.id, _encode_datetime(now), _encode_datetime(now)),
        )
        self._connection.execute(
            """
            INSERT INTO task_states
                (conversation_id, goal, constraints_json, terms_json,
                 clarifications_json, updated_at)
            VALUES (?, NULL, ?, ?, ?, ?)
            """,
            (
                conversation.id,
                _encode_json([]),
                _encode_json({}),
                _encode_json([]),
                _encode_datetime(now),
            ),
        )
        self._connection.commit()
        return conversation

    def get_conversation(self, conversation_id: str) -> Conversation:
        """Return a conversation or raise a domain-specific exception."""

        row = self._connection.execute(
            "SELECT id, created_at, updated_at FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
        if row is None:
            raise ConversationNotFoundError(conversation_id)
        return Conversation(
            id=row["id"],
            created_at=_decode_datetime(row["created_at"]),
            updated_at=_decode_datetime(row["updated_at"]),
        )

    def list_messages(self, conversation_id: str, limit: int = 10) -> list[ChatMessage]:
        """Return the newest ``limit`` messages in chronological order."""

        self._ensure_conversation(conversation_id)
        if limit < 0:
            raise ValueError("limit must be non-negative")
        rows = self._connection.execute(
            """
            SELECT id, conversation_id, role, content, created_at, response_json
            FROM messages
            WHERE conversation_id = ?
            ORDER BY sequence DESC
            LIMIT ?
            """,
            (conversation_id, limit),
        ).fetchall()
        return [_message_from_row(row) for row in reversed(rows)]

    def append_message(
        self,
        conversation_id: str,
        role: Literal["user", "assistant"],
        content: str,
        response_json: dict[str, Any] | None = None,
    ) -> ChatMessage:
        """Append a message and update the conversation timestamp atomically."""

        self._ensure_conversation(conversation_id)
        if role not in ("user", "assistant"):
            raise ValueError("role must be 'user' or 'assistant'")
        if not isinstance(content, str):
            raise TypeError("content must be a string")
        if response_json is not None and not isinstance(response_json, dict):
            raise TypeError("response_json must be a JSON object or None")

        now = _utc_now()
        message = ChatMessage(
            id=str(uuid4()),
            conversation_id=conversation_id,
            role=role,
            content=content,
            created_at=now,
            response_json=response_json,
        )
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO messages
                    (id, conversation_id, role, content, created_at, response_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    message.id,
                    message.conversation_id,
                    message.role,
                    message.content,
                    _encode_datetime(message.created_at),
                    _encode_json(response_json) if response_json is not None else None,
                ),
            )
            self._connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (_encode_datetime(now), conversation_id),
            )
        return message

    def get_task_state(self, conversation_id: str) -> TaskState:
        """Return the persisted task memory for a conversation."""

        self._ensure_conversation(conversation_id)
        row = self._connection.execute(
            """
            SELECT goal, constraints_json, terms_json, clarifications_json, updated_at
            FROM task_states
            WHERE conversation_id = ?
            """,
            (conversation_id,),
        ).fetchone()
        if row is None:
            # This keeps the method resilient to a database created by an older
            # version, without ever creating an unknown conversation silently.
            state = TaskState()
            self._write_task_state(conversation_id, state)
            return state
        return _task_state_from_row(row)

    def save_task_state(self, conversation_id: str, state: TaskState) -> TaskState:
        """Persist and return task memory with a fresh update timestamp."""

        self._ensure_conversation(conversation_id)
        saved_state = state.model_copy(update={"updated_at": _utc_now()})
        self._write_task_state(conversation_id, saved_state)
        return saved_state

    def _create_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT NOT NULL UNIQUE,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                response_json TEXT,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                    ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_messages_conversation_sequence
                ON messages (conversation_id, sequence);

            CREATE TABLE IF NOT EXISTS task_states (
                conversation_id TEXT PRIMARY KEY,
                goal TEXT,
                constraints_json TEXT NOT NULL,
                terms_json TEXT NOT NULL,
                clarifications_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                    ON DELETE CASCADE
            );
            """
        )
        self._connection.commit()

    def _ensure_conversation(self, conversation_id: str) -> None:
        exists = self._connection.execute(
            "SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
        if exists is None:
            raise ConversationNotFoundError(conversation_id)

    def _write_task_state(self, conversation_id: str, state: TaskState) -> None:
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO task_states
                    (conversation_id, goal, constraints_json, terms_json,
                     clarifications_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(conversation_id) DO UPDATE SET
                    goal = excluded.goal,
                    constraints_json = excluded.constraints_json,
                    terms_json = excluded.terms_json,
                    clarifications_json = excluded.clarifications_json,
                    updated_at = excluded.updated_at
                """,
                (
                    conversation_id,
                    state.goal,
                    _encode_json(state.constraints),
                    _encode_json(state.terms),
                    _encode_json(state.clarifications),
                    _encode_datetime(state.updated_at),
                ),
            )


def _message_from_row(row: sqlite3.Row) -> ChatMessage:
    response_json = row["response_json"]
    return ChatMessage(
        id=row["id"],
        conversation_id=row["conversation_id"],
        role=row["role"],
        content=row["content"],
        created_at=_decode_datetime(row["created_at"]),
        response_json=json.loads(response_json) if response_json is not None else None,
    )


def _task_state_from_row(row: sqlite3.Row) -> TaskState:
    return TaskState(
        goal=row["goal"],
        constraints=json.loads(row["constraints_json"]),
        terms=json.loads(row["terms_json"]),
        clarifications=json.loads(row["clarifications_json"]),
        updated_at=_decode_datetime(row["updated_at"]),
    )


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _encode_datetime(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="microseconds")


def _decode_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _encode_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
