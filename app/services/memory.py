"""Deterministic task memory for history-aware RAG conversations.

The service deliberately does not call an LLM.  It only extracts explicit
markers from a user's message and prepares a compact context for the actual
answering model.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from app.schemas.conversation import TaskState


class TaskMemoryService:
    """Maintain a small, deterministic state describing the user's task."""

    MAX_GOAL_LENGTH = 500
    MAX_ITEM_LENGTH = 300
    MAX_COLLECTION_SIZE = 20
    MAX_RECENT_MESSAGES = 10
    MAX_HISTORY_ITEM_LENGTH = 500

    _GOAL_PATTERNS = (
        re.compile(r"(?:моя\s+)?цель\s*:\s*(.+?)(?=\s*(?:[.!?;]|$))", re.IGNORECASE),
        re.compile(r"моя\s+цель\s*(?:—|-|:)\s*(.+?)(?=\s*(?:[.!?;]|$))", re.IGNORECASE),
        re.compile(r"\bхочу\s+(.+?)(?=\s*(?:[.!?;]|$))", re.IGNORECASE),
        re.compile(r"^\s*нужно\s+(.+?)(?=\s*(?:[.!?;]|$))", re.IGNORECASE),
    )
    _CONSTRAINT_PATTERNS = (
        re.compile(r"ограничение\s*:\s*(.+?)(?=\s*(?:[.!?;]|$))", re.IGNORECASE),
        re.compile(r"(\bтолько\s+.+?)(?=\s*(?:[.!?;]|$))", re.IGNORECASE),
        re.compile(r"(\bбез\s+.+?)(?=\s*(?:[.!?;]|$))", re.IGNORECASE),
        re.compile(
            r"(\bне\s+использовать\s+.+?)(?=\s*(?:[.!?;]|$))", re.IGNORECASE
        ),
        re.compile(r"(\bобязательно\s+.+?)(?=\s*(?:[.!?;]|$))", re.IGNORECASE),
    )
    _TERM_EQUALS = re.compile(
        r"термин\s*:\s*([^=\n]+?)\s*=\s*(.+?)(?=\s*(?:[.!?;]|$))", re.IGNORECASE
    )
    _TERM_UNDERSTAND = re.compile(
        r"под\s+(.+?)\s+я\s+понимаю\s+(.+?)(?=\s*(?:[.!?;]|$))", re.IGNORECASE
    )
    _TERM_MEANS = re.compile(
        r"(.+?)\s+означает\s+(.+?)(?=\s*(?:[.!?;]|$))", re.IGNORECASE
    )
    _CLARIFICATION = re.compile(
        r"уточнение\s*:\s*(.+?)(?=\s*(?:[.!?;]|$))", re.IGNORECASE
    )
    _FOLLOW_UP_PRONOUNS = re.compile(
        r"\b(?:это|этого|этому|его|её|ее|их|там|туда|здесь|такой|такая|такое|они|он|она)\b",
        re.IGNORECASE,
    )
    _RECALL_REQUEST = re.compile(
        r"\b(?:напомни|какая\s+у\s+меня|какие\s+у\s+меня|что\s+я\s+уточнил)\b",
        re.IGNORECASE,
    )

    def update(self, state: TaskState | None, user_message: str) -> TaskState:
        """Return a new state after extracting explicit facts from a message.

        The input model and all of its collections remain untouched.  Repeated
        values are ignored while the first-seen order is retained.
        """

        current = state or self._empty_state()
        goal = self._extract_goal(user_message) or self._value(current, "goal")

        constraints = self._append_unique(
            self._collection(current, "constraints"),
            self._extract_all(self._CONSTRAINT_PATTERNS, user_message),
        )
        clarifications = self._append_unique(
            self._collection(current, "clarifications"),
            self._extract_all((self._CLARIFICATION,), user_message),
        )
        terms = self._merge_terms(
            self._mapping(current, "terms"),
            self._extract_terms(user_message),
        )

        return self._copy_state(
            current,
            goal=self._clip(goal, self.MAX_GOAL_LENGTH) if goal else None,
            constraints=constraints,
            terms=terms,
            clarifications=clarifications,
            updated_at=datetime.now(UTC),
        )

    def build_context(
        self,
        state: TaskState | None,
        recent_messages: Sequence[Any] | None,
    ) -> str:
        """Build a compact Russian memory block for an answering prompt."""

        current = state or self._empty_state()
        lines = ["ПАМЯТЬ ЗАДАЧИ:"]
        goal = self._value(current, "goal")
        lines.append(f"Цель: {goal or 'не зафиксирована'}")

        lines.append("Ограничения:")
        constraints = self._collection(current, "constraints")
        lines.extend(f"- {item}" for item in constraints or ["не зафиксированы"])

        lines.append("Термины:")
        terms = self._mapping(current, "terms")
        lines.extend(
            f"- {key}: {value}" for key, value in terms.items()
        )
        if not terms:
            lines.append("- не зафиксированы")

        lines.append("Уточнения:")
        clarifications = self._collection(current, "clarifications")
        lines.extend(f"- {item}" for item in clarifications or ["не зафиксированы"])

        lines.append("Недавняя история:")
        normalized = self._normalize_messages(recent_messages or [])[-self.MAX_RECENT_MESSAGES :]
        if normalized:
            lines.extend(
                f"{role}: {self._clip(content, self.MAX_HISTORY_ITEM_LENGTH)}"
                for role, content in normalized
            )
        else:
            lines.append("- отсутствует")

        return "\n".join(lines)

    def rewrite_query(
        self,
        question: str,
        state: TaskState | None,
        recent_messages: Sequence[Any] | None,
    ) -> str:
        """Turn a short conversational follow-up into a standalone query.

        Substantive questions are returned unchanged.  Only the current user
        message, previous user messages, and explicit task state are used;
        assistant responses are never treated as remembered facts here.
        """

        cleaned_question = question.strip()
        current = state or self._empty_state()
        word_count = len(re.findall(r"[\wа-яё-]+", cleaned_question.lower()))
        needs_task_context = bool(self._value(current, "goal") and word_count <= 8)
        if not self._is_follow_up(cleaned_question) and not needs_task_context:
            return cleaned_question

        parts = [cleaned_question]
        goal = self._value(current, "goal")
        if goal:
            parts.append(f"Цель задачи: {goal}")

        terms = self._mapping(current, "terms")
        if terms:
            parts.append(
                "Термины: "
                + "; ".join(f"{key} = {value}" for key, value in terms.items())
            )

        constraints = self._collection(current, "constraints")
        if constraints:
            parts.append("Ограничения: " + "; ".join(constraints))

        previous_user = [
            content
            for role, content in self._normalize_messages(recent_messages or [])
            if role == "Пользователь"
        ]
        if previous_user:
            parts.append(f"Последний вопрос пользователя: {previous_user[-1]}")

        return " ".join(parts)

    @classmethod
    def is_task_state_recall(cls, message: str) -> bool:
        """Return whether the user asks to repeat remembered task state."""

        lower = message.lower()
        state_terms = ("цел", "огранич", "термин", "уточнен", "памят")
        return bool(cls._RECALL_REQUEST.search(message)) and any(
            term in lower for term in state_terms
        )

    @staticmethod
    def is_task_state_update(message: str) -> bool:
        """Return whether a message only records task-state information."""

        if "?" in message:
            return False
        return bool(
            re.match(
                r"^\s*(?:цель|моя\s+цель|ограничение|термин|уточнение)\s*[:—-]",
                message,
                re.IGNORECASE,
            )
        )

    def _empty_state(self) -> TaskState:
        return TaskState(
            goal=None,
            constraints=[],
            terms={},
            clarifications=[],
            updated_at=datetime.now(UTC),
        )

    @classmethod
    def _copy_state(cls, state: TaskState, **updates: Any) -> TaskState:
        if hasattr(state, "model_copy"):
            return state.model_copy(deep=True, update=updates)
        raise TypeError("TaskState must be a Pydantic model with model_copy")

    @classmethod
    def _extract_goal(cls, message: str) -> str | None:
        for pattern in cls._GOAL_PATTERNS:
            match = pattern.search(message)
            if match:
                return cls._clean_clause(match.group(1), cls.MAX_GOAL_LENGTH)
        return None

    @classmethod
    def _extract_all(cls, patterns: Iterable[re.Pattern[str]], message: str) -> list[str]:
        values: list[str] = []
        for pattern in patterns:
            values.extend(
                cls._clean_clause(match.group(1), cls.MAX_ITEM_LENGTH)
                for match in pattern.finditer(message)
            )
        return [value for value in values if value]

    @classmethod
    def _extract_terms(cls, message: str) -> list[tuple[str, str]]:
        result: list[tuple[str, str]] = []
        for pattern in (cls._TERM_EQUALS, cls._TERM_UNDERSTAND, cls._TERM_MEANS):
            for match in pattern.finditer(message):
                key = cls._clean_clause(match.group(1), cls.MAX_ITEM_LENGTH)
                value = cls._clean_clause(match.group(2), cls.MAX_ITEM_LENGTH)
                if key and value:
                    result.append((key, value))
        return result

    @classmethod
    def _merge_terms(
        cls,
        existing: Mapping[str, str],
        additions: Iterable[tuple[str, str]],
    ) -> dict[str, str]:
        result = dict(existing)
        keys = {key.casefold(): key for key in result}
        for key, value in additions:
            old_key = keys.get(key.casefold())
            if old_key is None:
                if len(result) >= cls.MAX_COLLECTION_SIZE:
                    continue
                result[key] = value
                keys[key.casefold()] = key
            else:
                # A repeated explicit definition updates the value but keeps its
                # original insertion position.
                result[old_key] = value
        return result

    @classmethod
    def _append_unique(cls, existing: Iterable[str], additions: Iterable[str]) -> list[str]:
        result: list[str] = []
        for value in (*existing, *additions):
            cleaned = cls._clean_clause(value, cls.MAX_ITEM_LENGTH)
            marker = cleaned.casefold()
            if not cleaned:
                continue
            marker_tokens = set(re.findall(r"[\wа-яё-]+", marker))
            overlapping_index = next(
                (
                    index
                    for index, current in enumerate(result)
                    if marker_tokens
                    and (
                        marker_tokens
                        <= set(re.findall(r"[\wа-яё-]+", current.casefold()))
                        or set(re.findall(r"[\wа-яё-]+", current.casefold()))
                        <= marker_tokens
                    )
                ),
                None,
            )
            if overlapping_index is None:
                result.append(cleaned)
            elif len(cleaned) > len(result[overlapping_index]):
                result[overlapping_index] = cleaned
            if len(result) >= cls.MAX_COLLECTION_SIZE:
                break
        return result

    @classmethod
    def _is_follow_up(cls, question: str) -> bool:
        words = re.findall(r"[\wа-яё-]+", question.lower())
        if not words:
            return False
        if cls._FOLLOW_UP_PRONOUNS.search(question):
            return True
        # A short question beginning with «а» or «и» normally depends on the
        # active topic: «А что с Redis?», «А где ключи?», «И что дальше?».
        if len(words) <= 6 and re.match(r"^(?:а|и)\b", question, re.IGNORECASE):
            return True
        return question.lower().strip() in {"а дальше?", "и дальше?", "что дальше?"}

    @classmethod
    def _normalize_messages(cls, messages: Sequence[Any]) -> list[tuple[str, str]]:
        normalized: list[tuple[str, str]] = []
        for message in messages:
            if isinstance(message, Mapping):
                role = message.get("role", "")
                content = message.get("content", "")
            else:
                role = getattr(message, "role", "")
                content = getattr(message, "content", "")
            if not isinstance(content, str) or not content.strip():
                continue
            role_text = str(role).lower()
            display_role = (
                "Ассистент" if role_text in {"assistant", "ai", "bot"} else "Пользователь"
            )
            normalized.append((display_role, content.strip()))
        return normalized

    @staticmethod
    def _value(state: TaskState, field: str) -> str | None:
        value = getattr(state, field, None)
        return value if isinstance(value, str) else None

    @staticmethod
    def _collection(state: TaskState, field: str) -> list[str]:
        value = getattr(state, field, None)
        return list(value) if value else []

    @staticmethod
    def _mapping(state: TaskState, field: str) -> dict[str, str]:
        value = getattr(state, field, None)
        return dict(value) if isinstance(value, Mapping) else {}

    @classmethod
    def _clean_clause(cls, value: str, limit: int) -> str:
        return cls._clip(value.strip().strip("-—:;,.!? "), limit)

    @staticmethod
    def _clip(value: str, limit: int) -> str:
        return value[:limit].rstrip()
