"""Chat and conversation endpoints."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import (
    get_chat_store,
    get_llm_service,
    get_task_memory_service,
)
from app.schemas.chat import ChatRequest, ChatResponse, ConversationResponse
from app.schemas.llm import LLMResponse, RAGCitation, RAGSource
from app.services.llm import LLMService
from app.services.memory import TaskMemoryService
from app.storage import ChatStore, ConversationNotFoundError

router = APIRouter(prefix="/api", tags=["chat"])
CONVERSATION_HISTORY_LIMIT = 200


def _not_found(exc: ConversationNotFoundError) -> HTTPException:
    return HTTPException(status_code=404, detail=str(exc))


def _store_lock(store: ChatStore):
    """Use the lock installed by the app factory, if the store has one."""

    return getattr(store, "_lock", nullcontext())


def _task_state_response(
    state,
    requested_model: str,
    rag_mode: str,
    finish_reason: str,
) -> LLMResponse:
    constraints = ", ".join(state.constraints) if state.constraints else "не зафиксированы"
    terms = (
        ", ".join(f"{key} = {value}" for key, value in state.terms.items())
        if state.terms
        else "не зафиксированы"
    )
    clarifications = (
        ", ".join(state.clarifications) if state.clarifications else "не зафиксированы"
    )
    quote = (
        f"Цель: {state.goal or 'не зафиксирована'}; "
        f"Ограничения: {constraints}; Термины: {terms}; "
        f"Уточнения: {clarifications}."
    )
    answer = (
        (
            "Память задачи обновлена.\n\n"
            if finish_reason == "task_state_updated"
            else "Текущее состояние задачи:\n\n"
        )
        + f"**Цель:** {state.goal or 'не зафиксирована'}.\n\n"
        f"**Ограничения:** {constraints}.\n\n"
        f"**Термины:** {terms}.\n\n"
        f"**Уточнения:** {clarifications}."
    )
    source = RAGSource(
        chunk_id="task_state",
        source="Память задачи",
        section="Текущее состояние диалога",
        similarity=1.0,
        rerank_score=1.0,
    )
    citation = RAGCitation(
        source=source.source,
        section=source.section,
        chunk_id=source.chunk_id,
        quote=quote,
    )
    return LLMResponse(
        provider="memory",
        model=requested_model,
        answer=answer,
        latency_ms=0.0,
        finish_reason=finish_reason,
        rag_mode=rag_mode,
        rag_sources=[source],
        citations=[citation],
        grounded=True,
    )


@router.post("/conversations", response_model=None, status_code=201)
async def create_conversation(
    store: Annotated[ChatStore, Depends(get_chat_store)],
):
    """Create an empty conversation with its initial task state."""

    with _store_lock(store):
        return store.create_conversation()


@router.get("/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: str,
    store: Annotated[ChatStore, Depends(get_chat_store)],
) -> ConversationResponse:
    """Return persisted messages and task memory for one conversation."""

    try:
        with _store_lock(store):
            conversation = store.get_conversation(conversation_id)
            # Retrieval prompts use only the last 10 messages below, while the
            # UI must be able to restore complete 10–15-turn demo scenarios.
            messages = store.list_messages(
                conversation_id,
                limit=CONVERSATION_HISTORY_LIMIT,
            )
            task_state = store.get_task_state(conversation_id)
    except ConversationNotFoundError as exc:
        raise _not_found(exc) from exc

    return ConversationResponse(
        conversation=conversation,
        messages=messages,
        task_state=task_state,
    )


@router.post(
    "/chat",
    # The endpoint has two wire shapes: legacy stateless LLMResponse and the
    # memory-aware response with conversation fields. Serialize explicitly so
    # the legacy shape remains byte-for-byte compatible.
    response_model=None,
)
async def chat(
    request: ChatRequest,
    service: Annotated[LLMService, Depends(get_llm_service)],
    store: Annotated[ChatStore | None, Depends(get_chat_store)] = None,
    memory: Annotated[TaskMemoryService | None, Depends(get_task_memory_service)] = None,
) -> LLMResponse | ChatResponse:
    """Answer one prompt, optionally continuing a persisted conversation."""

    if request.conversation_id is None:
        response = await service.chat(
            model=request.model,
            prompt=request.prompt,
            reasoning_effort=request.reasoning_effort,
            use_taskmanager_mcp=request.use_taskmanager_mcp,
            mcp_servers=request.mcp_servers,
            rag_mode=request.rag_mode,
        )
        return response

    # FastAPI always injects both dependencies.  The optional defaults retain
    # compatibility with direct stateless calls used by existing unit tests.
    if store is None or memory is None:
        raise RuntimeError("Conversation dependencies are not configured")

    conversation_id = request.conversation_id
    try:
        # Validate the ID before doing any memory or provider work. This keeps
        # a typo from silently creating a new conversation.
        with _store_lock(store):
            store.get_conversation(conversation_id)
            recent_messages = store.list_messages(conversation_id, limit=10)
            current_state = store.get_task_state(conversation_id)
    except ConversationNotFoundError as exc:
        raise _not_found(exc) from exc

    updated_state = memory.update(current_state, request.prompt)
    if memory.is_task_state_recall(request.prompt):
        response = _task_state_response(
            updated_state,
            request.model,
            request.rag_mode,
            "task_state_recall",
        )
    elif memory.is_task_state_update(request.prompt):
        response = _task_state_response(
            updated_state,
            request.model,
            request.rag_mode,
            "task_state_updated",
        )
    else:
        retrieval_query = memory.rewrite_query(
            request.prompt,
            updated_state,
            recent_messages,
        )
        # Previous assistant answers stay persisted and visible in the UI, but
        # they are not factual evidence for the next RAG answer.  Otherwise a
        # model can repeat an old source that was not retrieved for this turn.
        user_messages = [message for message in recent_messages if message.role == "user"]
        conversation_context = memory.build_context(updated_state, user_messages)

        response = await service.chat(
            model=request.model,
            prompt=request.prompt,
            reasoning_effort=request.reasoning_effort,
            use_taskmanager_mcp=request.use_taskmanager_mcp,
            mcp_servers=request.mcp_servers,
            rag_mode=request.rag_mode,
            retrieval_query=retrieval_query,
            conversation_context=conversation_context,
        )

    chat_response = ChatResponse.model_validate(
        {
            **response.model_dump(),
            "conversation_id": conversation_id,
            "task_state": updated_state.model_dump(mode="json"),
        }
    )
    # Store the complete provider-independent response so sources, citations,
    # grounding and unknown status survive a later GET request.
    with _store_lock(store):
        saved_state = store.save_task_state(conversation_id, updated_state)
        chat_response = chat_response.model_copy(
            update={"task_state": saved_state}
        )
        store.append_message(conversation_id, "user", request.prompt)
        store.append_message(
            conversation_id,
            "assistant",
            response.answer,
            response_json=chat_response.model_dump(mode="json"),
        )
    return chat_response
