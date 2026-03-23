"""
Chat routes — POST /api/chat (SSE streaming), conversation management.
Orchestrates: RAG context → LLM streaming → persistence.
"""
import json
import time

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from logger import logger
from models import (
    ChatRequest,
    CodeReference,
    ProjectStatus,
    SSEEvent,
)
from storage.metadata_db import db
from services.rag_service import rag_service
from services.llm_service import llm_service

router = APIRouter()


# ========================================================================
# GET /api/models — list available chat models
# ========================================================================

@router.get("/models")
async def list_models():
    """List available chat models from all configured providers."""
    models = await llm_service.list_models()
    return {"models": models}


# ========================================================================
# POST /api/chat — main chat endpoint (SSE streaming)
# ========================================================================

@router.post("/chat")
async def chat(request: ChatRequest):
    """
    Send a message and receive a streaming LLM response.

    SSE event types:
      - sources:  code references used for context (sent first)
      - token:    individual LLM response tokens
      - done:     final event with metadata (conversation_id, response_time_ms)
      - error:    error description
    """

    async def event_stream():
        start = time.perf_counter()

        try:
            # --- Validate project ---
            project = db.get_project(request.project_id)
            if not project:
                yield SSEEvent(
                    event="error",
                    data={"message": "Project not found"},
                ).format()
                return

            if project.status != ProjectStatus.INDEXED:
                yield SSEEvent(
                    event="error",
                    data={
                        "message": f"Project must be indexed before chatting (current status: {project.status.value})"
                    },
                ).format()
                return

            # --- Resolve or create conversation ---
            conversation_id = request.conversation_id
            if not conversation_id:
                conversation_id = db.create_conversation(
                    request.project_id,
                    title=request.message[:80],
                )

            # Verify conversation exists (if user-provided)
            elif not db.get_conversation(conversation_id):
                yield SSEEvent(
                    event="error",
                    data={"message": "Conversation not found"},
                ).format()
                return

            # --- Persist user message ---
            db.add_message(conversation_id, "user", request.message)

            # --- Build RAG context ---
            rag_context = await rag_service.build_context(
                project_id=request.project_id,
                query=request.message,
                conversation_id=conversation_id,
            )

            # --- Send sources to client ---
            sources_data = [s.model_dump() for s in rag_context.sources]
            yield SSEEvent(
                event="sources",
                data={
                    "sources": sources_data,
                    "search_results_count": rag_context.search_results_count,
                    "token_count": rag_context.token_count,
                },
            ).format()

            # --- Stream LLM response ---
            full_response: list[str] = []
            async for token in llm_service.stream_chat(
                rag_context.messages, model_override=request.model
            ):
                full_response.append(token)
                yield SSEEvent(event="token", data={"token": token}).format()

            # --- Persist assistant response ---
            response_text = "".join(full_response)
            elapsed_ms = (time.perf_counter() - start) * 1000

            db.add_message(
                conversation_id,
                "assistant",
                response_text,
                sources_json=json.dumps(sources_data) if sources_data else None,
                response_time_ms=elapsed_ms,
            )

            # --- Done event ---
            yield SSEEvent(
                event="done",
                data={
                    "conversation_id": conversation_id,
                    "response_time_ms": round(elapsed_ms, 1),
                },
            ).format()

        except Exception as e:
            logger.error(f"Chat stream error: {e}", exc_info=True)
            yield SSEEvent(
                event="error",
                data={"message": str(e)},
            ).format()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ========================================================================
# Conversation management endpoints
# ========================================================================

@router.get("/conversations/{project_id}")
async def list_conversations(
    project_id: str,
    limit: int = Query(50, ge=1, le=200),
):
    """List all conversations for a project."""
    project = db.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    conversations = db.list_conversations(project_id, limit)
    return {"conversations": conversations, "total": len(conversations)}


@router.get("/conversations/{project_id}/{conversation_id}")
async def get_conversation(project_id: str, conversation_id: str):
    """Get a conversation with its messages."""
    conv = db.get_conversation(conversation_id)
    if not conv or conv["project_id"] != project_id:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages = db.get_conversation_messages(conversation_id)
    # Parse sources_json for each message
    for msg in messages:
        if msg.get("sources_json"):
            try:
                msg["sources"] = json.loads(msg["sources_json"])
            except (json.JSONDecodeError, TypeError):
                msg["sources"] = []
        else:
            msg["sources"] = []
        msg.pop("sources_json", None)

    return {
        "conversation": conv,
        "messages": messages,
    }


@router.delete("/conversations/{project_id}/{conversation_id}")
async def delete_conversation(project_id: str, conversation_id: str):
    """Delete a conversation and its messages."""
    conv = db.get_conversation(conversation_id)
    if not conv or conv["project_id"] != project_id:
        raise HTTPException(status_code=404, detail="Conversation not found")

    db.delete_conversation(conversation_id)
    return {"success": True, "message": "Conversation deleted"}
