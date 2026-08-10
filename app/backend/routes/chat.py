"""
Chat routes — POST /api/chat (SSE streaming), conversation management.
Orchestrates: agent run → LLM streaming → persistence.
"""
import json
import time
from typing import Any, AsyncGenerator

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from logger import logger
from models import (
    ChatRequest,
    ProjectStatus,
    SSEEvent,
)
from storage.metadata_db import db
from services.agent_graph import build_agent_run
from services.agent_tools import result_to_reference
from services.llm_service import llm_service, _chunk_text

router = APIRouter()

# (event_name, event_data) pairs — the caller formats them as SSE.
StreamEvents = AsyncGenerator[tuple[str, dict], None]


def _tool_artifact(output: Any) -> list:
    """Pull the raw SearchResult list out of a finished tool call."""
    artifact = getattr(output, "artifact", None)
    if artifact is None and isinstance(output, tuple) and len(output) == 2:
        artifact = output[1]
    return artifact or []


async def _run_agent(request: ChatRequest, conversation_id: str) -> StreamEvents:
    """The model searches and reads the repo on its own, possibly several times,
    before answering.

    Sources are emitted cumulatively — once before the first token (so the
    frontend's "searching" state clears exactly as it does today), and again
    whenever a later hop turns up code the client hasn't seen. Each event carries
    the full list, so the client just replaces what it holds.

    Text the model emits alongside a tool call is streamed rather than buffered:
    withholding it until the turn ends would stall the final answer too.
    """
    run = await build_agent_run(
        project_id=request.project_id,
        query=request.message,
        conversation_id=conversation_id,
        model_override=request.model,
    )

    seen_chunks: set[str] = set()
    sources: list[dict] = []
    search_calls = 0
    sources_pending = True

    def sources_event() -> tuple[str, dict]:
        return "sources", {
            "sources": list(sources),
            "search_results_count": len(sources),
            "token_count": run.token_count,
        }

    async for event in run.runnable.astream_events(
        run.inputs, config=run.config or {}, version="v2"
    ):
        kind = event["event"]

        if kind == "on_tool_end" and event.get("name") == "search_codebase":
            search_calls += 1
            for result in _tool_artifact(event["data"].get("output")):
                key = f"{result.file_path}:{result.line_start}-{result.line_end}"
                if key in seen_chunks:
                    continue
                seen_chunks.add(key)
                sources.append(result_to_reference(result).model_dump())
                sources_pending = True

        elif kind == "on_chat_model_stream":
            token = _chunk_text(event["data"].get("chunk"))
            if not token:
                continue
            if sources_pending:
                sources_pending = False
                yield sources_event()
            yield "token", {"token": token}

    # No tokens ever streamed (or only tool calls) — the client still needs a
    # sources event to leave its searching state.
    if sources_pending:
        yield sources_event()

    logger.info(
        f"Agent run finished | intent={run.intent} "
        f"searches={search_calls} sources={len(sources)}"
    )


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
            is_new_conversation = not request.conversation_id
            conversation_id = request.conversation_id
            if not conversation_id:
                conversation_id = db.create_conversation(
                    request.project_id,
                    title=request.message[:80],  # placeholder; replaced by LLM title below
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

            # --- Stream the answer through the agent ---
            sources_data: list[dict] = []
            full_response: list[str] = []

            async for event, data in _run_agent(request, conversation_id):
                if event == "sources":
                    sources_data = data["sources"]
                elif event == "token":
                    full_response.append(data["token"])
                yield SSEEvent(event=event, data=data).format()

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

            # --- Generate conversation title for new chats (fast model) ---
            conversation_title: str | None = None
            if is_new_conversation:
                conversation_title = await llm_service.generate_title(request.message)
                db.update_conversation_title(conversation_id, conversation_title)
                logger.debug(f"Generated title: {conversation_title!r}")

            # --- Done event ---
            done_data: dict = {
                "conversation_id": conversation_id,
                "response_time_ms": round(elapsed_ms, 1),
            }
            if conversation_title:
                done_data["conversation_title"] = conversation_title

            yield SSEEvent(event="done", data=done_data).format()

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
