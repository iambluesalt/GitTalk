"""
Index route — POST /api/index
Streams SSE events for indexing progress.
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from logger import logger
from models import IndexRequest, ProjectStatus, SSEEvent
from storage.metadata_db import db
from services.indexing_service import indexing_service

router = APIRouter()


@router.post("/index")
async def index_project(request: IndexRequest):
    """
    Index a cloned project with SSE progress streaming.

    Parses code with tree-sitter, chunks at AST boundaries,
    generates embeddings via Ollama, and stores in LanceDB.
    Supports incremental re-indexing (only changed files).
    """
    # Validate project exists
    project = db.get_project(request.project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Validate project status
    allowed = {ProjectStatus.CLONED, ProjectStatus.INDEXED, ProjectStatus.ERROR}
    if project.status not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Project status is '{project.status.value}', "
                   f"must be one of: {', '.join(s.value for s in allowed)}",
        )

    async def event_stream():
        try:
            async for event_str in indexing_service.index_project(
                project_id=request.project_id,
                clone_path=project.clone_path,
                force_reindex=request.force_reindex,
            ):
                yield event_str
        except Exception as e:
            logger.error(f"Index stream error: {e}")
            yield SSEEvent(
                event="error",
                data={"message": f"Unexpected error: {str(e)}"},
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
