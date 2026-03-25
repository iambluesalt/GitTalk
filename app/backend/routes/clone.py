"""
Clone route — POST /api/clone
Streams SSE events for clone progress.
"""
import asyncio
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from config import settings
from logger import logger
from models import CloneRequest, PreflightResponse, ProjectMetadata, ProjectStatus, SSEEvent
from storage.metadata_db import db
from services.clone_service import CloneService, safe_rmtree
from services.analyzer_service import AnalyzerService
from utils.validators import validate_github_url, sanitize_clone_dir_name

router = APIRouter()
clone_service = CloneService()
analyzer_service = AnalyzerService()


@router.get("/clone/preflight", response_model=PreflightResponse)
async def clone_preflight(url: str):
    """
    Check a GitHub repo via API before cloning.
    Returns metadata, size, and a tiered size warning.
    One API call, no cloning.
    """
    try:
        normalized_url, owner, repo = validate_github_url(url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    import httpx

    headers = {"Accept": "application/vnd.github.v3+json"}
    if settings.GITHUB_TOKEN:
        headers["Authorization"] = f"token {settings.GITHUB_TOKEN}"

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://api.github.com/repos/{owner}/{repo}",
                headers=headers,
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise HTTPException(status_code=404, detail="Repository not found")
        if e.response.status_code == 403:
            raise HTTPException(status_code=429, detail="GitHub API rate limit reached. Configure a GitHub token in Settings.")
        raise HTTPException(status_code=502, detail=f"GitHub API error: {e.response.status_code}")
    except httpx.RequestError:
        raise HTTPException(status_code=502, detail="Could not reach GitHub API")

    size_kb = data.get("size", 0)
    size_mb = size_kb / 1024
    max_size = settings.MAX_REPO_SIZE_MB

    # Tiered warnings
    if size_mb > max_size:
        warning = "too_large"
        message = f"Repository is {size_mb:.0f}MB — exceeds the {max_size}MB limit"
    elif size_mb > max_size * 0.6:
        warning = "large"
        message = f"Large repository ({size_mb:.0f}MB) — clone may take a while"
    elif size_mb > 100:
        warning = "medium"
        message = f"Medium repository ({size_mb:.0f}MB) — should clone in under a minute"
    else:
        warning = "ok"
        message = f"Light repository ({size_mb:.1f}MB) — clones fast"

    return PreflightResponse(
        name=data.get("name", repo),
        full_name=data.get("full_name", f"{owner}/{repo}"),
        description=data.get("description"),
        size_kb=size_kb,
        size_mb=round(size_mb, 2),
        stars=data.get("stargazers_count", 0),
        forks=data.get("forks_count", 0),
        default_branch=data.get("default_branch", "main"),
        language=data.get("language"),
        updated_at=data.get("updated_at"),
        private=data.get("private", False),
        max_size_mb=max_size,
        size_warning=warning,
        size_warning_message=message,
    )


@router.post("/clone")
async def clone_repository(request: CloneRequest):
    """
    Clone a GitHub repository with SSE progress streaming.

    Flow:
    1. Validate URL
    2. Check for duplicates
    3. Check repo size via GitHub API
    4. Check disk space
    5. Create project record (CLONING)
    6. Clone via subprocess with progress
    7. Analyze repository
    8. Update project (CLONED + analysis)
    """

    async def event_stream():
        project_id = None
        clone_dir = None

        try:
            # 1. Validate GitHub URL
            yield SSEEvent(
                event="status",
                data={"message": "Validating URL...", "phase": "validate"}
            ).format()

            try:
                normalized_url, owner, repo = validate_github_url(str(request.github_url))
            except ValueError as e:
                yield SSEEvent(event="error", data={"message": str(e)}).format()
                return

            # 2. Check for duplicates
            existing = db.get_project_by_url(normalized_url)
            if existing:
                # Auto-clean errored projects — don't block the user
                if existing.status == ProjectStatus.ERROR:
                    logger.info(f"Auto-cleaning errored project {existing.id}")
                    safe_rmtree(Path(existing.clone_path))
                    db.delete_project(existing.id)
                    existing = None
                elif not request.force:
                    yield SSEEvent(
                        event="duplicate",
                        data={
                            "message": f"Repository already cloned as '{existing.name}'",
                            "project_id": existing.id,
                            "project": existing.model_dump(mode="json"),
                        }
                    ).format()
                    return

            # If force=true and project exists, remove old data
            if existing and request.force:
                safe_rmtree(Path(existing.clone_path))
                db.delete_project(existing.id)
                logger.info(f"Force re-clone: removed existing project {existing.id}")

            # 3. Check repo size via GitHub API
            yield SSEEvent(
                event="status",
                data={"message": "Checking repository...", "phase": "check"}
            ).format()

            try:
                repo_size_kb = await clone_service.check_repo_size(
                    owner, repo, settings.GITHUB_TOKEN
                )
                repo_size_mb = repo_size_kb / 1024
                if repo_size_mb > settings.MAX_REPO_SIZE_MB:
                    yield SSEEvent(
                        event="error",
                        data={
                            "message": (
                                f"Repository too large: {repo_size_mb:.0f}MB "
                                f"(limit: {settings.MAX_REPO_SIZE_MB}MB)"
                            )
                        }
                    ).format()
                    return
            except Exception as e:
                logger.warning(f"Could not check repo size (proceeding): {e}")

            # 4. Check disk space
            free_mb, required_mb = clone_service.check_disk_space(settings.GITTALK_REPOS_DIR)
            if free_mb < required_mb:
                yield SSEEvent(
                    event="error",
                    data={
                        "message": f"Low disk space: {free_mb}MB free, {required_mb}MB required"
                    }
                ).format()
                return

            # 5. Create project record
            project_id = str(uuid.uuid4())
            dir_name = sanitize_clone_dir_name(owner, repo)
            clone_dir = settings.GITTALK_REPOS_DIR / dir_name

            project = ProjectMetadata(
                id=project_id,
                name=repo,
                github_url=normalized_url,
                clone_path=str(clone_dir),
                status=ProjectStatus.CLONING,
                cloned_at=datetime.now(),
            )
            db.create_project(project)

            # 6. Clone with progress streaming
            error_occurred = False
            async for event_str in clone_service.clone_repository(
                normalized_url, clone_dir, settings.GITHUB_TOKEN
            ):
                yield event_str
                # Check if an error event was sent
                if '"error"' in event_str and "event: error" in event_str:
                    error_occurred = True

            if error_occurred:
                db.update_project_error(project_id, "Clone failed")
                return

            # 7. Analyze repository
            yield SSEEvent(
                event="status",
                data={"message": "Analyzing repository...", "phase": "analyze"}
            ).format()

            analysis = await asyncio.to_thread(
                analyzer_service.analyze_repository, clone_dir
            )
            db.update_project_analysis(project_id, analysis)
            db.update_project_status(project_id, ProjectStatus.CLONED)

            # 8. Complete — return full project metadata
            updated_project = db.get_project(project_id)
            yield SSEEvent(
                event="complete",
                data={
                    "message": "Repository cloned and analyzed successfully",
                    "project": updated_project.model_dump(mode="json") if updated_project else {},
                }
            ).format()

        except Exception as e:
            logger.error(f"Clone stream error: {e}")
            if project_id:
                db.update_project_error(project_id, str(e))
            safe_rmtree(clone_dir)
            yield SSEEvent(
                event="error",
                data={"message": f"Unexpected error: {str(e)}"}
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
