"""
Project management routes.
GET /api/projects — list projects
GET /api/projects/{project_id} — get project details
DELETE /api/projects — delete ALL projects
DELETE /api/projects/{project_id} — delete single project
"""
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from config import settings
from logger import logger
from models import ProjectListResponse, ProjectMetadata
from storage.metadata_db import db
from storage.vector_db import vector_db
from services.clone_service import safe_rmtree

router = APIRouter()


@router.get("/projects", response_model=ProjectListResponse)
async def list_projects(limit: int = Query(50, ge=1, le=200)):
    """List all projects with optional limit."""
    projects = db.list_projects(limit=limit)
    recent = db.get_recent_projects(limit=5)
    return ProjectListResponse(
        projects=projects,
        total=len(projects),
        recent_projects=recent,
    )


@router.get("/projects/{project_id}", response_model=ProjectMetadata)
async def get_project(project_id: str):
    """Get a single project by ID."""
    project = db.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@router.delete("/projects")
async def delete_all_projects():
    """Delete ALL projects, their clone directories, and vector DB collections."""
    projects = db.list_projects(limit=9999)
    if not projects:
        return {"success": True, "message": "No projects to delete", "deleted_count": 0}

    for project in projects:
        safe_rmtree(Path(project.clone_path))

        try:
            vector_db.delete_table(project.id)
        except Exception as e:
            logger.warning(f"Failed to delete vector table for {project.id}: {e}")

    count = db.delete_all_projects()

    # Also clean up the repos directory of any orphaned dirs
    repos_dir = settings.GITTALK_REPOS_DIR
    if repos_dir.exists():
        for child in repos_dir.iterdir():
            if child.is_dir():
                safe_rmtree(child)

    logger.info(f"Deleted all projects: {count} removed")
    return {"success": True, "message": f"Deleted {count} project(s)", "deleted_count": count}


@router.delete("/projects/{project_id}")
async def delete_project(project_id: str):
    """Delete a project, its clone directory, and vector DB collection."""
    project = db.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    safe_rmtree(Path(project.clone_path))
    logger.info(f"Removed clone directory: {project.clone_path}")

    try:
        vector_db.delete_table(project_id)
    except Exception as e:
        logger.warning(f"Failed to delete vector table: {e}")

    db.delete_project(project_id)

    return {"success": True, "message": f"Project '{project.name}' deleted"}
