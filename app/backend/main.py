"""
GitTalk Backend - FastAPI Application
Main entry point for the backend API server.
"""
import shutil

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
import traceback

from config import settings
from logger import logger
from models import HealthResponse, ErrorResponse

# Initialize FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="AI-powered codebase analysis with GitHub integration",
    version="1.0.0",
    debug=settings.DEBUG
)


# ========================================================================
# Security Middleware
# ========================================================================

ALLOWED_HOSTS = {"localhost", "127.0.0.1", "[::1]"}


@app.middleware("http")
async def host_header_validation(request: Request, call_next):
    """Reject requests with non-localhost Host headers (DNS rebinding protection)."""
    host = request.headers.get("host", "")
    # Strip port
    hostname = host.split(":")[0]
    if hostname not in ALLOWED_HOSTS:
        logger.warning(f"Blocked request with Host header: {host}")
        return JSONResponse(
            status_code=403,
            content={"error": "Forbidden", "detail": "Invalid Host header"}
        )
    return await call_next(request)


ALLOWED_ORIGINS = {*settings.CORS_ORIGINS, "http://localhost:8000", "http://127.0.0.1:8000"}


@app.middleware("http")
async def origin_header_validation(request: Request, call_next):
    """Reject requests with unexpected Origin headers."""
    origin = request.headers.get("origin")
    if origin and origin not in ALLOWED_ORIGINS:
        logger.warning(f"Blocked request with Origin header: {origin}")
        return JSONResponse(
            status_code=403,
            content={"error": "Forbidden", "detail": "Invalid Origin header"}
        )
    return await call_next(request)


# Configure CORS (after custom middleware so CORS headers are added properly)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception Handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {exc}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc) if settings.DEBUG else None
        ).model_dump()
    )


# Startup & Shutdown Events
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info(f"Starting {settings.APP_NAME}...")
    logger.info(f"Data directory: {settings.DATA_DIR}")
    logger.info(f"Repos directory: {settings.GITTALK_REPOS_DIR}")
    logger.info(f"LLM Provider: {settings.LLM_PROVIDER}")

    # Check git is available
    git_path = shutil.which("git")
    if git_path:
        logger.info(f"git found at: {git_path}")
    else:
        logger.error("git not found in PATH — clone functionality will not work")

    # Test database connections
    try:
        from storage.metadata_db import db
        from storage.vector_db import vector_db

        logger.info("Database connections initialized")

        # Purge orphaned vector tables (left behind by partial deletes or crashes)
        project_ids = {p.id for p in db.list_projects(limit=9999)}
        orphans = vector_db.purge_orphaned_tables(project_ids)
        if orphans:
            logger.info(f"Cleaned up {len(orphans)} orphaned vector table(s)")
    except Exception as e:
        logger.error(f"Failed to initialize databases: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info(f"Shutting down {settings.APP_NAME}...")


# Health Check Endpoint
@app.get(f"{settings.API_PREFIX}/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint to verify service status.

    Returns:
        HealthResponse with service statuses
    """
    services = {}

    # Check database
    try:
        from storage.metadata_db import db
        projects = db.list_projects(limit=1)
        services["metadata_db"] = True
    except Exception as e:
        logger.error(f"Metadata DB health check failed: {e}")
        services["metadata_db"] = False

    # Check vector database (LanceDB)
    try:
        from storage.vector_db import vector_db
        services["vector_db"] = vector_db.is_healthy()
    except Exception as e:
        logger.error(f"Vector DB health check failed: {e}")
        services["vector_db"] = False

    # Check LLM availability via llm_service
    try:
        from services.llm_service import llm_service
        llm_status = await llm_service.check_availability()
        services["ollama"] = llm_status["ollama"]
        services["cloud_api"] = llm_status["cloud"]
    except Exception as e:
        logger.error(f"LLM health check failed: {e}")
        services["ollama"] = False
        services["cloud_api"] = False

    # Check Ollama embed model
    if settings.LLM_PROVIDER in ["ollama", "hybrid"]:
        try:
            from services.embedding_service import embedding_service
            services["ollama_embed"] = await embedding_service.is_available()
        except Exception:
            services["ollama_embed"] = False
    else:
        services["ollama_embed"] = None

    # Overall status
    critical_services = ["metadata_db", "vector_db"]
    status = "healthy" if all(services.get(s, False) for s in critical_services) else "unhealthy"

    logger.info(f"Health check: {status} - {services}")

    return HealthResponse(
        status=status,
        timestamp=datetime.now(),
        services=services
    )


# Root Endpoint
@app.get("/", tags=["System"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.APP_NAME,
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": f"{settings.API_PREFIX}/health"
    }


# Import and Register Routes
from routes.clone import router as clone_router
from routes.projects import router as projects_router
from routes.index import router as index_router
from routes.chat import router as chat_router
from routes.system import router as system_router

app.include_router(clone_router, prefix=settings.API_PREFIX, tags=["Clone"])
app.include_router(projects_router, prefix=settings.API_PREFIX, tags=["Projects"])
app.include_router(index_router, prefix=settings.API_PREFIX, tags=["Indexing"])
app.include_router(chat_router, prefix=settings.API_PREFIX, tags=["Chat"])
app.include_router(system_router, prefix=settings.API_PREFIX, tags=["System"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="localhost",
        port=8000,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
