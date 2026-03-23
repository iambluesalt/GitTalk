"""
System routes: configuration read & update.
"""
from fastapi import APIRouter
from config import settings, update_env_file, reload_settings
from models import ConfigUpdate
from logger import logger

router = APIRouter()


def _config_dict() -> dict:
    """Build sanitised config response (no raw secrets)."""
    return {
        "app_name": settings.APP_NAME,
        "llm_provider": settings.LLM_PROVIDER,
        "ollama_base_url": settings.OLLAMA_BASE_URL,
        "ollama_model": settings.OLLAMA_MODEL,
        "ollama_embed_model": settings.OLLAMA_EMBED_MODEL,
        "ollama_timeout": settings.OLLAMA_TIMEOUT,
        "cloud_api_configured": bool(settings.CLOUD_API_KEY and settings.CLOUD_API_BASE_URL),
        "cloud_api_provider": settings.CLOUD_API_PROVIDER,
        "cloud_model": settings.CLOUD_MODEL,
        "cloud_api_base_url": settings.CLOUD_API_BASE_URL,
        "max_repo_size_mb": settings.MAX_REPO_SIZE_MB,
        "clone_timeout_seconds": settings.CLONE_TIMEOUT_SECONDS,
        "max_context_tokens": settings.MAX_CONTEXT_TOKENS,
        "max_search_results": settings.MAX_SEARCH_RESULTS,
        "chunk_max_tokens": settings.CHUNK_MAX_TOKENS,
        "retrieval_candidates": settings.RETRIEVAL_CANDIDATES,
        "embedding_dimensions": settings.EMBEDDING_DIMENSIONS,
        "embedding_batch_size": settings.EMBEDDING_BATCH_SIZE,
        "indexing_workers": settings.INDEXING_WORKERS,
        "github_token_configured": bool(settings.GITHUB_TOKEN),
    }


@router.get("/config")
async def get_config():
    """Return current configuration (sanitised — no secrets)."""
    return _config_dict()


@router.put("/config")
async def update_config(update: ConfigUpdate):
    """
    Update configuration, persist to .env, and hot-reload settings.
    Only fields present in the request body are changed.
    Sending null for a field clears it (comments it out in .env).
    """
    changes = update.model_dump(exclude_unset=True)
    if not changes:
        return _config_dict()

    # Map model field names → ENV variable names and stringify
    env_updates: dict[str, str | None] = {}
    for field_name, value in changes.items():
        env_key = field_name.upper()
        env_updates[env_key] = str(value) if value is not None else None

    logger.info(f"Config update: {list(env_updates.keys())}")
    update_env_file(env_updates)
    reload_settings()

    return _config_dict()
