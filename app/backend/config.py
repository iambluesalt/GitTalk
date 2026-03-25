"""
Configuration management for GitTalk backend.
Uses Pydantic settings for environment variable management.
"""
from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Literal


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    APP_NAME: str = "GitTalk"
    DEBUG: bool = False
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # Directories
    PROJECT_ROOT: Path = _PROJECT_ROOT
    GITTALK_REPOS_DIR: Path = _PROJECT_ROOT / "cloned_repos"
    DATA_DIR: Path = _PROJECT_ROOT / "data"
    VECTOR_DB_PATH: Path = _PROJECT_ROOT / "data" / "lancedb"
    METADATA_DB_PATH: Path = _PROJECT_ROOT / "data" / "metadata.db"

    # GitHub
    GITHUB_TOKEN: str | None = None

    # Clone settings
    MAX_REPO_SIZE_MB: int = 500
    CLONE_TIMEOUT_SECONDS: int = 600

    # LLM Configuration
    LLM_PROVIDER: Literal["ollama", "cloud", "hybrid"] = "hybrid"

    # Ollama (Primary)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5-coder:7b"
    OLLAMA_TIMEOUT: int = 120

    # Cloud API (Primary in hybrid mode)
    CLOUD_API_PROVIDER: str | None = None
    CLOUD_API_KEY: str | None = None
    CLOUD_API_BASE_URL: str | None = None
    CLOUD_MODEL: str | None = None
    CLOUD_TIMEOUT: int = 120

    # Router (small/fast model for query classification & summarization)
    ROUTER_MODEL: str = "lfm2.5-thinking:latest"

    # Embeddings
    OLLAMA_EMBED_MODEL: str = "nomic-embed-text"
    EMBEDDING_BATCH_SIZE: int = 64
    EMBEDDING_DIMENSIONS: int = 768  # nomic-embed-text dimensions

    # RAG Configuration
    MAX_CONTEXT_TOKENS: int = 32768
    MAX_SEARCH_RESULTS: int = 8
    CHUNK_MAX_TOKENS: int = 1000
    RETRIEVAL_CANDIDATES: int = 30
    MIN_RELEVANCE_SCORE: float = 0.15
    CHUNK_OVERLAP_LINES: int = 3

    # Performance
    INDEXING_WORKERS: int = 4
    CACHE_TTL_HOURS: int = 24

    # API
    API_PREFIX: str = "/api"
    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    class Config:
        env_file = str(_ENV_FILE)
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"


# Global settings instance
settings = Settings()

# Ensure directories exist
settings.GITTALK_REPOS_DIR.mkdir(parents=True, exist_ok=True)
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)


def update_env_file(updates: dict[str, str | None]):
    """
    Update .env file with new values, preserving structure and comments.
    - str value  → set the key (uncomments if commented)
    - None       → comment out the key
    """
    env_path = _ENV_FILE

    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    remaining = dict(updates)
    new_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        # Normalise: strip leading '#' and whitespace to get the bare KEY=...
        check_line = stripped.lstrip("#").strip()
        matched_key: str | None = None
        for key in remaining:
            if check_line.startswith(f"{key}=") or check_line == key:
                matched_key = key
                break

        if matched_key is not None:
            value = remaining.pop(matched_key)
            if value is None:
                new_lines.append(f"# {matched_key}=")
            else:
                new_lines.append(f"{matched_key}={value}")
        else:
            new_lines.append(line)

    # Append keys that weren't already in the file
    if remaining:
        new_lines.append("")
        for key, value in remaining.items():
            if value is not None:
                new_lines.append(f"{key}={value}")
            else:
                new_lines.append(f"# {key}=")

    env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def reload_settings():
    """Reload settings from .env, updating the global instance in-place."""
    fresh = Settings()
    for field_name in Settings.model_fields:
        setattr(settings, field_name, getattr(fresh, field_name))
    settings.GITTALK_REPOS_DIR.mkdir(parents=True, exist_ok=True)
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
