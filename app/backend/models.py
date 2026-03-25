"""
Pydantic models for request/response validation and data structures.
"""
from pydantic import BaseModel, HttpUrl, Field, field_validator
from datetime import datetime
from typing import Literal, Optional, Any
from enum import Enum


# Enums
class ProjectStatus(str, Enum):
    """Status of a cloned project."""
    CLONING = "cloning"
    CLONED = "cloned"
    INDEXING = "indexing"
    INDEXED = "indexed"
    ERROR = "error"


# Request Models
class CloneRequest(BaseModel):
    """Request to clone a GitHub repository."""
    github_url: HttpUrl = Field(..., description="GitHub repository URL")
    force: bool = Field(False, description="Force clone even if repo exists")

    @field_validator("github_url")
    @classmethod
    def validate_github_url(cls, v: HttpUrl) -> HttpUrl:
        """Ensure URL is from GitHub."""
        if "github.com" not in str(v):
            raise ValueError("URL must be a GitHub repository")
        return v


class ChatRequest(BaseModel):
    """Request to send a chat message."""
    project_id: str = Field(..., description="Project identifier")
    message: str = Field(..., min_length=1, max_length=1000, description="User query")
    conversation_id: Optional[str] = Field(None, description="Existing conversation ID (auto-created if omitted)")
    stream: bool = Field(True, description="Enable streaming response")
    model: Optional[str] = Field(None, description="Model override (e.g. 'cloud:gemini-2.5-flash-lite' or 'ollama:deepseek-r1:8b')")


class IndexRequest(BaseModel):
    """Request to index a project."""
    project_id: str = Field(..., description="Project identifier")
    force_reindex: bool = Field(False, description="Force reindexing")


class ConfigUpdate(BaseModel):
    """Partial config update — only include fields you want to change."""
    llm_provider: Optional[Literal["ollama", "cloud", "hybrid"]] = None
    ollama_base_url: Optional[str] = None
    ollama_model: Optional[str] = None
    ollama_embed_model: Optional[str] = None
    router_model: Optional[str] = None
    ollama_timeout: Optional[int] = Field(None, ge=10, le=600)
    cloud_api_provider: Optional[str] = None
    cloud_api_key: Optional[str] = None
    cloud_api_base_url: Optional[str] = None
    cloud_model: Optional[str] = None
    github_token: Optional[str] = None
    max_repo_size_mb: Optional[int] = Field(None, ge=10, le=5000)
    clone_timeout_seconds: Optional[int] = Field(None, ge=60, le=3600)
    max_context_tokens: Optional[int] = Field(None, ge=1024, le=65536)
    max_search_results: Optional[int] = Field(None, ge=1, le=50)
    chunk_max_tokens: Optional[int] = Field(None, ge=100, le=5000)
    retrieval_candidates: Optional[int] = Field(None, ge=5, le=200)
    min_relevance_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    chunk_overlap_lines: Optional[int] = Field(None, ge=0, le=20)
    embedding_dimensions: Optional[int] = Field(None, ge=128, le=4096)
    embedding_batch_size: Optional[int] = Field(None, ge=1, le=512)
    indexing_workers: Optional[int] = Field(None, ge=1, le=32)


# Response Models
class HealthResponse(BaseModel):
    """Health check response."""
    status: Literal["healthy", "unhealthy"]
    timestamp: datetime
    version: str = "1.0.0"
    services: dict[str, bool | None] = Field(default_factory=dict)


class LanguageStats(BaseModel):
    """Code structure statistics for a specific language."""
    file_count: int = 0
    lines_of_code: int = 0
    functions: int = 0
    classes: int = 0
    imports: int = 0


class RepositoryAnalysis(BaseModel):
    """Analysis results for a repository (language-agnostic)."""
    # Overall statistics
    total_files: int
    total_lines: int
    repository_size_mb: float = 0.0

    # File breakdowns
    file_types: dict[str, int] = Field(
        default_factory=dict,
        description="Extension -> count (e.g., {'.py': 150, '.js': 80})"
    )

    # Language-specific analysis
    languages: dict[str, LanguageStats] = Field(
        default_factory=dict,
        description="Language -> detailed stats (e.g., {'python': LanguageStats(...), 'javascript': ...})"
    )

    # Top language detected (by LOC)
    primary_language: Optional[str] = None


class ProjectMetadata(BaseModel):
    """Metadata for a cloned project."""
    id: str
    name: str
    github_url: str
    clone_path: str
    status: ProjectStatus
    cloned_at: datetime
    last_indexed: Optional[datetime] = None
    last_used: Optional[datetime] = None
    analysis: Optional[RepositoryAnalysis] = None
    repo_map: Optional[str] = None
    error_message: Optional[str] = None


class ProjectListResponse(BaseModel):
    """List of projects response."""
    projects: list[ProjectMetadata]
    total: int
    recent_projects: list[ProjectMetadata] = Field(default_factory=list)


class CodeReference(BaseModel):
    """Reference to a code location."""
    file_path: str
    line_start: int
    line_end: int
    code_snippet: str
    relevance_score: float = 0.0


class SearchResult(BaseModel):
    """A single search result from hybrid search."""
    chunk_id: str
    text: str
    file_path: str
    language: str
    function_name: str = ""
    class_name: str = ""
    line_start: int
    line_end: int
    chunk_type: str
    relevance_score: float = 0.0


class RAGContext(BaseModel):
    """Assembled context ready for LLM consumption."""
    messages: list[dict[str, str]]
    sources: list[CodeReference]
    token_count: int
    search_results_count: int


class ConversationTurn(BaseModel):
    """A single turn in conversation history."""
    role: Literal["user", "assistant"]
    content: str


class IndexProgressEvent(BaseModel):
    """SSE progress data for indexing."""
    files_processed: int = 0
    total_files: int = 0
    current_file: str = ""
    chunks_created: int = 0
    percent: float = 0.0


class PreflightResponse(BaseModel):
    """Pre-clone repo info from GitHub API."""
    name: str
    full_name: str
    description: Optional[str] = None
    size_kb: int = 0
    size_mb: float = 0.0
    stars: int = 0
    forks: int = 0
    default_branch: str = "main"
    language: Optional[str] = None
    updated_at: Optional[str] = None
    private: bool = False
    max_size_mb: int = 500
    size_warning: Literal["ok", "medium", "large", "too_large"] = "ok"
    size_warning_message: str = ""


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


# SSE Event Models for Clone Progress
class SSEEvent(BaseModel):
    """Server-Sent Event for clone progress streaming."""
    event: str = Field(..., description="Event type: status, progress, duplicate, complete, error")
    data: dict[str, Any] = Field(default_factory=dict)

    def format(self) -> str:
        """Format as SSE string."""
        import json
        return f"event: {self.event}\ndata: {json.dumps(self.data)}\n\n"
