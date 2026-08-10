"""
LLM integration service backed by LangChain chat models.
Routes between Ollama (local) and Groq (cloud) with automatic fallback.
"""
from typing import Any, AsyncGenerator, Sequence

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

from config import settings
from logger import logger


# ============================================================================
# Prompt Templates
# ============================================================================

PROMPT_TEMPLATES: dict[str, str] = {
    "code_agent": (
        "You are a knowledgeable code assistant for the '{project_name}' repository.\n"
        "You have tools that let you explore the repository yourself — use them before "
        "answering anything about the code.\n\n"
        "Tools:\n"
        "- `search_codebase` — semantic + keyword search over the indexed code. Start here.\n"
        "- `read_file` — read exact file contents when a snippet is truncated or you need "
        "the surrounding code.\n"
        "- `list_files` — list a directory when you need to find where something lives.\n\n"
        "Rules:\n"
        "- Search before you answer. If the first search isn't enough, search again with "
        "different wording or read the relevant file — several tool calls per question is normal.\n"
        "- ALWAYS cite what you used: file path and line numbers (e.g. `src/auth.py:42-58`).\n"
        "- Never invent code, file paths, or APIs you haven't seen through a tool.\n"
        "- If the repository genuinely doesn't contain the answer, say so explicitly and "
        "explain what would be needed.\n"
        "- Be concise but thorough. Explain the 'why', not just the 'what'."
    ),
    "general": (
        "You are a friendly assistant for the '{project_name}' repository.\n"
        "The user is having a casual conversation, asking a general knowledge question, "
        "or making small talk. Respond naturally and conversationally.\n"
        "You have access to the repository structure below for reference, "
        "but only mention it if the user's question is related."
    ),
}


def get_prompt_template(template_name: str, project_name: str = "unknown") -> str:
    """Get a system prompt by template name with project name substituted."""
    template = PROMPT_TEMPLATES.get(template_name, PROMPT_TEMPLATES["code_agent"])
    return template.format(project_name=project_name)


# ============================================================================
# Helpers
# ============================================================================

def _groq_base_url() -> str | None:
    """
    Normalise CLOUD_API_BASE_URL for the Groq SDK.

    Settings hold the full OpenAI-compatible endpoint
    (e.g. https://api.groq.com/openai/v1), but the groq client appends
    '/openai/v1' to every request path itself — so strip that suffix to
    avoid a doubled path. Returns None when no base URL is configured,
    letting the SDK use its own default.
    """
    raw = settings.CLOUD_API_BASE_URL
    if not raw:
        return None
    base = raw.rstrip("/")
    for suffix in ("/openai/v1", "/openai"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return base or None


def _chunk_text(chunk: Any) -> str:
    """Extract plain text from a streamed message chunk (str or content blocks)."""
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part if isinstance(part, str) else str(part.get("text", ""))
            for part in content
        )
    return ""


# ============================================================================
# LLM Service
# ============================================================================

class LLMService:
    """Routes LLM requests between Ollama and Groq with streaming."""

    # ====================================================================
    # Public API
    # ====================================================================

    def parse_model_override(self, model_str: str | None) -> tuple[str | None, str | None]:
        """
        Parse a model override string like 'cloud:gemini-2.5-flash-lite' or 'ollama:deepseek-r1:8b'.

        Returns (provider, model_name). If no prefix, returns (None, model_str).
        """
        if not model_str:
            return None, None
        if model_str.startswith("cloud:"):
            return "cloud", model_str[6:]
        if model_str.startswith("ollama:"):
            return "ollama", model_str[7:]
        return None, model_str

    def get_chat_model(
        self,
        model_override: str | None = None,
        tools: Sequence[Any] | None = None,
    ) -> Runnable[Any, BaseMessage]:
        """
        Build the chat model for this request.

        Args:
            model_override: Optional 'provider:model' string (e.g. 'cloud:gemini-2.5-flash-lite').
                            Overrides LLM_PROVIDER and model for this request.
            tools: Optional tools to bind. Bound to each provider *before* fallbacks
                   are composed — `RunnableWithFallbacks` has no `bind_tools`.

        Hybrid mode returns Groq with an Ollama fallback attached — LangChain only
        falls back if the primary fails before yielding its first chunk, which
        matches the previous hand-rolled behaviour.
        """
        override_provider, override_model = self.parse_model_override(model_override)
        provider = override_provider or settings.LLM_PROVIDER

        def bind(model: BaseChatModel) -> Runnable[Any, BaseMessage]:
            return model.bind_tools(tools) if tools else model

        if provider == "ollama":
            return bind(self._make_ollama(model=override_model))
        if provider == "cloud":
            return bind(self._make_groq(model=override_model))

        # hybrid: cloud first (fast), Ollama as fallback
        if self._cloud_configured():
            return bind(self._make_groq()).with_fallbacks([bind(self._make_ollama())])
        return bind(self._make_ollama())

    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        model_override: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream chat completion tokens.

        Args:
            messages: Chat messages in OpenAI format.
            model_override: Optional 'provider:model' string.
        """
        model = self.get_chat_model(model_override)
        async for chunk in model.astream(messages):
            token = _chunk_text(chunk)
            if token:
                yield token

    async def generate_title(self, user_message: str) -> str:
        """
        Generate a short conversation title using the fast Ollama model.

        Non-streaming so the result is available immediately after the main
        response stream finishes. Falls back gracefully to the first 60 chars
        of the user message if the fast model is unavailable or the call
        times out.
        """
        prompt = (
            "Generate a concise 4-7 word title for this conversation.\n"
            f"User message: {user_message[:300]}\n\n"
            "Reply with ONLY the title — no quotes, no punctuation at the end, no explanation."
        )
        try:
            model = ChatOllama(
                model=settings.OLLAMA_FAST_MODEL,
                base_url=settings.OLLAMA_BASE_URL,
                temperature=0.3,
                num_predict=20,
                client_kwargs={"timeout": 10.0},
            )
            result = await model.ainvoke(prompt)
            title = _chunk_text(result).strip().strip("\"'").strip()
            if title and len(title) <= 80:
                return title
        except Exception as e:
            logger.debug(f"Title generation failed ({settings.OLLAMA_FAST_MODEL}): {e}")

        # Fallback: truncate the user message
        return user_message[:60].rstrip()

    async def check_availability(self) -> dict[str, bool]:
        """Check which LLM providers are reachable."""
        status: dict[str, bool] = {"ollama": False, "cloud": False}

        # Check Ollama
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{settings.OLLAMA_BASE_URL}/api/tags")
                if resp.status_code == 200:
                    models = [m.get("name", "") for m in resp.json().get("models", [])]
                    target = settings.OLLAMA_MODEL
                    status["ollama"] = any(
                        m == target or m.startswith(f"{target}:")
                        or m.split(":")[0] == target
                        for m in models
                    )
        except Exception:
            pass

        # Check cloud — just config presence (no network call)
        status["cloud"] = self._cloud_configured()

        return status

    async def list_models(self) -> list[dict[str, str]]:
        """
        List all available chat models from Ollama and cloud.

        Returns list of {id, name, provider} dicts.
        """
        models: list[dict[str, str]] = []

        # Cloud model (if configured)
        if self._cloud_configured() and settings.CLOUD_MODEL:
            provider_label = settings.CLOUD_API_PROVIDER or "Cloud"
            models.append({
                "id": f"cloud:{settings.CLOUD_MODEL}",
                "name": f"{settings.CLOUD_MODEL}",
                "provider": provider_label,
            })

        # Ollama models
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{settings.OLLAMA_BASE_URL}/api/tags")
                if resp.status_code == 200:
                    for m in resp.json().get("models", []):
                        name = m.get("name", "")
                        if not name:
                            continue
                        # Skip embedding models
                        embed_model = settings.OLLAMA_EMBED_MODEL
                        if name == embed_model or name.startswith(f"{embed_model}:"):
                            continue
                        models.append({
                            "id": f"ollama:{name}",
                            "name": name,
                            "provider": "Ollama",
                        })
        except Exception as e:
            logger.debug(f"Could not list Ollama models: {e}")

        return models

    # ====================================================================
    # Model construction
    # ====================================================================

    def _cloud_configured(self) -> bool:
        return bool(settings.CLOUD_API_KEY and settings.CLOUD_API_BASE_URL)

    def _make_ollama(self, model: str | None = None) -> ChatOllama:
        """Build a ChatOllama client for the local Ollama server."""
        return ChatOllama(
            model=model or settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            client_kwargs={"timeout": float(settings.OLLAMA_TIMEOUT)},
        )

    def _make_groq(self, model: str | None = None) -> ChatGroq:
        """Build a ChatGroq client from the CLOUD_* settings."""
        if not self._cloud_configured():
            raise RuntimeError(
                "Cloud API not configured (set CLOUD_API_KEY and CLOUD_API_BASE_URL)"
            )
        return ChatGroq(
            model=model or settings.CLOUD_MODEL,
            api_key=settings.CLOUD_API_KEY,
            base_url=_groq_base_url(),
            timeout=float(settings.CLOUD_TIMEOUT),
            streaming=True,
        )


# Global instance
llm_service = LLMService()
