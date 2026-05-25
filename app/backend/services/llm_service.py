"""
LLM integration service with streaming support.
Routes between Ollama (local) and cloud APIs (OpenAI-compatible) with automatic fallback.
"""
import json
from typing import AsyncGenerator

import httpx

from config import settings
from logger import logger


# ============================================================================
# Prompt Templates
# ============================================================================

PROMPT_TEMPLATES: dict[str, str] = {
    "code_qa": (
        "You are a knowledgeable code assistant for the '{project_name}' repository.\n"
        "Answer questions about the codebase using ONLY the provided code context below.\n\n"
        "Rules:\n"
        "- Base your answer on the code snippets provided under '## Relevant Code'. "
        "Each snippet is numbered [1], [2], etc. with its file path and line range.\n"
        "- When referencing code, ALWAYS cite the source: mention the file path and line numbers "
        "(e.g. `src/auth.py:42-58`).\n"
        "- If you quote code, use fenced code blocks with the correct language.\n"
        "- If the provided context does NOT contain enough information to fully answer, "
        "say so explicitly and explain what would be needed.\n"
        "- Do NOT make up code or file paths that aren't in the provided context.\n"
        "- Be concise but thorough. Explain the 'why' not just the 'what'."
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
    template = PROMPT_TEMPLATES.get(template_name, PROMPT_TEMPLATES["code_qa"])
    return template.format(project_name=project_name)


# ============================================================================
# LLM Service
# ============================================================================

class LLMService:
    """Routes LLM requests between Ollama and cloud providers with streaming."""

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

    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        model_override: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream chat completion tokens.

        Args:
            messages: Chat messages in OpenAI format.
            model_override: Optional 'provider:model' string (e.g. 'cloud:gemini-2.5-flash-lite').
                            Overrides LLM_PROVIDER and model for this request.

        Hybrid mode: tries cloud first (fast), falls back to Ollama on failure.
        """
        override_provider, override_model = self.parse_model_override(model_override)
        provider = override_provider or settings.LLM_PROVIDER

        if provider == "ollama":
            async for token in self._stream_ollama(messages, model=override_model):
                yield token
        elif provider == "cloud":
            async for token in self._stream_cloud(messages, model=override_model):
                yield token
        elif provider == "hybrid":
            # Cloud first (fast), Ollama as fallback
            cloud_configured = bool(
                settings.CLOUD_API_KEY and settings.CLOUD_API_BASE_URL
            )
            if cloud_configured:
                yield_started = False
                try:
                    async for token in self._stream_cloud(messages):
                        yield_started = True
                        yield token
                    return
                except Exception as e:
                    if yield_started:
                        raise
                    logger.warning(f"Cloud API failed, falling back to Ollama: {e}")
            # Ollama fallback
            async for token in self._stream_ollama(messages):
                yield token

    async def generate_title(self, user_message: str) -> str:
        """
        Generate a short conversation title using the fast Ollama model.

        Uses a non-streaming /api/generate call so the result is available
        immediately after the main response stream finishes. Falls back
        gracefully to the first 60 chars of the user message if the fast
        model is unavailable or the call times out.
        """
        prompt = (
            "Generate a concise 4-7 word title for this conversation.\n"
            f"User message: {user_message[:300]}\n\n"
            "Reply with ONLY the title — no quotes, no punctuation at the end, no explanation."
        )
        try:
            payload = {
                "model": settings.OLLAMA_FAST_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 20},
            }
            timeout = httpx.Timeout(10.0, connect=5.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(
                    f"{settings.OLLAMA_BASE_URL}/api/generate",
                    json=payload,
                )
                resp.raise_for_status()
                title = resp.json().get("response", "").strip().strip("\"'").strip()
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
        status["cloud"] = bool(
            settings.CLOUD_API_KEY and settings.CLOUD_API_BASE_URL
        )

        return status

    async def list_models(self) -> list[dict[str, str]]:
        """
        List all available chat models from Ollama and cloud.

        Returns list of {id, name, provider} dicts.
        """
        models: list[dict[str, str]] = []

        # Cloud model (if configured)
        if settings.CLOUD_API_KEY and settings.CLOUD_API_BASE_URL and settings.CLOUD_MODEL:
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
    # Ollama Backend — /api/chat (NDJSON streaming)
    # ====================================================================

    async def _stream_ollama(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream tokens from Ollama /api/chat."""
        url = f"{settings.OLLAMA_BASE_URL}/api/chat"
        payload = {
            "model": model or settings.OLLAMA_MODEL,
            "messages": messages,
            "stream": True,
        }

        timeout = httpx.Timeout(settings.OLLAMA_TIMEOUT, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if data.get("done"):
                        break

                    token = data.get("message", {}).get("content", "")
                    if token:
                        yield token

    # ====================================================================
    # Cloud Backend — OpenAI-compatible /chat/completions (SSE streaming)
    # ====================================================================

    async def _stream_cloud(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream tokens from an OpenAI-compatible chat completions endpoint."""
        if not settings.CLOUD_API_KEY or not settings.CLOUD_API_BASE_URL:
            raise RuntimeError(
                "Cloud API not configured (set CLOUD_API_KEY and CLOUD_API_BASE_URL)"
            )

        base_url = settings.CLOUD_API_BASE_URL.rstrip("/")
        url = f"{base_url}/chat/completions"

        headers = {
            "Authorization": f"Bearer {settings.CLOUD_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model or settings.CLOUD_MODEL,
            "messages": messages,
            "stream": True,
        }

        timeout = httpx.Timeout(settings.CLOUD_TIMEOUT, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST", url, json=payload, headers=headers
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data: "):
                        continue

                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break

                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    choices = data.get("choices", [])
                    if choices:
                        token = choices[0].get("delta", {}).get("content", "")
                        if token:
                            yield token


# Global instance
llm_service = LLMService()
