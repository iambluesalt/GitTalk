"""
Tests for the LLM service (llm_service.py).

The service is backed by LangChain chat models, so transport-level parsing
(NDJSON / SSE) is LangChain's concern and is not retested here. These cover
our own seams: model-override parsing, provider routing, Groq base-URL
normalisation, chunk-text extraction, hybrid fallback semantics, and the
httpx-based introspection helpers (mocked via respx).
"""
import pytest
import respx
import httpx

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.runnables import RunnableWithFallbacks
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama


@pytest.fixture()
def llm():
    from services.llm_service import LLMService
    return LLMService()


@pytest.fixture()
def cloud_settings(monkeypatch):
    """Patch fake cloud credentials into settings."""
    monkeypatch.setattr("services.llm_service.settings.CLOUD_API_KEY",      "fake-key")
    monkeypatch.setattr("services.llm_service.settings.CLOUD_API_BASE_URL", "https://api.groq.com/openai/v1")
    monkeypatch.setattr("services.llm_service.settings.CLOUD_MODEL",        "fake-model")


# ═══════════════════════════════════════════════════════════════════════════════
# parse_model_override
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("override, expected_provider, expected_model", [
    ("cloud:gemini-2.0-flash",       "cloud",  "gemini-2.0-flash"),
    ("cloud:gpt-4o-mini",            "cloud",  "gpt-4o-mini"),
    ("ollama:qwen2.5-coder:7b",      "ollama", "qwen2.5-coder:7b"),
    ("ollama:deepseek-r1:8b",        "ollama", "deepseek-r1:8b"),
    ("bare-model-name",              None,     "bare-model-name"),
    (None,                           None,     None),
    ("",                             None,     None),
])
def test_parse_model_override(llm, override, expected_provider, expected_model):
    provider, model = llm.parse_model_override(override or None)
    assert provider == expected_provider
    assert model   == expected_model


# ═══════════════════════════════════════════════════════════════════════════════
# Groq base-URL normalisation
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("configured, expected", [
    # The groq client appends /openai/v1 itself — the suffix must be stripped
    ("https://api.groq.com/openai/v1",  "https://api.groq.com"),
    ("https://api.groq.com/openai/v1/", "https://api.groq.com"),
    ("https://api.groq.com/openai",     "https://api.groq.com"),
    ("https://api.groq.com",            "https://api.groq.com"),
    ("https://proxy.internal/openai/v1", "https://proxy.internal"),
    (None,                              None),
    ("",                                None),
])
def test_groq_base_url_normalisation(monkeypatch, configured, expected):
    from services.llm_service import _groq_base_url
    monkeypatch.setattr("services.llm_service.settings.CLOUD_API_BASE_URL", configured)
    assert _groq_base_url() == expected


# ═══════════════════════════════════════════════════════════════════════════════
# Chunk text extraction
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("content, expected", [
    ("plain text",                                          "plain text"),
    ("",                                                    ""),
    ([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}], "ab"),
    (["raw", {"type": "text", "text": "!"}],                "raw!"),
    ([{"type": "image", "url": "x"}],                       ""),
])
def test_chunk_text_extraction(content, expected):
    from services.llm_service import _chunk_text
    assert _chunk_text(AIMessageChunk(content=content)) == expected


# ═══════════════════════════════════════════════════════════════════════════════
# get_chat_model — provider routing
# ═══════════════════════════════════════════════════════════════════════════════

def test_get_chat_model_ollama_provider(llm, monkeypatch):
    monkeypatch.setattr("services.llm_service.settings.LLM_PROVIDER", "ollama")
    model = llm.get_chat_model()
    assert isinstance(model, ChatOllama)


def test_get_chat_model_cloud_provider(llm, monkeypatch, cloud_settings):
    monkeypatch.setattr("services.llm_service.settings.LLM_PROVIDER", "cloud")
    model = llm.get_chat_model()
    assert isinstance(model, ChatGroq)
    assert model.model_name == "fake-model"


def test_get_chat_model_hybrid_attaches_fallback(llm, monkeypatch, cloud_settings):
    monkeypatch.setattr("services.llm_service.settings.LLM_PROVIDER", "hybrid")
    model = llm.get_chat_model()
    assert isinstance(model, RunnableWithFallbacks)
    assert isinstance(model.runnable, ChatGroq)
    assert isinstance(model.fallbacks[0], ChatOllama)


def test_get_chat_model_hybrid_without_cloud_uses_ollama(llm, monkeypatch):
    """Hybrid with no cloud credentials must go straight to Ollama, not error."""
    monkeypatch.setattr("services.llm_service.settings.LLM_PROVIDER", "hybrid")
    monkeypatch.setattr("services.llm_service.settings.CLOUD_API_KEY", None)
    monkeypatch.setattr("services.llm_service.settings.CLOUD_API_BASE_URL", None)
    model = llm.get_chat_model()
    assert isinstance(model, ChatOllama)


@pytest.mark.parametrize("override, expected_cls, expected_model", [
    ("ollama:deepseek-r1:8b", ChatOllama, "deepseek-r1:8b"),
    ("cloud:llama-3.3-70b",   ChatGroq,   "llama-3.3-70b"),
])
def test_get_chat_model_override_wins(llm, monkeypatch, cloud_settings,
                                      override, expected_cls, expected_model):
    """A 'provider:model' override beats LLM_PROVIDER and the configured model."""
    monkeypatch.setattr("services.llm_service.settings.LLM_PROVIDER", "hybrid")
    model = llm.get_chat_model(override)
    assert isinstance(model, expected_cls)
    name = getattr(model, "model", None) or model.model_name
    assert name == expected_model


def test_make_groq_raises_when_not_configured(llm, monkeypatch):
    monkeypatch.setattr("services.llm_service.settings.CLOUD_API_KEY",     None)
    monkeypatch.setattr("services.llm_service.settings.CLOUD_API_BASE_URL", None)
    with pytest.raises(RuntimeError, match="not configured"):
        llm._make_groq()


# ═══════════════════════════════════════════════════════════════════════════════
# stream_chat
# ═══════════════════════════════════════════════════════════════════════════════

async def test_stream_chat_yields_tokens(llm, monkeypatch):
    fake = FakeListChatModel(responses=["Hi!"])
    monkeypatch.setattr(llm, "get_chat_model", lambda *_a, **_kw: fake)

    tokens = [t async for t in llm.stream_chat([{"role": "user", "content": "q"}])]

    assert "".join(tokens) == "Hi!"
    assert "" not in tokens


async def test_stream_chat_skips_empty_chunks(llm, monkeypatch):
    """Empty content chunks must not be forwarded as empty tokens."""
    class _EmptyThenReal(FakeListChatModel):
        async def _astream(self, *args, **kwargs):
            for text in ("", "real", ""):
                yield ChatGenerationChunk(message=AIMessageChunk(content=text))

    monkeypatch.setattr(
        llm, "get_chat_model", lambda *_a, **_kw: _EmptyThenReal(responses=["x"])
    )

    tokens = [t async for t in llm.stream_chat([{"role": "user", "content": "q"}])]
    assert tokens == ["real"]


# ═══════════════════════════════════════════════════════════════════════════════
# Hybrid fallback semantics
#
# We rely on RunnableWithFallbacks.astream pulling the *first* chunk inside its
# try block: a primary that fails before yielding anything falls back, one that
# fails mid-stream propagates. That matches the previous hand-rolled behaviour,
# so it is pinned here.
# ═══════════════════════════════════════════════════════════════════════════════

async def test_hybrid_falls_back_when_cloud_fails_immediately(llm, monkeypatch):
    monkeypatch.setattr("services.llm_service.settings.LLM_PROVIDER", "hybrid")
    monkeypatch.setattr(
        llm, "_make_groq",
        lambda *_a, **_kw: FakeListChatModel(responses=["cloud"], error_on_chunk_number=0),
    )
    monkeypatch.setattr(
        llm, "_make_ollama", lambda *_a, **_kw: FakeListChatModel(responses=["local"])
    )
    monkeypatch.setattr(llm, "_cloud_configured", lambda: True)

    tokens = [t async for t in llm.stream_chat([{"role": "user", "content": "q"}])]
    assert "".join(tokens) == "local"


async def test_hybrid_propagates_when_cloud_fails_mid_stream(llm, monkeypatch):
    """Once tokens have been emitted, a failure must surface — not silently restart."""
    monkeypatch.setattr("services.llm_service.settings.LLM_PROVIDER", "hybrid")
    monkeypatch.setattr(
        llm, "_make_groq",
        lambda *_a, **_kw: FakeListChatModel(responses=["cloud"], error_on_chunk_number=2),
    )
    monkeypatch.setattr(
        llm, "_make_ollama", lambda *_a, **_kw: FakeListChatModel(responses=["local"])
    )
    monkeypatch.setattr(llm, "_cloud_configured", lambda: True)

    tokens = []
    with pytest.raises(Exception):
        async for t in llm.stream_chat([{"role": "user", "content": "q"}]):
            tokens.append(t)

    assert "".join(tokens) == "cl"  # partial primary output, no fallback restart


# ═══════════════════════════════════════════════════════════════════════════════
# generate_title
# ═══════════════════════════════════════════════════════════════════════════════

async def test_generate_title_falls_back_to_truncated_message(llm, monkeypatch):
    """If the fast model is unreachable, fall back to the first 60 chars."""
    monkeypatch.setattr(
        "services.llm_service.settings.OLLAMA_BASE_URL", "http://127.0.0.1:1"
    )
    message = "How does the authentication middleware validate refresh tokens in this repo?"
    title = await llm.generate_title(message)
    assert title == message[:60].rstrip()


# ═══════════════════════════════════════════════════════════════════════════════
# check_availability
# ═══════════════════════════════════════════════════════════════════════════════

@respx.mock
async def test_check_availability_reports_both(llm, monkeypatch, cloud_settings):
    monkeypatch.setattr("services.llm_service.settings.OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setattr("services.llm_service.settings.OLLAMA_MODEL", "qwen2.5-coder")
    respx.get("http://localhost:11434/api/tags").mock(
        return_value=httpx.Response(200, json={"models": [{"name": "qwen2.5-coder:7b"}]})
    )

    status = await llm.check_availability()
    assert status == {"ollama": True, "cloud": True}


@respx.mock
async def test_check_availability_ollama_down(llm, monkeypatch):
    monkeypatch.setattr("services.llm_service.settings.OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setattr("services.llm_service.settings.CLOUD_API_KEY", None)
    respx.get("http://localhost:11434/api/tags").mock(side_effect=httpx.ConnectError("down"))

    status = await llm.check_availability()
    assert status == {"ollama": False, "cloud": False}


# ═══════════════════════════════════════════════════════════════════════════════
# list_models
# ═══════════════════════════════════════════════════════════════════════════════

@respx.mock
async def test_list_models_includes_ollama_models(llm, monkeypatch):
    monkeypatch.setattr("services.llm_service.settings.CLOUD_API_KEY", None)
    monkeypatch.setattr("services.llm_service.settings.OLLAMA_EMBED_MODEL", "nomic-embed-text")
    monkeypatch.setattr("services.llm_service.settings.OLLAMA_BASE_URL", "http://localhost:11434")

    payload = {"models": [{"name": "qwen2.5-coder:7b"}, {"name": "nomic-embed-text"}]}
    respx.get("http://localhost:11434/api/tags").mock(
        return_value=httpx.Response(200, json=payload)
    )

    models = await llm.list_models()
    ids = [m["id"] for m in models]
    assert "ollama:qwen2.5-coder:7b" in ids
    # Embed model must be excluded
    assert "ollama:nomic-embed-text" not in ids


@respx.mock
async def test_list_models_includes_cloud_model(llm, monkeypatch):
    monkeypatch.setattr("services.llm_service.settings.CLOUD_API_KEY",      "key")
    monkeypatch.setattr("services.llm_service.settings.CLOUD_API_BASE_URL", "http://x/v1")
    monkeypatch.setattr("services.llm_service.settings.CLOUD_MODEL",        "gemini-flash")
    monkeypatch.setattr("services.llm_service.settings.CLOUD_API_PROVIDER", "Gemini")

    respx.get("http://localhost:11434/api/tags").mock(
        return_value=httpx.Response(200, json={"models": []})
    )

    models = await llm.list_models()
    ids = [m["id"] for m in models]
    assert "cloud:gemini-flash" in ids


@respx.mock
async def test_list_models_ollama_down_returns_cloud_only(llm, monkeypatch):
    monkeypatch.setattr("services.llm_service.settings.CLOUD_API_KEY",      "key")
    monkeypatch.setattr("services.llm_service.settings.CLOUD_API_BASE_URL", "http://x/v1")
    monkeypatch.setattr("services.llm_service.settings.CLOUD_MODEL",        "gpt-4o-mini")
    monkeypatch.setattr("services.llm_service.settings.CLOUD_API_PROVIDER", "OpenAI")

    respx.get("http://localhost:11434/api/tags").mock(side_effect=httpx.ConnectError("down"))

    models = await llm.list_models()
    assert any(m["id"] == "cloud:gpt-4o-mini" for m in models)
