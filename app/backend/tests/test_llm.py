"""
Tests for the LLM service (llm_service.py).

httpx calls are mocked via respx so no real LLM endpoint is needed.
Covers Ollama streaming, cloud streaming, hybrid fallback, error paths,
and the model-override parsing logic.
"""
import json
import pytest
import respx
import httpx


# ═══════════════════════════════════════════════════════════════════════════════
# parse_model_override
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture()
def llm():
    from services.llm_service import LLMService
    return LLMService()


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
# Ollama streaming
# ═══════════════════════════════════════════════════════════════════════════════

def _ollama_ndjson(*tokens: str, done_at_end=True) -> bytes:
    """Build a fake Ollama /api/chat NDJSON response."""
    lines = []
    for t in tokens:
        lines.append(json.dumps({"message": {"content": t}, "done": False}))
    if done_at_end:
        lines.append(json.dumps({"done": True}))
    return b"\n".join(l.encode() for l in lines)


@respx.mock
async def test_ollama_stream_yields_tokens(llm):
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(200, content=_ollama_ndjson("Hello", " world", "!"))
    )

    tokens = []
    with respx.mock:
        respx.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(200, content=_ollama_ndjson("Hello", " world", "!"))
        )
        async for t in llm._stream_ollama([{"role": "user", "content": "hi"}]):
            tokens.append(t)

    assert tokens == ["Hello", " world", "!"]


@respx.mock
async def test_ollama_stream_stops_at_done(llm):
    """Tokens after `done: true` must be ignored."""
    ndjson = (
        b'{"message": {"content": "tok1"}, "done": false}\n'
        b'{"message": {"content": "HIDDEN"}, "done": true}\n'
        b'{"message": {"content": "NEVER"}, "done": false}\n'
    )
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(200, content=ndjson)
    )

    tokens = []
    async for t in llm._stream_ollama([{"role": "user", "content": "hi"}]):
        tokens.append(t)

    assert "HIDDEN" not in tokens
    assert "NEVER" not in tokens
    assert "tok1" in tokens


@respx.mock
async def test_ollama_stream_skips_empty_content(llm):
    """Chunks with empty content string should not yield empty tokens."""
    ndjson = (
        b'{"message": {"content": ""}, "done": false}\n'
        b'{"message": {"content": "real"}, "done": false}\n'
        b'{"done": true}\n'
    )
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(200, content=ndjson)
    )

    tokens = []
    async for t in llm._stream_ollama([{"role": "user", "content": "hi"}]):
        tokens.append(t)

    assert "" not in tokens
    assert "real" in tokens


@respx.mock
async def test_ollama_stream_malformed_json_skipped(llm):
    """Lines that are not valid JSON must be silently skipped."""
    ndjson = (
        b"NOT_JSON\n"
        b'{"message": {"content": "valid"}, "done": false}\n'
        b'{"done": true}\n'
    )
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(200, content=ndjson)
    )

    tokens = []
    async for t in llm._stream_ollama([{"role": "user", "content": "hi"}]):
        tokens.append(t)

    assert tokens == ["valid"]


@respx.mock
async def test_ollama_stream_http_error_raises(llm):
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(500, text="Internal Server Error")
    )
    with pytest.raises(httpx.HTTPStatusError):
        async for _ in llm._stream_ollama([{"role": "user", "content": "hi"}]):
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# Cloud (OpenAI-compatible) streaming
# ═══════════════════════════════════════════════════════════════════════════════

def _openai_sse(*tokens: str, done=True) -> bytes:
    """Build a fake OpenAI-compatible SSE response."""
    lines = []
    for t in tokens:
        data = {"choices": [{"delta": {"content": t}}]}
        lines.append(f"data: {json.dumps(data)}".encode())
    if done:
        lines.append(b"data: [DONE]")
    return b"\n".join(lines)


@pytest.fixture()
def cloud_llm(monkeypatch):
    """LLMService with fake cloud credentials patched into settings."""
    from services.llm_service import LLMService
    svc = LLMService()
    monkeypatch.setattr("services.llm_service.settings.CLOUD_API_KEY",     "fake-key")
    monkeypatch.setattr("services.llm_service.settings.CLOUD_API_BASE_URL","http://fake-cloud.test/v1")
    monkeypatch.setattr("services.llm_service.settings.CLOUD_MODEL",       "fake-model")
    return svc


@respx.mock
async def test_cloud_stream_yields_tokens(cloud_llm):
    respx.post("http://fake-cloud.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, content=_openai_sse("Hi", " there"))
    )

    tokens = []
    async for t in cloud_llm._stream_cloud([{"role": "user", "content": "hello"}]):
        tokens.append(t)

    assert tokens == ["Hi", " there"]


@respx.mock
async def test_cloud_stream_done_sentinel_stops_iteration(cloud_llm):
    respx.post("http://fake-cloud.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, content=_openai_sse("tok"))
    )

    tokens = []
    async for t in cloud_llm._stream_cloud([{"role": "user", "content": "hi"}]):
        tokens.append(t)

    assert "tok" in tokens


@respx.mock
async def test_cloud_stream_skips_empty_delta(cloud_llm):
    sse = (
        b'data: {"choices": [{"delta": {}}]}\n'
        b'data: {"choices": [{"delta": {"content": "real"}}]}\n'
        b"data: [DONE]"
    )
    respx.post("http://fake-cloud.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, content=sse)
    )

    tokens = []
    async for t in cloud_llm._stream_cloud([{"role": "user", "content": "hi"}]):
        tokens.append(t)

    assert tokens == ["real"]


@respx.mock
async def test_cloud_stream_401_raises(cloud_llm):
    respx.post("http://fake-cloud.test/v1/chat/completions").mock(
        return_value=httpx.Response(401, json={"error": "Unauthorized"})
    )
    with pytest.raises(httpx.HTTPStatusError):
        async for _ in cloud_llm._stream_cloud([{"role": "user", "content": "hi"}]):
            pass


@respx.mock
async def test_cloud_stream_429_raises(cloud_llm):
    respx.post("http://fake-cloud.test/v1/chat/completions").mock(
        return_value=httpx.Response(429, json={"error": "Rate limit"})
    )
    with pytest.raises(httpx.HTTPStatusError):
        async for _ in cloud_llm._stream_cloud([{"role": "user", "content": "hi"}]):
            pass


async def test_cloud_stream_raises_when_not_configured(monkeypatch):
    """_stream_cloud must raise RuntimeError when CLOUD_API_KEY is not set.
    We patch the settings so this test is independent of whatever .env has.
    """
    from services.llm_service import LLMService
    monkeypatch.setattr("services.llm_service.settings.CLOUD_API_KEY",     None)
    monkeypatch.setattr("services.llm_service.settings.CLOUD_API_BASE_URL", None)
    monkeypatch.setattr("services.llm_service.settings.CLOUD_MODEL",        None)

    svc = LLMService()

    with pytest.raises(RuntimeError, match="not configured"):
        async for _ in svc._stream_cloud([]):
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# Hybrid mode: cloud first, Ollama fallback
# ═══════════════════════════════════════════════════════════════════════════════

@respx.mock
async def test_hybrid_uses_cloud_when_available(monkeypatch):
    from services.llm_service import LLMService
    svc = LLMService()
    monkeypatch.setattr("services.llm_service.settings.LLM_PROVIDER",      "hybrid")
    monkeypatch.setattr("services.llm_service.settings.CLOUD_API_KEY",     "key")
    monkeypatch.setattr("services.llm_service.settings.CLOUD_API_BASE_URL","http://cloud.test/v1")
    monkeypatch.setattr("services.llm_service.settings.CLOUD_MODEL",       "m")

    respx.post("http://cloud.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, content=_openai_sse("cloud-tok"))
    )

    tokens = []
    async for t in svc.stream_chat([{"role": "user", "content": "q"}]):
        tokens.append(t)

    assert "cloud-tok" in tokens


# ═══════════════════════════════════════════════════════════════════════════════
# list_models
# ═══════════════════════════════════════════════════════════════════════════════

@respx.mock
async def test_list_models_includes_ollama_models(monkeypatch):
    from services.llm_service import LLMService
    svc = LLMService()
    monkeypatch.setattr("services.llm_service.settings.CLOUD_API_KEY", None)
    monkeypatch.setattr("services.llm_service.settings.OLLAMA_EMBED_MODEL", "nomic-embed-text")
    monkeypatch.setattr("services.llm_service.settings.OLLAMA_BASE_URL", "http://localhost:11434")

    payload = {"models": [{"name": "qwen2.5-coder:7b"}, {"name": "nomic-embed-text"}]}
    respx.get("http://localhost:11434/api/tags").mock(
        return_value=httpx.Response(200, json=payload)
    )

    models = await svc.list_models()
    ids = [m["id"] for m in models]
    assert "ollama:qwen2.5-coder:7b" in ids
    # Embed model must be excluded
    assert "ollama:nomic-embed-text" not in ids


@respx.mock
async def test_list_models_includes_cloud_model(monkeypatch):
    from services.llm_service import LLMService
    svc = LLMService()
    monkeypatch.setattr("services.llm_service.settings.CLOUD_API_KEY",      "key")
    monkeypatch.setattr("services.llm_service.settings.CLOUD_API_BASE_URL", "http://x/v1")
    monkeypatch.setattr("services.llm_service.settings.CLOUD_MODEL",        "gemini-flash")
    monkeypatch.setattr("services.llm_service.settings.CLOUD_API_PROVIDER", "Gemini")

    respx.get("http://localhost:11434/api/tags").mock(
        return_value=httpx.Response(200, json={"models": []})
    )

    models = await svc.list_models()
    ids = [m["id"] for m in models]
    assert "cloud:gemini-flash" in ids


@respx.mock
async def test_list_models_ollama_down_returns_cloud_only(monkeypatch):
    from services.llm_service import LLMService
    svc = LLMService()
    monkeypatch.setattr("services.llm_service.settings.CLOUD_API_KEY",      "key")
    monkeypatch.setattr("services.llm_service.settings.CLOUD_API_BASE_URL", "http://x/v1")
    monkeypatch.setattr("services.llm_service.settings.CLOUD_MODEL",        "gpt-4o-mini")
    monkeypatch.setattr("services.llm_service.settings.CLOUD_API_PROVIDER", "OpenAI")

    respx.get("http://localhost:11434/api/tags").mock(side_effect=httpx.ConnectError("down"))

    models = await svc.list_models()
    assert any(m["id"] == "cloud:gpt-4o-mini" for m in models)
