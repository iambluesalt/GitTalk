"""
Tests for all FastAPI API endpoints.

Uses httpx AsyncClient with ASGITransport — no real server needed.
Global db/vector_db singletons are monkeypatched with isolated test instances.
"""
import json
import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx

from helpers import _make_project_meta, _fake_vector


# ── test client factory ───────────────────────────────────────────────────────

@pytest.fixture()
async def client(tmp_path, test_db, test_vector_db, monkeypatch):
    """AsyncClient wired to the FastAPI app with isolated storage."""
    # Patch the global singletons used by routes
    monkeypatch.setattr("storage.metadata_db.db",  test_db,        raising=False)
    monkeypatch.setattr("storage.vector_db.vector_db", test_vector_db, raising=False)

    # Patch them in every route module too
    for module in ["routes.chat", "routes.projects", "routes.index",
                   "routes.clone", "routes.system", "main"]:
        try:
            monkeypatch.setattr(f"{module}.db", test_db, raising=False)
        except AttributeError:
            pass

    from main import app
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://localhost",
        headers={"host": "localhost"},
    ) as ac:
        yield ac


# ═══════════════════════════════════════════════════════════════════════════════
# GET /api/health
# ═══════════════════════════════════════════════════════════════════════════════

async def test_health_returns_200(client):
    r = await client.get("/api/health")
    assert r.status_code == 200


async def test_health_response_shape(client):
    r = await client.get("/api/health")
    body = r.json()
    assert "status" in body
    assert "services" in body
    assert body["status"] in ("healthy", "unhealthy")


async def test_health_services_includes_dbs(client):
    r = await client.get("/api/health")
    services = r.json()["services"]
    assert "metadata_db" in services
    assert "vector_db" in services


# ═══════════════════════════════════════════════════════════════════════════════
# Host / Origin middleware security
# ═══════════════════════════════════════════════════════════════════════════════

async def test_invalid_host_header_blocked():
    from main import app
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://evil.com",
                                  headers={"host": "evil.com"}) as ac:
        r = await ac.get("/api/health")
    assert r.status_code == 403


async def test_invalid_origin_blocked():
    from main import app
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost",
                                  headers={"host": "localhost",
                                           "origin": "http://malicious.com"}) as ac:
        r = await ac.get("/api/health")
    assert r.status_code == 403


async def test_no_origin_header_allowed(client):
    """Requests without Origin header (e.g. direct API calls) should pass."""
    r = await client.get("/api/health")
    assert r.status_code == 200


# ═══════════════════════════════════════════════════════════════════════════════
# GET /api/config  and  PUT /api/config
# ═══════════════════════════════════════════════════════════════════════════════

async def test_get_config_returns_200(client):
    r = await client.get("/api/config")
    assert r.status_code == 200


async def test_get_config_no_raw_secrets(client):
    """The GET config must not leak API keys."""
    r = await client.get("/api/config")
    body = r.json()
    assert "cloud_api_key" not in body
    assert "github_token" not in body
    # Boolean flags are OK
    assert "cloud_api_configured" in body
    assert "github_token_configured" in body


async def test_put_config_valid_field(client, tmp_path, monkeypatch):
    """Updating a safe field should return 200 and reflect the change."""
    # Patch update_env_file and reload_settings so we don't touch real .env
    monkeypatch.setattr("routes.system.update_env_file", lambda _: None)
    monkeypatch.setattr("routes.system.reload_settings",  lambda:  None)

    r = await client.put("/api/config", json={"max_search_results": 10})
    assert r.status_code == 200


async def test_put_config_invalid_value_rejected(client):
    """Pydantic validation: max_search_results > 50 must be rejected."""
    r = await client.put("/api/config", json={"max_search_results": 9999})
    assert r.status_code == 422


async def test_put_config_empty_body_is_noop(client, monkeypatch):
    monkeypatch.setattr("routes.system.update_env_file", lambda _: None)
    monkeypatch.setattr("routes.system.reload_settings",  lambda:  None)
    r = await client.put("/api/config", json={})
    assert r.status_code == 200


# ═══════════════════════════════════════════════════════════════════════════════
# GET /api/projects
# ═══════════════════════════════════════════════════════════════════════════════

async def test_get_projects_empty(client):
    r = await client.get("/api/projects")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 0
    assert body["projects"] == []


async def test_get_projects_lists_projects(client, test_db, tmp_path):
    pid = str(uuid.uuid4())
    meta = _make_project_meta(pid, tmp_path / "r")
    test_db.create_project(meta)

    r = await client.get("/api/projects")
    assert r.status_code == 200
    assert r.json()["total"] == 1


# ═══════════════════════════════════════════════════════════════════════════════
# GET /api/projects/:id
# ═══════════════════════════════════════════════════════════════════════════════

async def test_get_project_not_found(client):
    r = await client.get("/api/projects/nonexistent-id")
    assert r.status_code == 404


async def test_get_project_found(client, test_db, tmp_path):
    pid = str(uuid.uuid4())
    meta = _make_project_meta(pid, tmp_path / "r")
    test_db.create_project(meta)

    r = await client.get(f"/api/projects/{pid}")
    assert r.status_code == 200
    assert r.json()["id"] == pid


# ═══════════════════════════════════════════════════════════════════════════════
# DELETE /api/projects/:id
# ═══════════════════════════════════════════════════════════════════════════════

async def test_delete_project_not_found(client):
    r = await client.delete("/api/projects/nonexistent")
    assert r.status_code == 404


async def test_delete_project_ok(client, test_db, tmp_path, test_vector_db, monkeypatch):
    pid = str(uuid.uuid4())
    clone = tmp_path / "repo"
    clone.mkdir()
    meta = _make_project_meta(pid, clone)
    test_db.create_project(meta)

    monkeypatch.setattr("routes.projects.vector_db", test_vector_db)

    r = await client.delete(f"/api/projects/{pid}")
    assert r.status_code == 200
    assert r.json()["success"] is True


# ═══════════════════════════════════════════════════════════════════════════════
# GET /api/models
# ═══════════════════════════════════════════════════════════════════════════════

async def test_get_models_returns_list(client, monkeypatch):
    mock_llm = MagicMock()
    mock_llm.list_models = AsyncMock(return_value=[
        {"id": "ollama:qwen2.5-coder:7b", "name": "qwen2.5-coder:7b", "provider": "Ollama"}
    ])
    monkeypatch.setattr("routes.chat.llm_service", mock_llm)

    r = await client.get("/api/models")
    assert r.status_code == 200
    assert "models" in r.json()
    assert len(r.json()["models"]) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# POST /api/chat
# ═══════════════════════════════════════════════════════════════════════════════

async def test_chat_project_not_found(client, monkeypatch):
    """Chatting with a non-existent project must return an SSE error event."""
    r = await client.post("/api/chat", json={
        "project_id": "does-not-exist",
        "message": "hello",
    })
    assert r.status_code == 200  # SSE always 200
    assert b"error" in r.content


async def test_chat_project_not_indexed(client, test_db, tmp_path, monkeypatch):
    """Chatting with a CLONED (not indexed) project must return an SSE error."""
    from models import ProjectStatus
    pid = str(uuid.uuid4())
    meta = _make_project_meta(pid, tmp_path / "r", status="cloned")
    test_db.create_project(meta)
    monkeypatch.setattr("routes.chat.db", test_db)

    r = await client.post("/api/chat", json={
        "project_id": pid,
        "message": "what does this do?",
    })
    assert r.status_code == 200
    assert b"error" in r.content


async def test_chat_streams_tokens(client, test_db, tmp_path, monkeypatch):
    """Happy path: chat returns sources + tokens + done events."""
    from models import ProjectStatus
    from services.rag_service import RAGContext

    pid = str(uuid.uuid4())
    meta = _make_project_meta(pid, tmp_path / "r", status="indexed")
    test_db.create_project(meta)

    monkeypatch.setattr("routes.chat.db", test_db)

    mock_rag = MagicMock()
    mock_rag.build_context = AsyncMock(return_value=RAGContext(
        messages=[{"role": "user", "content": "test"}],
        sources=[],
        token_count=10,
        search_results_count=0,
    ))
    monkeypatch.setattr("routes.chat.rag_service", mock_rag)

    async def _fake_stream(messages, model_override=None):
        for tok in ["Hello", " world"]:
            yield tok

    mock_llm = MagicMock()
    mock_llm.stream_chat = _fake_stream
    monkeypatch.setattr("routes.chat.llm_service", mock_llm)

    r = await client.post("/api/chat", json={
        "project_id": pid,
        "message": "what is this?",
    })

    assert r.status_code == 200
    content = r.text
    assert "sources" in content
    assert "token" in content
    assert "done" in content
    assert "Hello" in content


async def test_chat_invalid_empty_message(client):
    r = await client.post("/api/chat", json={
        "project_id": "x",
        "message": "",  # min_length=1
    })
    assert r.status_code == 422


async def test_chat_message_too_long(client):
    r = await client.post("/api/chat", json={
        "project_id": "x",
        "message": "a" * 1001,  # max_length=1000
    })
    assert r.status_code == 422


# ═══════════════════════════════════════════════════════════════════════════════
# GET /api/conversations/:project_id
# ═══════════════════════════════════════════════════════════════════════════════

async def test_list_conversations_no_project(client, monkeypatch):
    monkeypatch.setattr("routes.chat.db", MagicMock(
        get_project=MagicMock(return_value=None)
    ))
    r = await client.get("/api/conversations/missing-proj")
    assert r.status_code == 404


async def test_list_conversations_empty(client, test_db, tmp_path, monkeypatch):
    pid = str(uuid.uuid4())
    meta = _make_project_meta(pid, tmp_path / "r")
    test_db.create_project(meta)
    monkeypatch.setattr("routes.chat.db", test_db)

    r = await client.get(f"/api/conversations/{pid}")
    assert r.status_code == 200
    assert r.json()["conversations"] == []


# ═══════════════════════════════════════════════════════════════════════════════
# DELETE /api/conversations/:project_id/:conversation_id
# ═══════════════════════════════════════════════════════════════════════════════

async def test_delete_conversation_not_found(client, test_db, tmp_path, monkeypatch):
    pid = str(uuid.uuid4())
    meta = _make_project_meta(pid, tmp_path / "r")
    test_db.create_project(meta)
    monkeypatch.setattr("routes.chat.db", test_db)

    r = await client.delete(f"/api/conversations/{pid}/nonexistent-conv")
    assert r.status_code == 404


async def test_delete_conversation_ok(client, test_db, tmp_path, monkeypatch):
    pid = str(uuid.uuid4())
    meta = _make_project_meta(pid, tmp_path / "r")
    test_db.create_project(meta)

    conv_id = test_db.create_conversation(pid, title="Test conv")
    monkeypatch.setattr("routes.chat.db", test_db)

    r = await client.delete(f"/api/conversations/{pid}/{conv_id}")
    assert r.status_code == 200
    assert r.json()["success"] is True
