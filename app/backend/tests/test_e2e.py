"""
End-to-end integration tests — clone a real GitHub repo, index it, and chat.

Uses httpx AsyncClient with ASGI transport (no running server needed).
Storage is isolated to a temp directory.  Only Ollama + git are required.

Requirements:
  - git on PATH
  - Ollama running with nomic-embed-text (embedding) and your chat model
    OR a configured cloud API key in .env

Run only these tests:
    pytest tests/test_e2e.py -m integration -v

Skip in fast CI:
    pytest tests/ -m "not integration"

Repos tested:
  User-provided:
    1. https://github.com/iambluesalt/BurnYourMoney
    2. https://github.com/iambluesalt/LangchainTest
    3. https://github.com/iambluesalt/WordSmith
  Auto-selected (tiny, well-known):
    4. https://github.com/karpathy/micrograd   (~400 lines, pure Python)
    5. https://github.com/pallets/itsdangerous (small Python security lib)
"""
import json
import shutil
import uuid
from pathlib import Path

import httpx
import pytest

pytestmark = pytest.mark.integration


# ── service availability checks ───────────────────────────────────────────────

def _ollama_available() -> bool:
    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _git_available() -> bool:
    return shutil.which("git") is not None


def _embed_model_available() -> bool:
    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=3)
        models = [m["name"] for m in r.json().get("models", [])]
        return any("nomic-embed-text" in m for m in models)
    except Exception:
        return False


def _llm_available() -> bool:
    """True if either cloud key or an Ollama chat model is configured."""
    from config import settings
    if settings.CLOUD_API_KEY:
        return True
    # Check Ollama has a chat model available
    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=3)
        models = [m["name"] for m in r.json().get("models", [])]
        embed_names = {"nomic-embed-text", "mxbai-embed-large"}
        chat_models = [m for m in models
                       if not any(e in m for e in embed_names)]
        return bool(chat_models)
    except Exception:
        return False


skip_no_services = pytest.mark.skipif(
    not (_ollama_available() and _git_available()
         and _embed_model_available() and _llm_available()),
    reason="Requires git + Ollama (nomic-embed-text) + chat model or cloud key",
)


# ── ASGI client fixture (no running server needed) ────────────────────────────

@pytest.fixture()
async def e2e_app(tmp_path):
    """
    Wire the FastAPI app to isolated temp storage.
    Real Ollama and LLM calls are made; only the DB paths are redirected
    so E2E tests never touch the production data directory.
    """
    from storage.metadata_db import MetadataDB
    from storage.vector_db import VectorDB
    import services.indexing_service as idx_svc
    import services.search_service as search_svc
    import services.rag_service as rag_svc
    import routes.clone as clone_rt
    import routes.index as index_rt
    import routes.chat as chat_rt
    import routes.projects as proj_rt
    import routes.system as sys_rt

    real_db  = MetadataDB(db_path=tmp_path / "e2e.db")
    real_vdb = VectorDB(persist_directory=tmp_path / "e2e_lancedb")

    # Patch every module that imports db / vector_db at module level
    for mod in (idx_svc, search_svc, rag_svc, clone_rt,
                index_rt, chat_rt, proj_rt, sys_rt):
        if hasattr(mod, "db"):
            mod.db = real_db
        if hasattr(mod, "vector_db"):
            mod.vector_db = real_vdb

    # Also patch clone_service's implicit usage (it reads db through the service)
    import services.clone_service as clone_svc_mod
    if hasattr(clone_svc_mod, "db"):
        clone_svc_mod.db = real_db

    from main import app
    yield app, real_db, real_vdb

    # Teardown: remove cloned repos created during this test
    from config import settings
    # Projects created will have been deleted by tests or we clean here


@pytest.fixture()
async def e2e_client(e2e_app):
    app, real_db, real_vdb = e2e_app
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://localhost",
        headers={"host": "localhost"},
        timeout=300.0,
    ) as client:
        yield client, real_db, real_vdb


# ── SSE helpers ───────────────────────────────────────────────────────────────

def _sse_events(raw_text: str) -> list[dict]:
    """Parse SSE response body into a list of {event, data} dicts."""
    events = []
    current_event = None
    for line in raw_text.splitlines():
        if line.startswith("event:"):
            current_event = line[6:].strip()
        elif line.startswith("data:") and current_event:
            try:
                payload = json.loads(line[5:].strip())
                events.append({"event": current_event, "data": payload})
            except json.JSONDecodeError:
                pass
    return events


async def _clone_repo(client: httpx.AsyncClient, url: str) -> str:
    r = await client.post("/api/clone", json={"github_url": url})
    assert r.status_code == 200, f"Clone HTTP error: {r.status_code}\n{r.text[:500]}"
    events = _sse_events(r.text)
    project_id = None
    for e in events:
        if e["event"] == "complete":
            # project_id is nested inside the project object
            project_id = (e["data"].get("project") or {}).get("id")
            break
        elif e["event"] == "duplicate":
            # duplicate carries project_id at the top level
            project_id = e["data"].get("project_id")
            break
    assert project_id, f"No project_id in clone events: {[e['event'] for e in events]}"
    return project_id


async def _index_repo(client: httpx.AsyncClient, project_id: str) -> list[dict]:
    r = await client.post("/api/index", json={"project_id": project_id})
    assert r.status_code == 200
    events = _sse_events(r.text)
    assert "error" not in [e["event"] for e in events], \
        f"Indexing produced an error: {events}"
    return events


async def _chat(client: httpx.AsyncClient, project_id: str,
                message: str, conversation_id: str | None = None) -> list[dict]:
    payload: dict = {"project_id": project_id, "message": message}
    if conversation_id:
        payload["conversation_id"] = conversation_id
    r = await client.post("/api/chat", json=payload)
    assert r.status_code == 200
    return _sse_events(r.text)


# ═══════════════════════════════════════════════════════════════════════════════
# Parametrised smoke test: clone → index → chat
# ═══════════════════════════════════════════════════════════════════════════════

E2E_REPOS = [
    ("BurnYourMoney",  "https://github.com/iambluesalt/BurnYourMoney"),
    ("LangchainTest",  "https://github.com/iambluesalt/LangchainTest"),
    ("WordSmith",      "https://github.com/iambluesalt/WordSmith"),
    ("micrograd",      "https://github.com/karpathy/micrograd"),
    ("itsdangerous",   "https://github.com/pallets/itsdangerous"),
]


@skip_no_services
@pytest.mark.parametrize("repo_name,repo_url", E2E_REPOS)
@pytest.mark.slow
async def test_smoke_clone_index_chat(repo_name, repo_url, e2e_client):
    """
    Full pipeline smoke test for each repo:
    1. Clone from GitHub
    2. Index (real embedding via Ollama)
    3. Code question → must get sources back and a non-trivial response
    4. Follow-up → must continue without error
    5. Greeting → must NOT return code sources
    """
    client, _, _ = e2e_client

    # 1. Clone
    project_id = await _clone_repo(client, repo_url)

    # 2. Index
    index_events = await _index_repo(client, project_id)
    complete = next(
        (e for e in index_events if e["event"] == "indexing_complete"), None
    )
    assert complete is not None, "indexing_complete event missing"
    print(f"\n[{repo_name}] chunks_created={complete['data']['chunks_created']}")

    # 3. Code question
    chat_events = await _chat(
        client, project_id,
        "What does this project do and what are its main components?"
    )
    assert "error" not in [e["event"] for e in chat_events], \
        f"Chat returned error: {chat_events}"

    source_event = next(
        (e for e in chat_events if e["event"] == "sources"), None
    )
    assert source_event is not None, "Missing 'sources' SSE event"

    tokens = [e["data"]["token"] for e in chat_events if e["event"] == "token"]
    assert len("".join(tokens)) > 20, "LLM response was too short"

    done = next((e for e in chat_events if e["event"] == "done"), None)
    assert done is not None
    conv_id = done["data"]["conversation_id"]

    # 4. Follow-up
    fu_events = await _chat(
        client, project_id,
        "Can you explain that in more detail?",
        conversation_id=conv_id,
    )
    assert "error" not in [e["event"] for e in fu_events]
    fu_tokens = "".join(e["data"]["token"] for e in fu_events if e["event"] == "token")
    assert len(fu_tokens) > 5

    # 5. General greeting → no code sources
    greet_events = await _chat(client, project_id, "hi")
    greet_sources = next(
        (e for e in greet_events if e["event"] == "sources"), None
    )
    if greet_sources:
        assert greet_sources["data"]["search_results_count"] == 0, \
            "General greeting should not trigger code retrieval"


# ═══════════════════════════════════════════════════════════════════════════════
# Re-indexing idempotency
# ═══════════════════════════════════════════════════════════════════════════════

@skip_no_services
@pytest.mark.slow
async def test_reindex_is_idempotent(e2e_client):
    """Re-indexing unchanged files must report 0 new chunks."""
    client, _, _ = e2e_client
    project_id = await _clone_repo(client, "https://github.com/karpathy/micrograd")
    await _index_repo(client, project_id)

    # Second index without any changes
    r = await client.post("/api/index", json={"project_id": project_id})
    events = _sse_events(r.text)

    skip_events = [e for e in events if e.get("data", {}).get("phase") == "skip"]
    complete    = next((e for e in events if e["event"] == "indexing_complete"), None)

    assert skip_events or (complete and complete["data"]["chunks_created"] == 0), \
        "Second index of unchanged repo should produce 0 new chunks"


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-turn conversation coherence
# ═══════════════════════════════════════════════════════════════════════════════

@skip_no_services
@pytest.mark.slow
async def test_multiturn_conversation(e2e_client):
    """
    Three-turn conversation:
      Turn 1: code question
      Turn 2: follow-up ("tell me more")
      Turn 3: different code question in same conversation
    Each turn must complete without an error event.
    """
    client, _, _ = e2e_client
    project_id = await _clone_repo(client, "https://github.com/karpathy/micrograd")
    await _index_repo(client, project_id)

    t1 = await _chat(client, project_id, "What is the Value class and what does it do?")
    assert "error" not in [e["event"] for e in t1]
    done1 = next(e for e in t1 if e["event"] == "done")
    conv_id = done1["data"]["conversation_id"]

    t2 = await _chat(client, project_id, "Tell me more about its backward method",
                     conversation_id=conv_id)
    assert "error" not in [e["event"] for e in t2]

    t3 = await _chat(client, project_id, "Where is the MLP class defined?",
                     conversation_id=conv_id)
    assert "error" not in [e["event"] for e in t3]


# ═══════════════════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════════════════

@skip_no_services
async def test_clone_invalid_url_rejected(e2e_client):
    """A non-GitHub URL (e.g. GitLab) must be rejected with 422 before cloning."""
    client, _, _ = e2e_client
    r = await client.post("/api/clone",
                          json={"github_url": "https://gitlab.com/user/repo"})
    assert r.status_code == 422


@skip_no_services
async def test_clone_nonexistent_repo_yields_error(e2e_client):
    """A valid-looking but non-existent GitHub URL must produce an SSE error event."""
    client, _, _ = e2e_client
    r = await client.post("/api/clone", json={
        "github_url": "https://github.com/nonexistent-user-xyz-abc/nonexistent-repo-abc123"
    })
    assert r.status_code == 200
    events = _sse_events(r.text)
    assert any(e["event"] == "error" for e in events), \
        f"Expected error event, got: {[e['event'] for e in events]}"


@skip_no_services
async def test_chat_before_index_gives_error(e2e_client):
    """Chatting with a cloned-but-not-indexed project must return an SSE error."""
    client, real_db, _ = e2e_client

    # Clone but do NOT index
    project_id = await _clone_repo(client, "https://github.com/karpathy/micrograd")

    # Force status back to 'cloned' in case the project was already indexed
    from models import ProjectStatus
    proj = real_db.get_project(project_id)
    if proj and proj.status == ProjectStatus.INDEXED:
        real_db.update_project_status(project_id, ProjectStatus.CLONED)

    chat_events = await _chat(client, project_id, "what is this?")
    assert any(e["event"] == "error" for e in chat_events), \
        "Expected an error event when chatting before indexing"


@skip_no_services
@pytest.mark.slow
async def test_conversation_persisted_and_retrievable(e2e_client):
    """After a chat turn, the conversation must be retrievable via GET."""
    client, _, _ = e2e_client
    project_id = await _clone_repo(client, "https://github.com/karpathy/micrograd")
    await _index_repo(client, project_id)

    chat_events = await _chat(client, project_id, "explain the Neuron class")
    done = next(e for e in chat_events if e["event"] == "done")
    conv_id = done["data"]["conversation_id"]

    r = await client.get(f"/api/conversations/{project_id}/{conv_id}")
    assert r.status_code == 200
    body = r.json()
    assert len(body["messages"]) == 2          # user + assistant
    assert body["messages"][0]["role"] == "user"
    assert body["messages"][1]["role"] == "assistant"
    assert len(body["messages"][1]["content"]) > 10
