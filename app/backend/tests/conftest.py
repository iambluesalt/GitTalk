"""
Shared pytest fixtures for the GitTalk backend test suite.

Constants and pure helpers live in helpers.py (a regular importable module).
This file only contains pytest fixtures.
"""
import sys
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# ── sys.path setup ────────────────────────────────────────────────────────────
BACKEND_DIR = Path(__file__).resolve().parent.parent
TESTS_DIR   = Path(__file__).resolve().parent

for p in (str(BACKEND_DIR), str(TESTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from helpers import _fake_vector, _make_project_meta, EMBED_DIM  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════════
# Isolated storage fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture()
def test_db(tmp_path):
    """Fresh MetadataDB backed by a temp SQLite file — discarded after each test."""
    from storage.metadata_db import MetadataDB
    return MetadataDB(db_path=tmp_path / "test.db")


@pytest.fixture()
def test_vector_db(tmp_path):
    """Fresh VectorDB backed by a temp LanceDB dir — discarded after each test."""
    from storage.vector_db import VectorDB
    return VectorDB(persist_directory=tmp_path / "lancedb")


# ═══════════════════════════════════════════════════════════════════════════════
# Fake embedding service (no Ollama needed)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture()
def fake_embedding_service():
    """EmbeddingService mock that returns cheap deterministic vectors."""
    from services.embedding_service import EmbeddingService

    svc = MagicMock(spec=EmbeddingService)
    svc.batch_size = 64

    async def _embed_texts(texts):
        return [_fake_vector(i) for i in range(len(texts))]

    async def _embed_single(_text):
        return _fake_vector(0)

    async def _is_available():
        return True

    svc.embed_texts   = AsyncMock(side_effect=_embed_texts)
    svc.embed_single  = AsyncMock(side_effect=_embed_single)
    svc.is_available  = AsyncMock(side_effect=_is_available)
    return svc


# ═══════════════════════════════════════════════════════════════════════════════
# Sample project fixture
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture()
def sample_project_id():
    return str(uuid.uuid4())


@pytest.fixture()
def sample_project(test_db, tmp_path, sample_project_id):
    """Insert an INDEXED project into the test DB and return its metadata."""
    from models import ProjectStatus

    clone_path = tmp_path / "repos" / "testrepo"
    clone_path.mkdir(parents=True)

    meta = _make_project_meta(sample_project_id, clone_path)
    test_db.create_project(meta)
    test_db.update_project_status(sample_project_id, ProjectStatus.INDEXED)
    return test_db.get_project(sample_project_id)


# ═══════════════════════════════════════════════════════════════════════════════
# FastAPI test client
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture()
def app_client():
    """
    httpx AsyncClient wired to the FastAPI app.
    Host is set to 'localhost' so the host-header middleware passes.
    """
    import httpx
    from main import app

    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(
        transport=transport,
        base_url="http://localhost",
        headers={"host": "localhost"},
    )
