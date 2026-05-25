"""
Tests for the indexing pipeline (indexing_service.py).

Uses real temp directories and real tree-sitter parsing.
The embedding service is mocked (fake_embedding_service fixture).
"""
import json
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from helpers import (
    SIMPLE_PY, CLASS_PY, IMPORTS_ONLY_PY, EMPTY_PY,
    LARGE_FN_PY, MARKDOWN_MD, _fake_vector,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _write(path: Path, name: str, content: str) -> Path:
    f = path / name
    f.write_text(content, encoding="utf-8")
    return f


def _register_project(db, project_id: str, clone_path: Path):
    """Insert a minimal project row so FK constraints pass during indexing."""
    from helpers import _make_project_meta
    from models import ProjectStatus
    meta = _make_project_meta(project_id, clone_path, status="cloned")
    db.create_project(meta)


async def _run_indexing(service, project_id: str, clone_path: Path,
                        force: bool = False) -> list[dict]:
    """Drain the SSE generator and collect all event dicts."""
    events = []
    async for raw in service.index_project(project_id, str(clone_path), force):
        # raw is "event: TYPE\ndata: JSON\n\n"
        for line in raw.splitlines():
            if line.startswith("event:"):
                current_event = line[6:].strip()
            elif line.startswith("data:"):
                payload = json.loads(line[5:].strip())
                events.append({"event": current_event, "data": payload})
    return events


def _event_types(events):
    return [e["event"] for e in events]


# ═══════════════════════════════════════════════════════════════════════════════
# Normal indexing flow
# ═══════════════════════════════════════════════════════════════════════════════

async def test_normal_indexing_completes(tmp_path, test_db, test_vector_db,
                                         fake_embedding_service):
    project_id = str(uuid.uuid4())
    _register_project(test_db, project_id, tmp_path)
    _write(tmp_path, "auth.py",    SIMPLE_PY)
    _write(tmp_path, "models.py",  CLASS_PY)
    _write(tmp_path, "README.md",  MARKDOWN_MD)

    from services.indexing_service import IndexingService
    svc = IndexingService()

    with patch("services.indexing_service.db",              test_db), \
         patch("services.indexing_service.vector_db",       test_vector_db), \
         patch("services.indexing_service.embedding_service", fake_embedding_service):
        events = await _run_indexing(svc, project_id, tmp_path)

    assert "indexing_complete" in _event_types(events)
    assert "error" not in _event_types(events)


async def test_normal_indexing_chunks_created(tmp_path, test_db, test_vector_db,
                                               fake_embedding_service):
    project_id = str(uuid.uuid4())
    _register_project(test_db, project_id, tmp_path)
    _write(tmp_path, "main.py", SIMPLE_PY)

    from services.indexing_service import IndexingService
    svc = IndexingService()

    with patch("services.indexing_service.db",               test_db), \
         patch("services.indexing_service.vector_db",        test_vector_db), \
         patch("services.indexing_service.embedding_service", fake_embedding_service):
        events = await _run_indexing(svc, project_id, tmp_path)

    complete = next(e for e in events if e["event"] == "indexing_complete")
    assert complete["data"]["chunks_created"] > 0
    assert complete["data"]["files_indexed"] > 0


async def test_indexing_updates_project_status(tmp_path, test_db, test_vector_db,
                                                fake_embedding_service):
    from models import ProjectMetadata, ProjectStatus, ProjectStatus
    from datetime import datetime

    project_id = str(uuid.uuid4())
    _write(tmp_path, "foo.py", SIMPLE_PY)

    meta = ProjectMetadata(
        id=project_id, name="repo",
        github_url=f"https://github.com/t/{project_id}",
        clone_path=str(tmp_path),
        status=ProjectStatus.CLONED,
        cloned_at=datetime.now(),
    )
    test_db.create_project(meta)

    from services.indexing_service import IndexingService
    svc = IndexingService()

    with patch("services.indexing_service.db",               test_db), \
         patch("services.indexing_service.vector_db",        test_vector_db), \
         patch("services.indexing_service.embedding_service", fake_embedding_service):
        await _run_indexing(svc, project_id, tmp_path)

    project = test_db.get_project(project_id)
    assert project.status == ProjectStatus.INDEXED


# ═══════════════════════════════════════════════════════════════════════════════
# Empty / trivial repos
# ═══════════════════════════════════════════════════════════════════════════════

async def test_empty_repo_completes_gracefully(tmp_path, test_db, test_vector_db,
                                                fake_embedding_service):
    """A repo with no eligible files should still finish without error."""
    project_id = str(uuid.uuid4())
    _register_project(test_db, project_id, tmp_path)

    from services.indexing_service import IndexingService
    svc = IndexingService()

    with patch("services.indexing_service.db",               test_db), \
         patch("services.indexing_service.vector_db",        test_vector_db), \
         patch("services.indexing_service.embedding_service", fake_embedding_service):
        events = await _run_indexing(svc, project_id, tmp_path)

    assert "error" not in _event_types(events)
    # Either 0 files or the 'skip' event fires
    event_type_set = set(_event_types(events))
    assert "indexing_complete" in event_type_set or "skip" in event_type_set


async def test_only_import_files_produces_zero_chunks(tmp_path, test_db,
                                                       test_vector_db,
                                                       fake_embedding_service):
    project_id = str(uuid.uuid4())
    _register_project(test_db, project_id, tmp_path)
    _write(tmp_path, "imports.py", IMPORTS_ONLY_PY)

    from services.indexing_service import IndexingService
    svc = IndexingService()

    with patch("services.indexing_service.db",               test_db), \
         patch("services.indexing_service.vector_db",        test_vector_db), \
         patch("services.indexing_service.embedding_service", fake_embedding_service):
        events = await _run_indexing(svc, project_id, tmp_path)

    complete_events = [e for e in events if e["event"] == "indexing_complete"]
    if complete_events:
        assert complete_events[0]["data"]["chunks_created"] == 0


async def test_empty_python_file_handled(tmp_path, test_db, test_vector_db,
                                          fake_embedding_service):
    project_id = str(uuid.uuid4())
    _register_project(test_db, project_id, tmp_path)
    _write(tmp_path, "empty.py", EMPTY_PY)

    from services.indexing_service import IndexingService
    svc = IndexingService()

    with patch("services.indexing_service.db",               test_db), \
         patch("services.indexing_service.vector_db",        test_vector_db), \
         patch("services.indexing_service.embedding_service", fake_embedding_service):
        events = await _run_indexing(svc, project_id, tmp_path)

    assert "error" not in _event_types(events)


# ═══════════════════════════════════════════════════════════════════════════════
# Incremental re-indexing
# ═══════════════════════════════════════════════════════════════════════════════

async def test_unchanged_file_skipped_on_reindex(tmp_path, test_db, test_vector_db,
                                                  fake_embedding_service):
    """Re-indexing the same unchanged file should report 0 new files.
    The repo lives in a subdirectory so LanceDB files (at tmp_path/lancedb/)
    don't pollute the scanned directory on the second run.
    """
    project_id = str(uuid.uuid4())
    # Use a dedicated sub-dir as the clone path so the lancedb directory
    # (created at tmp_path/lancedb/ by test_vector_db fixture) is NOT inside
    # the scanned repo tree — otherwise the second index pass picks up those files.
    repo = tmp_path / "repo"
    repo.mkdir()
    _register_project(test_db, project_id, repo)
    _write(repo, "stable.py", SIMPLE_PY)

    from services.indexing_service import IndexingService
    svc = IndexingService()

    run = lambda: _run_indexing(svc, project_id, repo)

    with patch("services.indexing_service.db",               test_db), \
         patch("services.indexing_service.vector_db",        test_vector_db), \
         patch("services.indexing_service.embedding_service", fake_embedding_service):
        await run()  # first index
        events2 = await run()  # second index — nothing changed

    # Either 'skip' event fires or chunks_created == 0
    skip_events = [e for e in events2 if e["event"] == "status"
                   and e["data"].get("phase") == "skip"]
    complete_events = [e for e in events2 if e["event"] == "indexing_complete"]

    if complete_events:
        assert complete_events[0]["data"]["chunks_created"] == 0
    else:
        assert len(skip_events) > 0


async def test_changed_file_reindexed(tmp_path, test_db, test_vector_db,
                                       fake_embedding_service):
    """Modifying a file must cause it to be re-processed on next index."""
    project_id = str(uuid.uuid4())
    _register_project(test_db, project_id, tmp_path)
    f = _write(tmp_path, "changing.py", SIMPLE_PY)

    from services.indexing_service import IndexingService
    svc = IndexingService()

    with patch("services.indexing_service.db",               test_db), \
         patch("services.indexing_service.vector_db",        test_vector_db), \
         patch("services.indexing_service.embedding_service", fake_embedding_service):
        await _run_indexing(svc, project_id, tmp_path)

        # Modify the file
        f.write_text(SIMPLE_PY + "\ndef new_fn(): pass\n", encoding="utf-8")

        events2 = await _run_indexing(svc, project_id, tmp_path)

    complete_events = [e for e in events2 if e["event"] == "indexing_complete"]
    assert complete_events, "Second index must complete"
    assert complete_events[0]["data"]["files_indexed"] >= 1


async def test_force_reindex_reprocesses_all(tmp_path, test_db, test_vector_db,
                                              fake_embedding_service):
    project_id = str(uuid.uuid4())
    _register_project(test_db, project_id, tmp_path)
    _write(tmp_path, "main.py", SIMPLE_PY)

    from services.indexing_service import IndexingService
    svc = IndexingService()

    with patch("services.indexing_service.db",               test_db), \
         patch("services.indexing_service.vector_db",        test_vector_db), \
         patch("services.indexing_service.embedding_service", fake_embedding_service):
        await _run_indexing(svc, project_id, tmp_path)
        events2 = await _run_indexing(svc, project_id, tmp_path, force=True)

    complete_events = [e for e in events2 if e["event"] == "indexing_complete"]
    assert complete_events[0]["data"]["files_indexed"] >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# Embedding service unavailable
# ═══════════════════════════════════════════════════════════════════════════════

async def test_embedding_unavailable_yields_error(tmp_path, test_db, test_vector_db):
    from unittest.mock import AsyncMock, MagicMock
    from services.embedding_service import EmbeddingService

    project_id = str(uuid.uuid4())
    _register_project(test_db, project_id, tmp_path)
    _write(tmp_path, "code.py", SIMPLE_PY)

    down_embed = MagicMock(spec=EmbeddingService)
    down_embed.is_available = AsyncMock(return_value=False)

    from services.indexing_service import IndexingService
    svc = IndexingService()

    with patch("services.indexing_service.db",               test_db), \
         patch("services.indexing_service.vector_db",        test_vector_db), \
         patch("services.indexing_service.embedding_service", down_embed):
        events = await _run_indexing(svc, project_id, tmp_path)

    assert "error" in _event_types(events)


# ═══════════════════════════════════════════════════════════════════════════════
# File exclusions
# ═══════════════════════════════════════════════════════════════════════════════

async def test_large_file_over_1mb_excluded(tmp_path, test_db, test_vector_db,
                                             fake_embedding_service):
    """Files larger than 1 MB must be silently excluded from indexing."""
    project_id = str(uuid.uuid4())
    _register_project(test_db, project_id, tmp_path)
    big_file = tmp_path / "huge.py"
    big_file.write_bytes(b"x = 1\n" * 200_000)  # well over 1 MB

    _write(tmp_path, "small.py", SIMPLE_PY)

    from services.indexing_service import IndexingService
    svc = IndexingService()

    with patch("services.indexing_service.db",               test_db), \
         patch("services.indexing_service.vector_db",        test_vector_db), \
         patch("services.indexing_service.embedding_service", fake_embedding_service):
        events = await _run_indexing(svc, project_id, tmp_path)

    assert "error" not in _event_types(events)
    complete = next((e for e in events if e["event"] == "indexing_complete"), None)
    if complete:
        # Only small.py should be indexed — huge.py skipped
        assert complete["data"]["files_indexed"] <= 1


async def test_git_dir_excluded(tmp_path, test_db, test_vector_db,
                                 fake_embedding_service):
    project_id = str(uuid.uuid4())
    _register_project(test_db, project_id, tmp_path)
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("fake git config")
    _write(tmp_path, "real.py", SIMPLE_PY)

    from services.indexing_service import IndexingService
    svc = IndexingService()

    with patch("services.indexing_service.db",               test_db), \
         patch("services.indexing_service.vector_db",        test_vector_db), \
         patch("services.indexing_service.embedding_service", fake_embedding_service):
        events = await _run_indexing(svc, project_id, tmp_path)

    # .git/config must not appear in indexed files
    progress_events = [e for e in events if e["event"] == "indexing_progress"]
    for e in progress_events:
        assert ".git" not in e["data"].get("current_file", "")


# ═══════════════════════════════════════════════════════════════════════════════
# Non-existent clone path
# ═══════════════════════════════════════════════════════════════════════════════

async def test_missing_clone_path_yields_error(tmp_path, test_db, test_vector_db,
                                                fake_embedding_service):
    project_id = str(uuid.uuid4())
    missing = tmp_path / "does_not_exist"

    from services.indexing_service import IndexingService
    svc = IndexingService()

    with patch("services.indexing_service.db",               test_db), \
         patch("services.indexing_service.vector_db",        test_vector_db), \
         patch("services.indexing_service.embedding_service", fake_embedding_service):
        events = await _run_indexing(svc, project_id, missing)

    assert "error" in _event_types(events)
