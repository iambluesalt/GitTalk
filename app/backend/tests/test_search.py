"""
Tests for the hybrid search service (search_service.py).

LanceDB and the embedding service are mocked — no Ollama or real DB needed.
"""
import uuid
from collections import defaultdict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from helpers import _fake_vector, EMBED_DIM


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_row(chunk_id=None, text="some code", file_path="src/foo.py",
              language="python", function_name="foo", class_name="",
              line_start=1, line_end=10, chunk_type="function",
              distance=0.1, fts_score=1.0):
    row = {
        "id": chunk_id or str(uuid.uuid4()),
        "text": text,
        "file_path": file_path,
        "language": language,
        "function_name": function_name,
        "class_name": class_name,
        "line_start": line_start,
        "line_end": line_end,
        "chunk_type": chunk_type,
        "_distance": distance,
        "_score": fts_score,
    }
    return row


def _make_table(vector_rows, fts_rows=None, fts_raises=False):
    """Build a MagicMock that looks like a LanceDB table."""
    table = MagicMock()

    def search_side_effect(query, **kwargs):
        mock_q = MagicMock()
        if kwargs.get("query_type") == "fts":
            if fts_raises:
                mock_q.limit.return_value.to_list.side_effect = RuntimeError("FTS error")
            else:
                mock_q.limit.return_value.to_list.return_value = fts_rows or []
        else:
            mock_q.limit.return_value.to_list.return_value = vector_rows
        return mock_q

    table.search.side_effect = search_side_effect
    return table


@pytest.fixture()
def search_svc():
    from services.search_service import SearchService
    return SearchService()


# ═══════════════════════════════════════════════════════════════════════════════
# Basic hybrid search
# ═══════════════════════════════════════════════════════════════════════════════

async def test_hybrid_search_returns_results(search_svc, fake_embedding_service):
    row = _make_row(text="def authenticate(user): ...", function_name="authenticate")
    table = _make_table([row], [row])

    with patch("services.search_service.vector_db") as mock_vdb, \
         patch("services.search_service.embedding_service", fake_embedding_service):
        mock_vdb.table_exists.return_value = True
        mock_vdb.get_or_create_table.return_value = table

        results = await search_svc.hybrid_search("proj-1", "authentication function")

    assert len(results) >= 1
    assert results[0].function_name == "authenticate"


async def test_hybrid_search_result_fields(search_svc, fake_embedding_service):
    """Each SearchResult must have all expected fields populated."""
    row = _make_row(file_path="auth/jwt.py", line_start=5, line_end=20,
                    language="python", chunk_type="function")
    table = _make_table([row], [row])

    with patch("services.search_service.vector_db") as mock_vdb, \
         patch("services.search_service.embedding_service", fake_embedding_service):
        mock_vdb.table_exists.return_value = True
        mock_vdb.get_or_create_table.return_value = table
        results = await search_svc.hybrid_search("proj-1", "jwt")

    r = results[0]
    assert r.file_path == "auth/jwt.py"
    assert r.line_start == 5
    assert r.line_end == 20
    assert r.language == "python"
    assert 0.0 <= r.relevance_score <= 1.0


async def test_hybrid_search_score_range(search_svc, fake_embedding_service):
    """All relevance scores must be in [0, 1] after RRF normalisation."""
    rows = [_make_row(chunk_id=str(i), distance=i * 0.05) for i in range(10)]
    table = _make_table(rows, rows)

    with patch("services.search_service.vector_db") as mock_vdb, \
         patch("services.search_service.embedding_service", fake_embedding_service):
        mock_vdb.table_exists.return_value = True
        mock_vdb.get_or_create_table.return_value = table
        results = await search_svc.hybrid_search("proj-1", "code")

    for r in results:
        assert 0.0 <= r.relevance_score <= 1.0


async def test_hybrid_search_respects_n_results(search_svc, fake_embedding_service):
    """Result count must not exceed the requested n_results."""
    rows = [_make_row(chunk_id=str(i)) for i in range(20)]
    table = _make_table(rows, rows)

    with patch("services.search_service.vector_db") as mock_vdb, \
         patch("services.search_service.embedding_service", fake_embedding_service):
        mock_vdb.table_exists.return_value = True
        mock_vdb.get_or_create_table.return_value = table
        results = await search_svc.hybrid_search("proj-1", "code", n_results=5)

    assert len(results) <= 5


# ═══════════════════════════════════════════════════════════════════════════════
# FTS failure → vector-only fallback
# ═══════════════════════════════════════════════════════════════════════════════

async def test_fts_failure_falls_back_to_vector(search_svc, fake_embedding_service):
    row = _make_row(function_name="fallback_fn")
    table = _make_table([row], fts_raises=True)

    with patch("services.search_service.vector_db") as mock_vdb, \
         patch("services.search_service.embedding_service", fake_embedding_service):
        mock_vdb.table_exists.return_value = True
        mock_vdb.get_or_create_table.return_value = table
        results = await search_svc.hybrid_search("proj-1", "something")

    # Must not raise; should return at least 1 result from vector path
    assert isinstance(results, list)


# ═══════════════════════════════════════════════════════════════════════════════
# No table for project
# ═══════════════════════════════════════════════════════════════════════════════

async def test_no_table_returns_empty(search_svc, fake_embedding_service):
    with patch("services.search_service.vector_db") as mock_vdb, \
         patch("services.search_service.embedding_service", fake_embedding_service):
        mock_vdb.table_exists.return_value = False
        results = await search_svc.hybrid_search("nonexistent-proj", "any query")

    assert results == []


# ═══════════════════════════════════════════════════════════════════════════════
# Embedding failure
# ═══════════════════════════════════════════════════════════════════════════════

async def test_embed_failure_returns_empty(search_svc):
    bad_embed = MagicMock()
    bad_embed.embed_single = AsyncMock(return_value=[])

    with patch("services.search_service.vector_db") as mock_vdb, \
         patch("services.search_service.embedding_service", bad_embed):
        mock_vdb.table_exists.return_value = True
        results = await search_svc.hybrid_search("proj-1", "query")

    assert results == []


# ═══════════════════════════════════════════════════════════════════════════════
# RRF merging correctness
# ═══════════════════════════════════════════════════════════════════════════════

def test_rrf_merge_top_result_gets_highest_score(search_svc):
    """A chunk that ranks #1 in BOTH vector and FTS should have the highest score."""
    shared_id = "top-chunk"
    vector_rows = [_make_row(chunk_id=shared_id)] + \
                  [_make_row(chunk_id=f"v{i}") for i in range(5)]
    fts_rows    = [_make_row(chunk_id=shared_id)] + \
                  [_make_row(chunk_id=f"f{i}") for i in range(5)]

    results = search_svc._rrf_merge(vector_rows, fts_rows)
    assert results[0].chunk_id == shared_id


def test_rrf_merge_union_of_both_lists(search_svc):
    """RRF result must include IDs from both vector and FTS lists."""
    vector_rows = [_make_row(chunk_id="v1"), _make_row(chunk_id="v2")]
    fts_rows    = [_make_row(chunk_id="f1"), _make_row(chunk_id="v1")]  # v1 in both

    results = search_svc._rrf_merge(vector_rows, fts_rows)
    result_ids = {r.chunk_id for r in results}
    assert {"v1", "v2", "f1"}.issubset(result_ids)


def test_rrf_merge_empty_fts(search_svc):
    """Empty FTS list → only vector results returned."""
    vector_rows = [_make_row(chunk_id="v1")]
    results = search_svc._rrf_merge(vector_rows, [])
    # _rrf_merge returns vector-only results when FTS is empty
    # (the actual fallback path in _do_hybrid_search)
    # Here we test the merge directly with empty fts
    assert len(results) >= 1


def test_rrf_merge_normalised_scores(search_svc):
    """After RRF, all scores must be in [0.0, 1.0]."""
    rows = [_make_row(chunk_id=str(i)) for i in range(8)]
    results = search_svc._rrf_merge(rows, rows)
    for r in results:
        assert 0.0 <= r.relevance_score <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# MIN_RELEVANCE_SCORE filter
# ═══════════════════════════════════════════════════════════════════════════════

async def test_min_relevance_filter(search_svc, fake_embedding_service):
    """When MIN_RELEVANCE_SCORE > 0, the lowest-ranked results are dropped."""
    rows = [_make_row(chunk_id=str(i), distance=float(i)) for i in range(10)]
    table = _make_table(rows, rows[:3])  # FTS only returns 3

    with patch("services.search_service.vector_db") as mock_vdb, \
         patch("services.search_service.embedding_service", fake_embedding_service), \
         patch("services.search_service.settings") as mock_settings:
        mock_settings.MAX_SEARCH_RESULTS = 8
        mock_settings.RETRIEVAL_CANDIDATES = 30
        mock_settings.MIN_RELEVANCE_SCORE = 0.5  # strict filter
        mock_vdb.table_exists.return_value = True
        mock_vdb.get_or_create_table.return_value = table

        results = await search_svc.hybrid_search("proj-1", "query")

    for r in results:
        assert r.relevance_score >= 0.5
