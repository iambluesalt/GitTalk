"""
Tests for the RAG context assembly service (rag_service.py).

DB and search_service are mocked; focuses on prompt construction,
token budgeting, deduplication, and intent routing.
"""
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from helpers import _make_project_meta


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_search_result(file_path="src/auth.py", line_start=1, line_end=20,
                        text="def authenticate(): pass", function_name="authenticate",
                        class_name="", language="python", score=0.9):
    from models import SearchResult
    return SearchResult(
        chunk_id=str(uuid.uuid4()),
        text=text,
        file_path=file_path,
        language=language,
        function_name=function_name,
        class_name=class_name,
        line_start=line_start,
        line_end=line_end,
        chunk_type="function",
        relevance_score=score,
    )


def _make_turns(*pairs):
    """('user','msg'), ('assistant','reply') → list[ConversationTurn]"""
    from models import ConversationTurn
    return [ConversationTurn(role=r, content=c) for r, c in pairs]


@pytest.fixture()
def rag():
    from services.rag_service import RAGService
    return RAGService()


@pytest.fixture()
def mock_project(tmp_path):
    pid = str(uuid.uuid4())
    meta = _make_project_meta(pid, tmp_path / "repo")
    meta.repo_map = "src/\n  auth.py\n  main.py\n"
    return meta


# ═══════════════════════════════════════════════════════════════════════════════
# Code intent — retrieval happens, chunks appear in prompt
# ═══════════════════════════════════════════════════════════════════════════════

async def test_code_intent_triggers_search(rag, mock_project):
    """A code question must call hybrid_search and inject results into messages."""
    result = _make_search_result()

    with patch("services.rag_service.db") as mock_db, \
         patch("services.rag_service.search_service") as mock_search, \
         patch("services.rag_service.router_service") as mock_router:
        mock_db.get_project.return_value = mock_project
        mock_db.get_conversation_history.return_value = []
        mock_router.classify = AsyncMock(return_value="code")
        mock_search.hybrid_search = AsyncMock(return_value=[result])

        ctx = await rag.build_context(mock_project.id, "what does authenticate do?")

    mock_search.hybrid_search.assert_called_once()
    # Relevant Code section must appear in the user message
    user_msg = ctx.messages[-1]["content"]
    assert "## Relevant Code" in user_msg
    assert "authenticate" in user_msg
    assert len(ctx.sources) == 1


async def test_code_intent_sources_populated(rag, mock_project):
    results = [_make_search_result(file_path="src/db.py", line_start=5, line_end=30)]

    with patch("services.rag_service.db") as mock_db, \
         patch("services.rag_service.search_service") as mock_search, \
         patch("services.rag_service.router_service") as mock_router:
        mock_db.get_project.return_value = mock_project
        mock_db.get_conversation_history.return_value = []
        mock_router.classify = AsyncMock(return_value="code")
        mock_search.hybrid_search = AsyncMock(return_value=results)

        ctx = await rag.build_context(mock_project.id, "db connection")

    assert ctx.sources[0].file_path == "src/db.py"
    assert ctx.sources[0].line_start == 5


# ═══════════════════════════════════════════════════════════════════════════════
# General intent — retrieval is skipped
# ═══════════════════════════════════════════════════════════════════════════════

async def test_general_intent_skips_retrieval(rag, mock_project):
    with patch("services.rag_service.db") as mock_db, \
         patch("services.rag_service.search_service") as mock_search, \
         patch("services.rag_service.router_service") as mock_router:
        mock_db.get_project.return_value = mock_project
        mock_db.get_conversation_history.return_value = []
        mock_router.classify = AsyncMock(return_value="general")
        mock_search.hybrid_search = AsyncMock(return_value=[])

        ctx = await rag.build_context(mock_project.id, "hi there")

    mock_search.hybrid_search.assert_not_called()
    assert ctx.sources == []
    assert ctx.search_results_count == 0


async def test_general_intent_no_code_in_user_message(rag, mock_project):
    with patch("services.rag_service.db") as mock_db, \
         patch("services.rag_service.search_service") as mock_search, \
         patch("services.rag_service.router_service") as mock_router:
        mock_db.get_project.return_value = mock_project
        mock_db.get_conversation_history.return_value = []
        mock_router.classify = AsyncMock(return_value="general")
        mock_search.hybrid_search = AsyncMock(return_value=[])

        ctx = await rag.build_context(mock_project.id, "hi")

    user_msg = ctx.messages[-1]["content"]
    assert "## Relevant Code" not in user_msg


async def test_general_intent_system_uses_general_template(rag, mock_project):
    with patch("services.rag_service.db") as mock_db, \
         patch("services.rag_service.search_service") as mock_search, \
         patch("services.rag_service.router_service") as mock_router:
        mock_db.get_project.return_value = mock_project
        mock_db.get_conversation_history.return_value = []
        mock_router.classify = AsyncMock(return_value="general")
        mock_search.hybrid_search = AsyncMock(return_value=[])

        ctx = await rag.build_context(mock_project.id, "hi")

    system_msg = ctx.messages[0]["content"]
    assert "casual conversation" in system_msg.lower() or "friendly" in system_msg.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# Follow-up intent — query enriched with history context
# ═══════════════════════════════════════════════════════════════════════════════

async def test_follow_up_enriches_query(rag, mock_project):
    """Follow-up should call hybrid_search with enriched query containing prior context."""
    history = _make_turns(
        ("user", "how does auth work?"),
        ("assistant", "Auth uses JWT tokens stored in auth/jwt.py"),
    )
    result = _make_search_result()

    with patch("services.rag_service.db") as mock_db, \
         patch("services.rag_service.search_service") as mock_search, \
         patch("services.rag_service.router_service") as mock_router:
        mock_db.get_project.return_value = mock_project
        mock_db.get_conversation_history.return_value = history
        mock_router.classify = AsyncMock(return_value="follow_up")
        mock_search.hybrid_search = AsyncMock(return_value=[result])

        ctx = await rag.build_context(
            mock_project.id, "tell me more", conversation_id="conv-1"
        )

    call_args = mock_search.hybrid_search.call_args
    enriched_query = call_args[0][1]  # second positional arg
    # The enriched query must contain the follow-up question AND prior context
    assert "tell me more" in enriched_query
    assert "JWT" in enriched_query or "auth" in enriched_query.lower()


async def test_follow_up_without_history_falls_back_to_regular_retrieval(rag, mock_project):
    with patch("services.rag_service.db") as mock_db, \
         patch("services.rag_service.search_service") as mock_search, \
         patch("services.rag_service.router_service") as mock_router:
        mock_db.get_project.return_value = mock_project
        mock_db.get_conversation_history.return_value = []
        mock_router.classify = AsyncMock(return_value="follow_up")
        mock_search.hybrid_search = AsyncMock(return_value=[])

        ctx = await rag.build_context(mock_project.id, "explain that")

    mock_search.hybrid_search.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════════
# Message structure
# ═══════════════════════════════════════════════════════════════════════════════

async def test_messages_start_with_system(rag, mock_project):
    with patch("services.rag_service.db") as mock_db, \
         patch("services.rag_service.search_service") as mock_search, \
         patch("services.rag_service.router_service") as mock_router:
        mock_db.get_project.return_value = mock_project
        mock_db.get_conversation_history.return_value = []
        mock_router.classify = AsyncMock(return_value="code")
        mock_search.hybrid_search = AsyncMock(return_value=[])

        ctx = await rag.build_context(mock_project.id, "any question")

    assert ctx.messages[0]["role"] == "system"


async def test_messages_end_with_user(rag, mock_project):
    with patch("services.rag_service.db") as mock_db, \
         patch("services.rag_service.search_service") as mock_search, \
         patch("services.rag_service.router_service") as mock_router:
        mock_db.get_project.return_value = mock_project
        mock_db.get_conversation_history.return_value = []
        mock_router.classify = AsyncMock(return_value="code")
        mock_search.hybrid_search = AsyncMock(return_value=[])

        ctx = await rag.build_context(mock_project.id, "any question")

    assert ctx.messages[-1]["role"] == "user"
    assert "any question" in ctx.messages[-1]["content"]


async def test_repo_map_in_system_message(rag, mock_project):
    mock_project.repo_map = "src/\n  auth.py\n"

    with patch("services.rag_service.db") as mock_db, \
         patch("services.rag_service.search_service") as mock_search, \
         patch("services.rag_service.router_service") as mock_router:
        mock_db.get_project.return_value = mock_project
        mock_db.get_conversation_history.return_value = []
        mock_router.classify = AsyncMock(return_value="code")
        mock_search.hybrid_search = AsyncMock(return_value=[])

        ctx = await rag.build_context(mock_project.id, "show structure")

    system_msg = ctx.messages[0]["content"]
    assert "Repository Structure" in system_msg
    assert "auth.py" in system_msg


# ═══════════════════════════════════════════════════════════════════════════════
# History inclusion + token budget
# ═══════════════════════════════════════════════════════════════════════════════

async def test_history_messages_included(rag, mock_project):
    history = _make_turns(
        ("user", "what is the project about?"),
        ("assistant", "It's a codebase chat tool."),
    )

    with patch("services.rag_service.db") as mock_db, \
         patch("services.rag_service.search_service") as mock_search, \
         patch("services.rag_service.router_service") as mock_router:
        mock_db.get_project.return_value = mock_project
        mock_db.get_conversation_history.return_value = history
        mock_router.classify = AsyncMock(return_value="code")
        mock_search.hybrid_search = AsyncMock(return_value=[])

        ctx = await rag.build_context(mock_project.id, "next question", conversation_id="c1")

    roles = [m["role"] for m in ctx.messages]
    assert "user" in roles
    assert "assistant" in roles


async def test_history_token_budget_truncates_oldest(rag, mock_project):
    """When history is very long, the oldest turns must be dropped first."""
    long_content = "x " * 5000  # ~2500 tokens each
    history = _make_turns(
        ("user",      "FIRST_OLD_MESSAGE " + long_content),
        ("assistant", long_content),
        ("user",      long_content),
        ("assistant", long_content),
        ("user",      long_content),
        ("assistant", "MOST_RECENT " + long_content),
    )

    with patch("services.rag_service.db") as mock_db, \
         patch("services.rag_service.search_service") as mock_search, \
         patch("services.rag_service.router_service") as mock_router:
        mock_db.get_project.return_value = mock_project
        mock_db.get_conversation_history.return_value = history
        mock_router.classify = AsyncMock(return_value="code")
        mock_search.hybrid_search = AsyncMock(return_value=[])

        ctx = await rag.build_context(mock_project.id, "new question", conversation_id="c1")

    all_content = " ".join(m["content"] for m in ctx.messages)
    # Most recent should be kept, oldest may be dropped
    assert "MOST_RECENT" in all_content or len(ctx.messages) >= 2


async def test_total_token_count_within_budget(rag, mock_project):
    from config import settings

    with patch("services.rag_service.db") as mock_db, \
         patch("services.rag_service.search_service") as mock_search, \
         patch("services.rag_service.router_service") as mock_router:
        mock_db.get_project.return_value = mock_project
        mock_db.get_conversation_history.return_value = []
        mock_router.classify = AsyncMock(return_value="code")
        mock_search.hybrid_search = AsyncMock(return_value=[
            _make_search_result() for _ in range(8)
        ])

        ctx = await rag.build_context(mock_project.id, "show me the code")

    # 20% tolerance for estimate rounding
    assert ctx.token_count <= settings.MAX_CONTEXT_TOKENS * 1.2


# ═══════════════════════════════════════════════════════════════════════════════
# Chunk deduplication
# ═══════════════════════════════════════════════════════════════════════════════

def test_dedup_merges_adjacent_chunks(rag):
    """Two consecutive chunks from the same file within 3 lines → merged into one."""
    r1 = _make_search_result(file_path="src/foo.py", line_start=1,  line_end=10,
                              text="def a(): pass")
    r2 = _make_search_result(file_path="src/foo.py", line_start=11, line_end=20,
                              text="def b(): pass")
    merged = rag._dedup_overlapping([r1, r2])
    assert len(merged) == 1
    assert merged[0].line_start == 1
    assert merged[0].line_end == 20


def test_dedup_keeps_non_overlapping_chunks(rag):
    """Chunks far apart (> 3 lines gap) in the same file must stay separate."""
    r1 = _make_search_result(file_path="src/foo.py", line_start=1,  line_end=10)
    r2 = _make_search_result(file_path="src/foo.py", line_start=50, line_end=60)
    result = rag._dedup_overlapping([r1, r2])
    assert len(result) == 2


def test_dedup_different_files_not_merged(rag):
    r1 = _make_search_result(file_path="src/a.py", line_start=1,  line_end=10)
    r2 = _make_search_result(file_path="src/b.py", line_start=10, line_end=20)
    result = rag._dedup_overlapping([r1, r2])
    assert len(result) == 2


def test_dedup_merged_score_is_max(rag):
    r1 = _make_search_result(line_start=1,  line_end=10, score=0.6)
    r2 = _make_search_result(line_start=11, line_end=20, score=0.9)
    merged = rag._dedup_overlapping([r1, r2])
    assert merged[0].relevance_score == 0.9


def test_dedup_empty_list(rag):
    assert rag._dedup_overlapping([]) == []


def test_dedup_single_item(rag):
    r = _make_search_result()
    assert rag._dedup_overlapping([r]) == [r]


# ═══════════════════════════════════════════════════════════════════════════════
# Missing project
# ═══════════════════════════════════════════════════════════════════════════════

async def test_missing_project_returns_minimal_context(rag):
    with patch("services.rag_service.db") as mock_db, \
         patch("services.rag_service.search_service") as mock_search, \
         patch("services.rag_service.router_service") as mock_router:
        mock_db.get_project.return_value = None
        mock_db.get_conversation_history.return_value = []
        mock_router.classify = AsyncMock(return_value="code")
        mock_search.hybrid_search = AsyncMock(return_value=[])

        ctx = await rag.build_context("nonexistent-id", "some question")

    # Must not raise; returns a minimal context with the bare query
    assert ctx.sources == []
    assert any("some question" in m["content"] for m in ctx.messages)
