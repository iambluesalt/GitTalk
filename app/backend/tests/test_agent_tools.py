"""
Tests for the agent's tools (agent_tools.py).

Files are real (a tmp_path repo); only the project lookup and hybrid search are
mocked. The path guards in particular are worth exercising against a real
filesystem — they're the only thing standing between a model-supplied string and
arbitrary reads.
"""
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


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


@pytest.fixture()
def repo(tmp_path):
    """A small on-disk repo with excluded dirs, secrets and lock files."""
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "node_modules").mkdir()
    (root / ".hidden").mkdir()
    (root / "node_modules" / "junk.js").write_text("junk")
    (root / "src" / "auth.py").write_text(
        "\n".join(f"line{i}" for i in range(1, 51)), encoding="utf-8"
    )
    (root / "README.md").write_text("# hi", encoding="utf-8")
    (root / ".env").write_text("SECRET=nope", encoding="utf-8")
    (root / "package-lock.json").write_text("{}", encoding="utf-8")
    (root / "logo.png").write_bytes(b"\x89PNG")
    return root


@pytest.fixture()
def tools(repo):
    """The three tools, bound to the tmp repo, keyed by name."""
    from services.agent_tools import make_tools

    project = MagicMock()
    project.clone_path = str(repo)
    with patch("services.agent_tools.db") as mock_db:
        mock_db.get_project.return_value = project
        built = make_tools("proj-1")
    return {t.name: t for t in built}


# ═══════════════════════════════════════════════════════════════════════════════
# search_codebase
# ═══════════════════════════════════════════════════════════════════════════════

async def test_search_returns_formatted_text_and_raw_artifact(tools):
    """Formatted snippets go to the model; raw SearchResults ride in the artifact."""
    results = [_make_search_result(file_path="src/db.py", line_start=5, line_end=30)]

    with patch("services.agent_tools.search_service") as mock_search:
        mock_search.hybrid_search = AsyncMock(return_value=results)
        msg = await tools["search_codebase"].ainvoke(
            {"name": "search_codebase", "args": {"query": "db"},
             "id": "1", "type": "tool_call"}
        )

    assert "src/db.py:5-30" in msg.content
    assert "```python" in msg.content
    assert [r.file_path for r in msg.artifact] == ["src/db.py"]


async def test_search_passes_project_id_not_model_supplied(tools):
    """project_id is closed over, never taken from the model."""
    assert "project_id" not in tools["search_codebase"].args

    with patch("services.agent_tools.search_service") as mock_search:
        mock_search.hybrid_search = AsyncMock(return_value=[])
        await tools["search_codebase"].ainvoke({"query": "anything"})

    assert mock_search.hybrid_search.call_args[0][0] == "proj-1"


async def test_search_empty_results_tells_model_to_retry(tools):
    with patch("services.agent_tools.search_service") as mock_search:
        mock_search.hybrid_search = AsyncMock(return_value=[])
        out = await tools["search_codebase"].ainvoke({"query": "nothing"})

    assert "No matching code" in out


async def test_search_dedups_adjacent_chunks(tools):
    """Adjacent chunks from one file are merged before the model ever sees them."""
    results = [
        _make_search_result(file_path="src/foo.py", line_start=1, line_end=10, text="def a(): pass"),
        _make_search_result(file_path="src/foo.py", line_start=11, line_end=20, text="def b(): pass"),
    ]

    with patch("services.agent_tools.search_service") as mock_search:
        mock_search.hybrid_search = AsyncMock(return_value=results)
        msg = await tools["search_codebase"].ainvoke(
            {"name": "search_codebase", "args": {"query": "foo"},
             "id": "1", "type": "tool_call"}
        )

    assert len(msg.artifact) == 1
    assert (msg.artifact[0].line_start, msg.artifact[0].line_end) == (1, 20)


def test_format_search_results_respects_char_ceiling():
    """A single huge result is truncated rather than blowing the context budget."""
    from services.agent_tools import SEARCH_RESULT_MAX_CHARS, format_search_results

    huge = _make_search_result(text="x" * (SEARCH_RESULT_MAX_CHARS * 2))
    out = format_search_results([huge])

    assert "truncated" in out
    assert len(out) < SEARCH_RESULT_MAX_CHARS + 500


def test_format_search_results_omits_overflow_results():
    from services.agent_tools import SEARCH_RESULT_MAX_CHARS, format_search_results

    many = [_make_search_result(text="y" * 5000) for _ in range(10)]
    out = format_search_results(many)

    assert "more results omitted" in out
    assert len(out) < SEARCH_RESULT_MAX_CHARS + 500


# ═══════════════════════════════════════════════════════════════════════════════
# read_file
# ═══════════════════════════════════════════════════════════════════════════════

def test_read_file_returns_numbered_lines(tools):
    out = tools["read_file"].invoke({"path": "src/auth.py"})
    assert "src/auth.py (lines 1-50 of 50)" in out
    assert "1\tline1" in out


def test_read_file_line_range_is_inclusive(tools):
    out = tools["read_file"].invoke({"path": "src/auth.py", "line_start": 3, "line_end": 5})
    assert "3\tline3" in out
    assert "5\tline5" in out
    assert "line6" not in out
    assert "line2" not in out


def test_read_file_caps_long_ranges(tools, repo):
    from services.agent_tools import READ_FILE_MAX_LINES

    (repo / "big.py").write_text(
        "\n".join(f"l{i}" for i in range(READ_FILE_MAX_LINES + 200)), encoding="utf-8"
    )
    out = tools["read_file"].invoke({"path": "big.py"})

    assert "truncated" in out
    assert out.count("\n") <= READ_FILE_MAX_LINES + 5


def test_read_file_rejects_traversal(tools):
    for escape in ("../outside.py", "../../etc/hosts", "src/../../nope.py"):
        out = tools["read_file"].invoke({"path": escape})
        assert "outside the repository" in out, escape


def test_read_file_rejects_secrets(tools):
    out = tools["read_file"].invoke({"path": ".env"})
    assert "excluded" in out
    assert "SECRET" not in out


def test_read_file_rejects_excluded_lockfiles(tools):
    out = tools["read_file"].invoke({"path": "package-lock.json"})
    assert "excluded" in out


def test_read_file_missing_file(tools):
    out = tools["read_file"].invoke({"path": "src/nope.py"})
    assert "does not exist" in out


def test_read_file_directory_is_not_readable(tools):
    out = tools["read_file"].invoke({"path": "src"})
    assert "does not exist" in out


def test_read_file_start_beyond_end_of_file(tools):
    out = tools["read_file"].invoke({"path": "src/auth.py", "line_start": 9999})
    assert "out of range" in out


def test_read_file_end_before_start_reads_to_eof(tools):
    out = tools["read_file"].invoke({"path": "src/auth.py", "line_start": 10, "line_end": 2})
    assert "lines 10-50" in out


def test_read_file_without_clone_path_reports_cleanly():
    """A project whose clone is gone must degrade, not raise."""
    from services.agent_tools import make_tools

    with patch("services.agent_tools.db") as mock_db:
        mock_db.get_project.return_value = None
        built = {t.name: t for t in make_tools("missing")}

    assert "not available on disk" in built["read_file"].invoke({"path": "a.py"})
    assert "not available on disk" in built["list_files"].invoke({"dir": ""})


# ═══════════════════════════════════════════════════════════════════════════════
# list_files
# ═══════════════════════════════════════════════════════════════════════════════

def test_list_files_marks_directories(tools):
    out = tools["list_files"].invoke({"dir": ""})
    assert "src/" in out
    assert "README.md" in out


def test_list_files_filters_noise(tools):
    """Excluded dirs, hidden dirs, binaries, lock files and secrets stay hidden."""
    out = tools["list_files"].invoke({"dir": ""})
    for hidden in ("node_modules", ".hidden", "package-lock.json", ".env", "logo.png"):
        assert hidden not in out, hidden


def test_list_files_subdirectory(tools):
    out = tools["list_files"].invoke({"dir": "src"})
    assert "auth.py" in out


def test_list_files_rejects_traversal(tools):
    out = tools["list_files"].invoke({"dir": "../.."})
    assert "outside the repository" in out


def test_list_files_missing_directory(tools):
    out = tools["list_files"].invoke({"dir": "does/not/exist"})
    assert "not a directory" in out


def test_list_files_on_a_file_is_rejected(tools):
    out = tools["list_files"].invoke({"dir": "README.md"})
    assert "not a directory" in out


def test_list_files_caps_entry_count(tools, repo):
    from services.agent_tools import LIST_FILES_MAX_ENTRIES

    big = repo / "many"
    big.mkdir()
    for i in range(LIST_FILES_MAX_ENTRIES + 50):
        (big / f"f{i}.py").write_text("x", encoding="utf-8")

    out = tools["list_files"].invoke({"dir": "many"})
    assert "truncated at" in out


# ═══════════════════════════════════════════════════════════════════════════════
# Deduplication (moved here from rag_service)
# ═══════════════════════════════════════════════════════════════════════════════

def test_dedup_merges_adjacent_chunks():
    from services.agent_tools import dedup_overlapping

    r1 = _make_search_result(file_path="src/foo.py", line_start=1, line_end=10, text="def a(): pass")
    r2 = _make_search_result(file_path="src/foo.py", line_start=11, line_end=20, text="def b(): pass")
    merged = dedup_overlapping([r1, r2])

    assert len(merged) == 1
    assert (merged[0].line_start, merged[0].line_end) == (1, 20)


def test_dedup_keeps_distant_chunks_separate():
    from services.agent_tools import dedup_overlapping

    r1 = _make_search_result(file_path="src/foo.py", line_start=1, line_end=10)
    r2 = _make_search_result(file_path="src/foo.py", line_start=50, line_end=60)
    assert len(dedup_overlapping([r1, r2])) == 2


def test_dedup_never_merges_across_files():
    from services.agent_tools import dedup_overlapping

    r1 = _make_search_result(file_path="src/a.py", line_start=1, line_end=10)
    r2 = _make_search_result(file_path="src/b.py", line_start=10, line_end=20)
    assert len(dedup_overlapping([r1, r2])) == 2


def test_dedup_merged_score_is_max():
    from services.agent_tools import dedup_overlapping

    r1 = _make_search_result(line_start=1, line_end=10, score=0.6)
    r2 = _make_search_result(line_start=11, line_end=20, score=0.9)
    assert dedup_overlapping([r1, r2])[0].relevance_score == 0.9


def test_dedup_empty_and_single():
    from services.agent_tools import dedup_overlapping

    assert dedup_overlapping([]) == []
    r = _make_search_result()
    assert dedup_overlapping([r]) == [r]


def test_result_to_reference_truncates_snippet():
    from services.agent_tools import result_to_reference

    ref = result_to_reference(_make_search_result(text="z" * 2000))
    assert len(ref.code_snippet) == 500
    assert ref.file_path == "src/auth.py"
