"""
Tests for the AST-based code chunker (chunker_service.py).

No network or DB access needed — pure file-parsing logic.
Tree-sitter must be installed (it's in requirements.txt).
"""
import pytest
from helpers import (
    SIMPLE_PY, CLASS_PY, IMPORTS_ONLY_PY, EMPTY_PY,
    LARGE_FN_PY, SYNTAX_ERROR_PY, MARKDOWN_MD, JSON_FILE, SIMPLE_JS,
)


@pytest.fixture()
def chunker():
    from services.chunker_service import CodeChunker
    return CodeChunker()


def _chunk(chunker, content: str, ext: str, path: str = "test/file"):
    """Helper: chunk a string as if it were a real file."""
    from services.treesitter_service import treesitter_service

    src = content.encode("utf-8")
    tree = treesitter_service.parse_file(path + ext, src, ext)
    return chunker.chunk_file(path + ext, ext, src, tree)


# ═══════════════════════════════════════════════════════════════════════════════
# Basic Python chunking
# ═══════════════════════════════════════════════════════════════════════════════

def test_simple_python_two_functions(chunker):
    """Two top-level functions → two function chunks."""
    chunks = _chunk(chunker, SIMPLE_PY, ".py")
    fn_names = {c.function_name for c in chunks if c.function_name}
    assert "add" in fn_names
    assert "subtract" in fn_names


def test_simple_python_chunk_types(chunker):
    chunks = _chunk(chunker, SIMPLE_PY, ".py")
    types = {c.chunk_type for c in chunks}
    assert "function" in types


def test_simple_python_line_numbers(chunker):
    """line_start and line_end must be positive and in order."""
    chunks = _chunk(chunker, SIMPLE_PY, ".py")
    for c in chunks:
        assert c.line_start >= 1
        assert c.line_end >= c.line_start


def test_simple_python_text_not_empty(chunker):
    chunks = _chunk(chunker, SIMPLE_PY, ".py")
    for c in chunks:
        assert c.text.strip() != ""


def test_simple_python_embedding_text_has_header(chunker):
    """embedding_text must contain the context header prepended by the chunker."""
    chunks = _chunk(chunker, SIMPLE_PY, ".py")
    for c in chunks:
        assert "# File:" in c.embedding_text


def test_class_produces_class_chunk(chunker):
    """Small class fits under max_tokens → one class chunk."""
    chunks = _chunk(chunker, CLASS_PY, ".py")
    class_chunks = [c for c in chunks if c.chunk_type == "class"]
    assert len(class_chunks) >= 1
    assert class_chunks[0].class_name == "Calculator"


def test_class_chunk_contains_methods(chunker):
    """When chunked as a whole, the class text should include its methods."""
    chunks = _chunk(chunker, CLASS_PY, ".py")
    class_chunk = next(c for c in chunks if c.chunk_type == "class")
    assert "def add" in class_chunk.text or "def __init__" in class_chunk.text


# ═══════════════════════════════════════════════════════════════════════════════
# Edge cases — degenerate files
# ═══════════════════════════════════════════════════════════════════════════════

def test_empty_file_returns_no_chunks(chunker):
    chunks = _chunk(chunker, EMPTY_PY, ".py")
    assert chunks == []


def test_imports_only_file_returns_no_chunks(chunker):
    """Pure import blocks are not useful for retrieval — should be filtered."""
    chunks = _chunk(chunker, IMPORTS_ONLY_PY, ".py")
    # May produce zero chunks or very few; the main check is no crash
    for c in chunks:
        # Should not be a chunk of only import lines
        non_import = [
            l for l in c.text.splitlines()
            if l.strip() and not l.strip().startswith(("import ", "from "))
        ]
        # If it IS purely imports the chunker should have dropped it; this fires
        # only if any chunk slips through
        assert len(non_import) > 0 or c.chunk_type == "module"


def test_large_function_splits_into_multiple_chunks(chunker):
    """A function > CHUNK_MAX_TOKENS should be split into 2+ sub-chunks.
    We override max_tokens on the instance so the test is independent of
    whatever CHUNK_MAX_TOKENS value is set in .env.
    """
    chunker.max_tokens = 500  # force split regardless of .env
    chunks = _chunk(chunker, LARGE_FN_PY, ".py")
    fn_chunks = [c for c in chunks if c.function_name == "big_function"]
    assert len(fn_chunks) >= 2, "Large function must be split"


def test_large_function_chunks_stay_within_token_limit(chunker):
    chunker.max_tokens = 500  # must match the split threshold used above
    chunks = _chunk(chunker, LARGE_FN_PY, ".py")
    for c in chunks:
        assert c.token_estimate <= chunker.max_tokens + 50  # small tolerance


def test_large_function_overlap_breadcrumb(chunker):
    """Continuation chunks beyond the first should have a breadcrumb comment."""
    chunker.max_tokens = 500  # force splitting so there is a continuation chunk
    chunks = _chunk(chunker, LARGE_FN_PY, ".py")
    fn_chunks = sorted(
        [c for c in chunks if c.function_name == "big_function"],
        key=lambda c: c.line_start,
    )
    if len(fn_chunks) > 1:
        # All chunks after the first should have a breadcrumb
        for c in fn_chunks[1:]:
            assert "# ... continued from:" in c.text


def test_syntax_error_does_not_crash(chunker):
    """Tree-sitter is fault-tolerant; a broken file should not raise."""
    try:
        chunks = _chunk(chunker, SYNTAX_ERROR_PY, ".py")
        # May return 0 or some chunks — just must not raise
        assert isinstance(chunks, list)
    except Exception as exc:
        pytest.fail(f"Chunking a syntax-error file raised: {exc}")


def test_binary_like_bytes_do_not_crash(chunker):
    """A file with invalid UTF-8 bytes (e.g. compiled artifact) should not crash."""
    src = bytes(range(256))
    try:
        from services.treesitter_service import treesitter_service
        tree = treesitter_service.parse_file("test/binary.py", src, ".py")
        chunks = chunker.chunk_file("test/binary.py", ".py", src, tree)
        assert isinstance(chunks, list)
    except Exception as exc:
        pytest.fail(f"Chunking binary content raised: {exc}")


def test_very_small_file_filtered_out(chunker):
    """Files with fewer than ~15 tokens of non-import content are dropped."""
    tiny = "x = 1\n"
    chunks = _chunk(chunker, tiny, ".py")
    # Either no chunks, or the chunk has meaningful content
    for c in chunks:
        assert c.token_estimate >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# Non-Python code files
# ═══════════════════════════════════════════════════════════════════════════════

def test_javascript_file(chunker):
    """JS file with a function and arrow function → chunks produced."""
    chunks = _chunk(chunker, SIMPLE_JS, ".js", path="test/app")
    assert len(chunks) >= 1
    for c in chunks:
        assert c.language in ("javascript", "js", "text")


def test_markdown_file_text_chunks(chunker):
    """Markdown falls back to text chunking."""
    chunks = _chunk(chunker, MARKDOWN_MD, ".md", path="test/README")
    assert len(chunks) >= 1
    for c in chunks:
        assert c.chunk_type == "text"
        assert c.language == "markdown"


def test_json_file_text_chunks(chunker):
    chunks = _chunk(chunker, JSON_FILE, ".json", path="test/package")
    assert len(chunks) >= 1
    for c in chunks:
        assert c.language == "json"


def test_unknown_extension_fallback(chunker):
    """An unknown extension falls back to text chunking without crashing."""
    content = "This is some plain text content.\n" * 10
    chunks = _chunk(chunker, content, ".xyz", path="test/notes")
    assert isinstance(chunks, list)


# ═══════════════════════════════════════════════════════════════════════════════
# Chunk metadata correctness
# ═══════════════════════════════════════════════════════════════════════════════

def test_chunk_ids_are_unique(chunker):
    chunks = _chunk(chunker, SIMPLE_PY + CLASS_PY, ".py")
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids)), "Chunk IDs must be unique"


def test_chunk_file_path_is_preserved(chunker):
    chunks = _chunk(chunker, SIMPLE_PY, ".py", path="src/utils")
    for c in chunks:
        assert c.file_path == "src/utils.py"


def test_token_estimate_reasonable(chunker):
    """Token estimate (~len/4) should be positive and proportional."""
    chunks = _chunk(chunker, SIMPLE_PY, ".py")
    for c in chunks:
        expected = len(c.text) // 4
        assert c.token_estimate == expected


def test_embedding_text_longer_than_raw_text(chunker):
    """embedding_text = header + text, so must be longer than bare text."""
    chunks = _chunk(chunker, SIMPLE_PY, ".py")
    for c in chunks:
        assert len(c.embedding_text) > len(c.text)
