"""
Tools exposed to the LangGraph code agent.

`make_tools(project_id)` returns per-request tool instances that close over the
project — `project_id` and the on-disk clone path must never be LLM-visible
parameters, so they are bound at construction time instead of being arguments.
"""
from collections import defaultdict
from pathlib import Path

from langchain_core.tools import BaseTool, tool

from logger import logger
from models import CodeReference, SearchResult
from services.search_service import search_service
from storage.metadata_db import db
from utils.exclusions import EXCLUDED_DIRS, load_gitignore_patterns, should_exclude


# Per-call output ceilings. Tool results accumulate in the message list across
# hops, so each individual call has to stay small enough that a 4-6 hop run
# still fits in MAX_CONTEXT_TOKENS.
SEARCH_RESULT_MAX_CHARS = 12_000
READ_FILE_MAX_LINES = 400
READ_FILE_MAX_CHARS = 20_000
LIST_FILES_MAX_ENTRIES = 200


# ============================================================================
# Result formatting / conversion
# ============================================================================

def result_to_reference(result: SearchResult) -> CodeReference:
    """Convert a SearchResult into the CodeReference shape the SSE layer sends."""
    return CodeReference(
        file_path=result.file_path,
        line_start=result.line_start,
        line_end=result.line_end,
        code_snippet=result.text[:500],
        relevance_score=result.relevance_score,
    )


def format_search_results(results: list[SearchResult]) -> str:
    """Render search results as numbered, fenced code blocks for the model."""
    if not results:
        return "No matching code found. Try different search terms."

    parts: list[str] = []
    used = 0
    for i, result in enumerate(results, 1):
        location = f"{result.file_path}:{result.line_start}-{result.line_end}"
        label = result.function_name or result.class_name or result.chunk_type
        header = f"### [{i}] {location}"
        if label:
            header += f" — {label}"

        body = result.text
        remaining = SEARCH_RESULT_MAX_CHARS - used - len(header) - len(result.language) - 10
        if remaining <= 0:
            parts.append(f"... ({len(results) - i + 1} more results omitted)")
            break
        if len(body) > remaining:
            body = body[:remaining] + "\n... (truncated)"

        block = f"{header}\n```{result.language}\n{body}\n```"
        parts.append(block)
        used += len(block)

    return "\n\n".join(parts)


# ============================================================================
# Chunk deduplication
# ============================================================================

def dedup_overlapping(results: list[SearchResult]) -> list[SearchResult]:
    """Merge results from the same file with overlapping or adjacent line ranges.

    A large function split across chunks tends to return both halves (similar
    embeddings). Merging them means the model sees one continuous block rather
    than duplicated fragments.
    """
    by_file: dict[str, list[SearchResult]] = defaultdict(list)
    for r in results:
        by_file[r.file_path].append(r)

    merged: list[SearchResult] = []
    for file_results in by_file.values():
        file_results.sort(key=lambda r: r.line_start)

        current = file_results[0]
        for next_r in file_results[1:]:
            # Overlap or adjacent (within 3 lines)
            if next_r.line_start <= current.line_end + 3:
                current = _merge_results(current, next_r)
            else:
                merged.append(current)
                current = next_r
        merged.append(current)

    merged.sort(key=lambda r: r.relevance_score, reverse=True)
    return merged


def _merge_results(a: SearchResult, b: SearchResult) -> SearchResult:
    """Merge two overlapping SearchResults into one, stitching their text."""
    a_lines = a.text.split("\n")
    b_lines = b.text.split("\n")

    overlap_start = b.line_start - a.line_start
    if overlap_start < len(a_lines):
        b_offset = a.line_end - b.line_start + 1
        if b_offset < len(b_lines):
            merged_text = a.text + "\n" + "\n".join(b_lines[b_offset:])
        else:
            merged_text = a.text
    else:
        merged_text = a.text + "\n" + b.text

    return SearchResult(
        chunk_id=a.chunk_id,
        text=merged_text,
        file_path=a.file_path,
        language=a.language,
        function_name=a.function_name or b.function_name,
        class_name=a.class_name or b.class_name,
        line_start=min(a.line_start, b.line_start),
        line_end=max(a.line_end, b.line_end),
        chunk_type=a.chunk_type,
        relevance_score=max(a.relevance_score, b.relevance_score),
    )


# ============================================================================
# Path safety
# ============================================================================

def _resolve_in_repo(root: Path, rel_path: str) -> Path | None:
    """Resolve a model-supplied path inside the clone, or None if it escapes."""
    cleaned = (rel_path or "").strip().strip('"').strip("'").replace("\\", "/")
    cleaned = cleaned.lstrip("/")
    try:
        candidate = (root / cleaned).resolve()
        candidate.relative_to(root)
    except (ValueError, OSError):
        return None
    return candidate


# ============================================================================
# Tool factory
# ============================================================================

def make_tools(project_id: str) -> list[BaseTool]:
    """Build the tool set for one chat request, bound to a single project."""
    project = db.get_project(project_id)
    root: Path | None = None
    if project:
        try:
            root = Path(project.clone_path).resolve()
        except OSError:
            root = None
    if root is None or not root.is_dir():
        logger.warning(f"No clone path on disk for project {project_id}; file tools disabled")
        root = None

    gitignore_patterns = load_gitignore_patterns(root) if root else []

    @tool(response_format="content_and_artifact", parse_docstring=True)
    async def search_codebase(query: str) -> tuple[str, list[SearchResult]]:
        """Search the repository for code relevant to a natural-language query.

        Combines semantic (embedding) and keyword search over the indexed code.
        Call this before answering any question about the codebase, and call it
        again with different wording if the first results are not enough.

        Args:
            query: What to look for, e.g. "how are JWT tokens validated" or
                "database connection setup".
        """
        results = await search_service.hybrid_search(project_id, query)
        if results:
            results = dedup_overlapping(results)
        logger.info(f"search_codebase({query[:60]!r}) -> {len(results)} results")
        return format_search_results(results), results

    @tool(parse_docstring=True)
    def read_file(
        path: str,
        line_start: int | None = None,
        line_end: int | None = None,
    ) -> str:
        """Read a file from the repository, optionally a specific line range.

        Use this when a search snippet is truncated, when you need the code
        surrounding a match, or when you already know which file to look at.

        Args:
            path: Repository-relative path, e.g. "src/auth/session.py".
            line_start: First line to read, 1-indexed and inclusive. Omit for the
                start of the file.
            line_end: Last line to read, inclusive. Omit to read to the end.
        """
        if root is None:
            return "Error: the repository files are not available on disk."

        target = _resolve_in_repo(root, path)
        if target is None:
            return f"Error: '{path}' is outside the repository."
        if not target.exists() or not target.is_file():
            return f"Error: '{path}' does not exist in the repository."
        if should_exclude(target, root, gitignore_patterns):
            return f"Error: '{path}' is excluded from this repository's readable files."

        try:
            content = target.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            return f"Error: could not read '{path}': {e}"

        lines = content.split("\n")
        total = len(lines)

        start = max(1, line_start or 1)
        end = min(total, line_end or total)
        if start > total:
            return f"Error: '{path}' has only {total} lines; line {start} is out of range."
        if end < start:
            end = total

        truncated = False
        if end - start + 1 > READ_FILE_MAX_LINES:
            end = start + READ_FILE_MAX_LINES - 1
            truncated = True

        selected = lines[start - 1 : end]
        body = "\n".join(f"{start + i}\t{line}" for i, line in enumerate(selected))
        if len(body) > READ_FILE_MAX_CHARS:
            body = body[:READ_FILE_MAX_CHARS]
            truncated = True

        header = f"{path} (lines {start}-{end} of {total})"
        if truncated:
            body += "\n... (truncated — request a narrower range to see more)"
        return f"{header}\n{body}"

    @tool(parse_docstring=True)
    def list_files(dir: str = "") -> str:
        """List the files and subdirectories of a repository directory.

        Use this to discover where something lives before searching or reading.
        Directories are suffixed with '/'.

        Args:
            dir: Repository-relative directory, e.g. "src/services". Omit or pass
                "" for the repository root.
        """
        if root is None:
            return "Error: the repository files are not available on disk."

        target = _resolve_in_repo(root, dir)
        if target is None:
            return f"Error: '{dir}' is outside the repository."
        if not target.exists() or not target.is_dir():
            return f"Error: '{dir}' is not a directory in the repository."

        try:
            children = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except OSError as e:
            return f"Error: could not list '{dir}': {e}"

        entries: list[str] = []
        for child in children:
            if child.is_dir():
                if child.name in EXCLUDED_DIRS or child.name.startswith("."):
                    continue
                entries.append(f"{child.name}/")
            else:
                if should_exclude(child, root, gitignore_patterns):
                    continue
                entries.append(child.name)

            if len(entries) >= LIST_FILES_MAX_ENTRIES:
                entries.append(f"... (truncated at {LIST_FILES_MAX_ENTRIES} entries)")
                break

        if not entries:
            return f"{dir or '.'} is empty (or contains only excluded files)."
        return f"{dir or '.'}/\n" + "\n".join(entries)

    return [search_codebase, read_file, list_files]
