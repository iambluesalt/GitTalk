"""
AST-based code chunker.
Splits code files at function/class boundaries using tree-sitter.
Falls back to text-based chunking for non-code files.
"""
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from config import settings
from logger import logger
from services.treesitter_service import TreeSitterService, treesitter_service


@dataclass
class CodeChunk:
    """A chunk of code or text ready for embedding."""
    id: str
    text: str
    file_path: str          # relative to repo root
    language: str
    function_name: str | None
    class_name: str | None
    line_start: int
    line_end: int
    chunk_type: str         # function, class, method, module, text
    token_estimate: int     # rough estimate: len(text) // 4


class CodeChunker:
    """Chunks code files at AST boundaries."""

    def __init__(self, ts_service: TreeSitterService | None = None):
        self.ts = ts_service or treesitter_service
        self.max_tokens = settings.CHUNK_MAX_TOKENS

    def chunk_file(
        self,
        file_path: str,
        extension: str,
        source_bytes: bytes,
        tree=None,
    ) -> list[CodeChunk]:
        """
        Chunk a file using AST boundaries.

        Args:
            file_path: Relative path from repo root
            extension: File extension (e.g. ".py")
            source_bytes: Raw file content as bytes
            tree: Pre-parsed tree-sitter tree (optional)

        Returns:
            List of CodeChunks
        """
        language = self.ts.get_lang_name(extension)

        # If we can't parse with tree-sitter, fall back to text chunking
        if language is None or tree is None:
            detected_lang = _detect_text_language(extension)
            return self._chunk_text_file(file_path, source_bytes, detected_lang)

        source_text = source_bytes.decode("utf-8", errors="replace")
        source_lines = source_text.split("\n")

        # Extract all code elements
        functions, classes, imports = self.ts.extract_all(tree, source_bytes, language)

        chunks: list[CodeChunk] = []

        # Track which lines are covered by functions/classes
        covered_lines: set[int] = set()

        # Track classes chunked as a whole (methods already included)
        fully_chunked_classes: set[str] = set()

        # Process classes (full class as one chunk)
        for cls in classes:
            cls_text = cls.text.decode("utf-8", errors="replace")
            token_est = len(cls_text) // 4

            if token_est <= self.max_tokens:
                fully_chunked_classes.add(cls.name)
                chunks.append(self._make_chunk(
                    text=cls_text,
                    file_path=file_path,
                    language=language,
                    function_name=None,
                    class_name=cls.name,
                    line_start=cls.line_start,
                    line_end=cls.line_end,
                    chunk_type="class",
                ))
                for ln in range(cls.line_start, cls.line_end + 1):
                    covered_lines.add(ln)
            else:
                # Large class: mark the class itself, methods are handled separately
                # Add class header (first few lines before first method)
                header_end = cls.line_start
                for fn in functions:
                    if fn.class_name == cls.name:
                        header_end = fn.line_start - 1
                        break

                if header_end > cls.line_start:
                    header_text = "\n".join(
                        source_lines[cls.line_start - 1 : header_end]
                    )
                    if header_text.strip():
                        chunks.append(self._make_chunk(
                            text=header_text,
                            file_path=file_path,
                            language=language,
                            function_name=None,
                            class_name=cls.name,
                            line_start=cls.line_start,
                            line_end=header_end,
                            chunk_type="class",
                        ))
                # Mark class lines as covered (methods handle their own lines)
                for ln in range(cls.line_start, cls.line_end + 1):
                    covered_lines.add(ln)

        # Process functions and methods
        for fn in functions:
            # Skip methods already included in a fully-chunked class
            if fn.class_name and fn.class_name in fully_chunked_classes:
                continue

            fn_text = fn.text.decode("utf-8", errors="replace")
            token_est = len(fn_text) // 4

            if token_est <= self.max_tokens:
                chunks.append(self._make_chunk(
                    text=fn_text,
                    file_path=file_path,
                    language=language,
                    function_name=fn.name,
                    class_name=fn.class_name,
                    line_start=fn.line_start,
                    line_end=fn.line_end,
                    chunk_type=fn.node_type,
                ))
            else:
                # Large function: split into smaller pieces
                sub_chunks = self._split_large_block(
                    fn_text, file_path, language,
                    fn.name, fn.class_name, fn.line_start, fn.node_type,
                )
                chunks.extend(sub_chunks)

            for ln in range(fn.line_start, fn.line_end + 1):
                covered_lines.add(ln)

        # Module-level code: lines not covered by any function/class
        module_lines = []
        current_start = None
        for i, line in enumerate(source_lines, start=1):
            if i not in covered_lines:
                if current_start is None:
                    current_start = i
                module_lines.append((i, line))
            else:
                if module_lines:
                    self._flush_module_chunk(
                        module_lines, file_path, language, imports, chunks
                    )
                    module_lines = []
                    current_start = None

        # Flush any remaining module lines
        if module_lines:
            self._flush_module_chunk(
                module_lines, file_path, language, imports, chunks
            )

        return chunks

    def _flush_module_chunk(
        self,
        lines: list[tuple[int, str]],
        file_path: str,
        language: str,
        imports: list[str],
        chunks: list[CodeChunk],
    ):
        """Create a module chunk from uncovered lines."""
        text = "\n".join(line for _, line in lines)
        if not text.strip():
            return

        token_est = len(text) // 4
        if token_est < 15:
            return  # Skip trivially short chunks

        # Skip chunks that are purely imports/requires (no meaningful code)
        non_empty = [l.strip() for _, l in lines if l.strip() and not l.strip().startswith("#") and not l.strip().startswith("//")]
        if non_empty and all(
            l.startswith(("import ", "from ", "require(", "const {", "export {"))
            or l.startswith(("using ", "#include", "package "))
            for l in non_empty
        ):
            return  # Pure import block — not useful for retrieval

        line_start = lines[0][0]
        line_end = lines[-1][0]

        if token_est <= self.max_tokens:
            chunks.append(self._make_chunk(
                text=text,
                file_path=file_path,
                language=language,
                function_name=None,
                class_name=None,
                line_start=line_start,
                line_end=line_end,
                chunk_type="module",
            ))
        else:
            # Split large module-level code
            sub = self._split_text_by_lines(
                text, file_path, language, None, None, line_start, "module"
            )
            chunks.extend(sub)

    def _split_large_block(
        self,
        text: str,
        file_path: str,
        language: str,
        function_name: str | None,
        class_name: str | None,
        base_line: int,
        chunk_type: str,
    ) -> list[CodeChunk]:
        """Split a large code block into smaller chunks along line boundaries."""
        return self._split_text_by_lines(
            text, file_path, language, function_name, class_name,
            base_line, chunk_type,
        )

    def _split_text_by_lines(
        self,
        text: str,
        file_path: str,
        language: str,
        function_name: str | None,
        class_name: str | None,
        base_line: int,
        chunk_type: str,
    ) -> list[CodeChunk]:
        """Split text into chunks that fit within token limits, splitting at line boundaries.
        Carries forward overlap lines between chunks to preserve context at boundaries."""
        lines = text.split("\n")
        chunks = []
        current_lines: list[str] = []
        current_start = base_line
        overlap = settings.CHUNK_OVERLAP_LINES

        for i, line in enumerate(lines):
            current_lines.append(line)
            current_text = "\n".join(current_lines)
            token_est = len(current_text) // 4

            if token_est >= self.max_tokens and len(current_lines) > 1:
                # Emit chunk without the last line
                emit_lines = current_lines[:-1]
                chunk_text = "\n".join(emit_lines)
                if chunk_text.strip():
                    chunks.append(self._make_chunk(
                        text=chunk_text,
                        file_path=file_path,
                        language=language,
                        function_name=function_name,
                        class_name=class_name,
                        line_start=current_start,
                        line_end=current_start + len(emit_lines) - 1,
                        chunk_type=chunk_type,
                    ))
                # Carry forward overlap lines from end of emitted chunk
                carry = emit_lines[-overlap:] if overlap > 0 else []
                current_lines = carry + [line]
                current_start = base_line + i - len(carry)

        # Flush remaining
        if current_lines:
            chunk_text = "\n".join(current_lines)
            if chunk_text.strip():
                chunks.append(self._make_chunk(
                    text=chunk_text,
                    file_path=file_path,
                    language=language,
                    function_name=function_name,
                    class_name=class_name,
                    line_start=current_start,
                    line_end=current_start + len(current_lines) - 1,
                    chunk_type=chunk_type,
                ))

        return chunks

    def _chunk_text_file(
        self, file_path: str, source_bytes: bytes, language: str
    ) -> list[CodeChunk]:
        """Chunk a non-code file (markdown, yaml, json, etc.) using text splitting."""
        text = source_bytes.decode("utf-8", errors="replace")
        if not text.strip():
            return []

        token_est = len(text) // 4
        if token_est <= self.max_tokens:
            return [self._make_chunk(
                text=text,
                file_path=file_path,
                language=language,
                function_name=None,
                class_name=None,
                line_start=1,
                line_end=len(text.split("\n")),
                chunk_type="text",
            )]

        # Split at paragraph/section boundaries
        return self._split_text_by_paragraphs(text, file_path, language)

    def _split_text_by_paragraphs(
        self, text: str, file_path: str, language: str
    ) -> list[CodeChunk]:
        """Split text at double-newline boundaries."""
        paragraphs = text.split("\n\n")
        chunks = []
        current_parts: list[str] = []
        current_start_line = 1
        lines_so_far = 0

        for para in paragraphs:
            current_parts.append(para)
            combined = "\n\n".join(current_parts)
            token_est = len(combined) // 4

            if token_est >= self.max_tokens and len(current_parts) > 1:
                # Emit without last paragraph
                chunk_text = "\n\n".join(current_parts[:-1])
                chunk_lines = chunk_text.count("\n") + 1
                if chunk_text.strip():
                    chunks.append(self._make_chunk(
                        text=chunk_text,
                        file_path=file_path,
                        language=language,
                        function_name=None,
                        class_name=None,
                        line_start=current_start_line,
                        line_end=current_start_line + chunk_lines - 1,
                        chunk_type="text",
                    ))
                current_start_line += chunk_lines + 1  # +1 for the double newline
                current_parts = [para]

        # Flush remaining
        if current_parts:
            chunk_text = "\n\n".join(current_parts)
            chunk_lines = chunk_text.count("\n") + 1
            if chunk_text.strip():
                chunks.append(self._make_chunk(
                    text=chunk_text,
                    file_path=file_path,
                    language=language,
                    function_name=None,
                    class_name=None,
                    line_start=current_start_line,
                    line_end=current_start_line + chunk_lines - 1,
                    chunk_type="text",
                ))

        return chunks

    def _make_chunk(
        self,
        text: str,
        file_path: str,
        language: str,
        function_name: str | None,
        class_name: str | None,
        line_start: int,
        line_end: int,
        chunk_type: str,
    ) -> CodeChunk:
        """Create a CodeChunk with a unique ID and contextual header."""
        contextualized = self._add_context_header(
            text, file_path, language, function_name, class_name, chunk_type
        )
        return CodeChunk(
            id=str(uuid.uuid4()),
            text=contextualized,
            file_path=file_path,
            language=language,
            function_name=function_name,
            class_name=class_name,
            line_start=line_start,
            line_end=line_end,
            chunk_type=chunk_type,
            token_estimate=len(contextualized) // 4,
        )

    def _add_context_header(
        self,
        text: str,
        file_path: str,
        language: str,
        function_name: str | None,
        class_name: str | None,
        chunk_type: str,
    ) -> str:
        """
        Prepend a short metadata header to chunk text before embedding.

        This dramatically improves retrieval quality because the embedding model
        can associate code with its file path, function/class name, and role.
        """
        parts = [f"# File: {file_path}"]
        if class_name and function_name:
            parts.append(f"# Method: {class_name}.{function_name}")
        elif function_name:
            parts.append(f"# Function: {function_name}")
        elif class_name:
            parts.append(f"# Class: {class_name}")
        if language:
            parts.append(f"# Language: {language}")
        if chunk_type and chunk_type not in ("function", "method", "class"):
            parts.append(f"# Type: {chunk_type}")

        header = "\n".join(parts)
        return f"{header}\n\n{text}"


def _detect_text_language(extension: str) -> str:
    """Detect language for non-tree-sitter files."""
    ext = extension.lower()
    mapping = {
        ".md": "markdown", ".mdx": "markdown",
        ".json": "json", ".yaml": "yaml", ".yml": "yaml",
        ".toml": "toml", ".xml": "xml", ".html": "html",
        ".css": "css", ".scss": "scss", ".less": "less",
        ".sh": "shell", ".bash": "shell",
        ".sql": "sql", ".ini": "ini", ".cfg": "ini",
        ".txt": "text", ".rst": "restructuredtext",
    }
    return mapping.get(ext, "text")


# Global instance
code_chunker = CodeChunker()
