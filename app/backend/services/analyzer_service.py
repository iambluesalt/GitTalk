"""
Repository analysis service.
Walks directory trees, counts files/LOC, detects languages.
Uses tree-sitter for function/class/import counting on supported languages.
"""
from pathlib import Path

from logger import logger
from models import RepositoryAnalysis, LanguageStats
from services.treesitter_service import treesitter_service
from utils.exclusions import load_gitignore_patterns, should_exclude


# Extension-to-language mapping — each variant gets its own entry for visibility
EXTENSION_TO_LANGUAGE: dict[str, str] = {
    # Python
    ".py": "python", ".pyi": "python", ".pyw": "python",
    # JavaScript
    ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".jsx": "jsx",
    # TypeScript
    ".ts": "typescript", ".mts": "typescript", ".cts": "typescript",
    ".tsx": "tsx",
    # Web
    ".html": "html", ".htm": "html",
    ".css": "css", ".scss": "scss", ".sass": "sass", ".less": "less",
    ".vue": "vue", ".svelte": "svelte",
    # Go
    ".go": "go",
    # Rust
    ".rs": "rust",
    # Java / Kotlin
    ".java": "java", ".kt": "kotlin", ".kts": "kotlin",
    # C / C++
    ".c": "c", ".h": "c-header",
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp",
    ".hpp": "cpp-header", ".hh": "cpp-header",
    # C#
    ".cs": "csharp",
    # Ruby
    ".rb": "ruby", ".erb": "erb",
    # PHP
    ".php": "php",
    # Swift
    ".swift": "swift",
    # Shell
    ".sh": "shell", ".bash": "shell", ".zsh": "shell", ".fish": "shell",
    ".ps1": "powershell", ".psm1": "powershell",
    # Config / Data
    ".json": "json", ".yaml": "yaml", ".yml": "yaml", ".toml": "toml",
    ".xml": "xml", ".ini": "ini", ".cfg": "ini",
    # Markdown / Docs
    ".md": "markdown", ".mdx": "mdx", ".rst": "restructuredtext",
    # SQL
    ".sql": "sql",
    # Lua
    ".lua": "lua",
    # R
    ".r": "r", ".R": "r",
    # Scala
    ".scala": "scala",
    # Dart
    ".dart": "dart",
    # Elixir / Erlang
    ".ex": "elixir", ".exs": "elixir", ".erl": "erlang",
    # Haskell
    ".hs": "haskell",
    # Dockerfile
    ".dockerfile": "dockerfile",
    # Protobuf
    ".proto": "protobuf",
    # Zig
    ".zig": "zig",
}

# Maps analysis language names → tree-sitter language names for AST parsing
# (only needed where they differ)
_ANALYSIS_TO_TS_LANG: dict[str, str] = {
    "jsx": "javascript",
    "tsx": "tsx",
    "c-header": "c",
    "cpp-header": "cpp",
}


class AnalyzerService:
    """Analyzes cloned repositories for file statistics and language detection."""

    def analyze_repository(self, clone_path: Path) -> RepositoryAnalysis:
        """
        Walk directory tree and produce analysis.

        Args:
            clone_path: Root path of the cloned repository

        Returns:
            RepositoryAnalysis with file counts, LOC, and language stats
        """
        gitignore_patterns = load_gitignore_patterns(clone_path)

        total_files = 0
        total_lines = 0
        file_types: dict[str, int] = {}
        languages: dict[str, LanguageStats] = {}
        total_size_bytes = 0

        for file_path in clone_path.rglob("*"):
            if not file_path.is_file():
                continue

            if should_exclude(file_path, clone_path, gitignore_patterns):
                continue

            total_files += 1

            try:
                total_size_bytes += file_path.stat().st_size
            except OSError:
                pass

            # Count by extension
            ext = file_path.suffix.lower()
            if ext:
                file_types[ext] = file_types.get(ext, 0) + 1

            # Detect language and count lines
            language = self._detect_language(file_path)
            line_count = self._count_lines(file_path)
            total_lines += line_count

            if language:
                if language not in languages:
                    languages[language] = LanguageStats()
                stats = languages[language]
                stats.file_count += 1
                stats.lines_of_code += line_count

                # AST-level counts for tree-sitter supported languages
                ts_lang = _ANALYSIS_TO_TS_LANG.get(language) or treesitter_service.get_lang_name(ext)
                if ts_lang:
                    try:
                        source_bytes = file_path.read_bytes()
                        tree = treesitter_service.parse_file(
                            str(file_path), source_bytes, ext
                        )
                        if tree:
                            funcs, classes, imports = treesitter_service.extract_all(
                                tree, source_bytes, ts_lang
                            )
                            stats.functions += len(funcs)
                            stats.classes += len(classes)
                            stats.imports += len(imports)
                    except Exception:
                        pass

        # Determine primary language by LOC
        primary_language = None
        if languages:
            primary_language = max(languages, key=lambda k: languages[k].lines_of_code)

        repo_size_mb = round(total_size_bytes / (1024 * 1024), 2)

        analysis = RepositoryAnalysis(
            total_files=total_files,
            total_lines=total_lines,
            repository_size_mb=repo_size_mb,
            file_types=file_types,
            languages=languages,
            primary_language=primary_language,
        )

        logger.info(
            f"Repository analysis: {total_files} files, {total_lines} lines, "
            f"{repo_size_mb}MB, primary language: {primary_language}"
        )
        return analysis

    def _detect_language(self, file_path: Path) -> str | None:
        """
        Map file extension to language name.

        Args:
            file_path: Path to the file

        Returns:
            Language name or None if unknown
        """
        ext = file_path.suffix.lower()
        if ext in EXTENSION_TO_LANGUAGE:
            return EXTENSION_TO_LANGUAGE[ext]

        # Handle special filenames
        name = file_path.name.lower()
        special_names = {
            "dockerfile": "dockerfile",
            "makefile": "makefile",
            "cmakelists.txt": "cmake",
            "gemfile": "ruby",
            "rakefile": "ruby",
            "vagrantfile": "ruby",
            "jenkinsfile": "groovy",
        }
        return special_names.get(name)

    def _count_lines(self, file_path: Path) -> int:
        """
        Count lines of code in a file, handling encoding errors gracefully.

        Args:
            file_path: Path to the file

        Returns:
            Number of lines (0 if file can't be read)
        """
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
            return len(content.splitlines())
        except (OSError, UnicodeDecodeError):
            try:
                content = file_path.read_text(encoding="latin-1")
                return len(content.splitlines())
            except Exception:
                return 0
