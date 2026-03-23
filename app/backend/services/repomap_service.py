"""
Repository map generator.
Creates a compact structural summary of a codebase for LLM context.
"""
from pathlib import Path

from logger import logger
from services.treesitter_service import TreeSitterService, treesitter_service
from utils.exclusions import load_gitignore_patterns, should_exclude

# Maximum repo map size in characters
REPO_MAP_MAX_CHARS = 50_000


class RepoMapService:
    """Generates a tree-style structural summary of a repository."""

    def __init__(self, ts_service: TreeSitterService | None = None):
        self.ts = ts_service or treesitter_service

    def generate_repo_map(self, project_id: str, clone_path: Path) -> str:
        """
        Generate a repo map showing file tree + function/class signatures.

        Args:
            project_id: Project identifier
            clone_path: Root path of the cloned repository

        Returns:
            Formatted repo map string
        """
        gitignore_patterns = load_gitignore_patterns(clone_path)

        # Collect file entries: (relative_path, signatures)
        entries: list[tuple[str, list[tuple[str, str]]]] = []

        for file_path in sorted(clone_path.rglob("*")):
            if not file_path.is_file():
                continue
            if should_exclude(file_path, clone_path, gitignore_patterns):
                continue

            rel_path = file_path.relative_to(clone_path).as_posix()
            ext = file_path.suffix.lower()
            language = self.ts.get_lang_name(ext)

            if language is None:
                # Non-parseable file — just show the filename
                entries.append((rel_path, []))
                continue

            try:
                source_bytes = file_path.read_bytes()
                tree = self.ts.parse_file(str(file_path), source_bytes, ext)
                if tree is None:
                    entries.append((rel_path, []))
                    continue

                functions, classes, imports = self.ts.extract_all(
                    tree, source_bytes, language
                )

                signatures: list[tuple[str, str]] = []  # (indent_prefix, signature)

                # Group methods under their classes
                class_methods: dict[str, list] = {}
                standalone_fns = []

                for fn in functions:
                    if fn.class_name:
                        class_methods.setdefault(fn.class_name, []).append(fn)
                    else:
                        standalone_fns.append(fn)

                for cls in classes:
                    signatures.append(("├── ", f"class {cls.name}"))
                    methods = class_methods.get(cls.name, [])
                    for i, method in enumerate(methods):
                        is_last = i == len(methods) - 1
                        prefix = "│   └── " if is_last else "│   ├── "
                        sig = self._format_function_sig(method, language)
                        signatures.append((prefix, sig))

                for i, fn in enumerate(standalone_fns):
                    is_last = i == len(standalone_fns) - 1
                    prefix = "└── " if is_last else "├── "
                    sig = self._format_function_sig(fn, language)
                    signatures.append((prefix, sig))

                entries.append((rel_path, signatures))
            except Exception as e:
                logger.debug(f"Failed to map {rel_path}: {e}")
                entries.append((rel_path, []))

        # Build the output string
        lines: list[str] = []
        total_chars = 0

        for rel_path, signatures in entries:
            line = rel_path
            if total_chars + len(line) + 1 > REPO_MAP_MAX_CHARS:
                lines.append("... (truncated)")
                break
            lines.append(line)
            total_chars += len(line) + 1

            for prefix, sig in signatures:
                line = f"{prefix}{sig}"
                if total_chars + len(line) + 1 > REPO_MAP_MAX_CHARS:
                    lines.append("... (truncated)")
                    break
                lines.append(line)
                total_chars += len(line) + 1
            else:
                if signatures:
                    lines.append("")  # blank line between files with signatures
                continue
            break

        result = "\n".join(lines)
        logger.info(
            f"Generated repo map for {project_id}: "
            f"{len(entries)} files, {len(result)} chars"
        )
        return result

    def _format_function_sig(self, fn, language: str) -> str:
        """Format a function/method signature for the repo map."""
        sig = fn.signature
        # Remove body indicators
        for trim in (" {", " =>", " where"):
            idx = sig.find(trim)
            if idx > 0:
                sig = sig[:idx]

        # For Python, remove the colon at end
        if language == "python" and sig.endswith(":"):
            sig = sig[:-1].rstrip()

        # Limit length
        if len(sig) > 120:
            sig = sig[:120] + "..."

        return sig


# Global instance
repomap_service = RepoMapService()
