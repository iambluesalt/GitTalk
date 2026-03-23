"""
File and directory exclusion rules for repository analysis.
"""
from pathlib import Path

# Directories to always skip during analysis
EXCLUDED_DIRS: set[str] = {
    "node_modules", "vendor", ".git", "__pycache__", "dist", "build",
    ".venv", "venv", ".tox", ".mypy_cache", ".pytest_cache",
    ".next", ".nuxt", ".svelte-kit", "target", "out",
    ".eggs", ".cache", "bower_components",
}

# Binary/non-text file extensions to skip
EXCLUDED_EXTENSIONS: set[str] = {
    # Images
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg", ".webp", ".tiff",
    # Compiled
    ".pyc", ".pyo", ".class", ".o", ".obj", ".exe", ".dll", ".so", ".dylib",
    ".wasm",
    # Archives
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".rar", ".7z",
    # Fonts
    ".woff", ".woff2", ".ttf", ".otf", ".eot",
    # Media
    ".mp3", ".mp4", ".avi", ".mov", ".wav", ".flac",
    # Data
    ".db", ".sqlite", ".sqlite3", ".bin", ".dat",
    # Documents
    ".pdf", ".doc", ".docx", ".xls", ".xlsx",
    # Maps
    ".map",
}

# Specific filenames to exclude (lock files, etc.)
EXCLUDED_FILES: set[str] = {
    "package-lock.json", "poetry.lock", "Cargo.lock", "yarn.lock",
    "pnpm-lock.yaml", "composer.lock", "Gemfile.lock", "go.sum",
    ".DS_Store", "Thumbs.db",
}

# Patterns that may contain secrets — skip for security
SECRET_PATTERNS: set[str] = {
    ".env", ".env.local", ".env.production", ".env.development",
    "credentials.json", "credentials.yaml", "credentials.yml",
    ".aws", ".ssh", "id_rsa", "id_ed25519",
}

# Secret file extensions
SECRET_EXTENSIONS: set[str] = {
    ".key", ".pem", ".p12", ".pfx", ".keystore",
}


def load_gitignore_patterns(repo_path: Path) -> list[str]:
    """
    Parse .gitignore file and return patterns.

    Args:
        repo_path: Root path of the repository

    Returns:
        List of gitignore pattern strings
    """
    gitignore = repo_path / ".gitignore"
    if not gitignore.exists():
        return []

    patterns = []
    try:
        for line in gitignore.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                patterns.append(line)
    except OSError:
        pass
    return patterns


def _matches_gitignore(file_path: Path, repo_root: Path, patterns: list[str]) -> bool:
    """Check if a path matches any gitignore pattern (simple matching)."""
    rel = file_path.relative_to(repo_root)
    rel_str = rel.as_posix()

    for pattern in patterns:
        clean = pattern.rstrip("/")
        # Directory pattern
        if pattern.endswith("/"):
            for part in rel.parts:
                if part == clean:
                    return True
        # Wildcard extension pattern like *.log
        elif clean.startswith("*."):
            ext = clean[1:]  # e.g. ".log"
            if rel.suffix == ext:
                return True
        # Exact name match against any path component
        elif "/" not in clean:
            if clean in rel.parts or rel.name == clean:
                return True
        # Path prefix match
        elif rel_str.startswith(clean) or rel_str == clean:
            return True
    return False


def should_exclude(file_path: Path, repo_root: Path, gitignore_patterns: list[str]) -> bool:
    """
    Master exclusion check for a file path.

    Args:
        file_path: Absolute path to the file
        repo_root: Root path of the repository
        gitignore_patterns: Parsed gitignore patterns

    Returns:
        True if the file should be excluded
    """
    # Check directory exclusions
    for part in file_path.relative_to(repo_root).parts:
        if part in EXCLUDED_DIRS:
            return True

    # Check extension exclusions
    if file_path.suffix.lower() in EXCLUDED_EXTENSIONS:
        return True

    # Check filename exclusions
    if file_path.name in EXCLUDED_FILES:
        return True

    # Check secret patterns
    if file_path.name in SECRET_PATTERNS:
        return True
    if file_path.suffix.lower() in SECRET_EXTENSIONS:
        return True

    # Check gitignore patterns
    if gitignore_patterns and _matches_gitignore(file_path, repo_root, gitignore_patterns):
        return True

    return False
