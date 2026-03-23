"""
URL validation and sanitization utilities.
"""
import re
from urllib.parse import urlparse


def validate_github_url(url: str) -> tuple[str, str, str]:
    """
    Parse and validate a GitHub repository URL.

    Args:
        url: GitHub URL to validate

    Returns:
        Tuple of (normalized_url, owner, repo)

    Raises:
        ValueError: If URL is not a valid GitHub repository URL
    """
    parsed = urlparse(str(url))

    if parsed.hostname not in ("github.com", "www.github.com"):
        raise ValueError(f"URL must be from github.com, got: {parsed.hostname}")

    if parsed.scheme not in ("https", "http"):
        raise ValueError(f"URL must use https or http scheme, got: {parsed.scheme}")

    # Extract path segments: /owner/repo[.git][/...]
    path = parsed.path.strip("/")
    if path.endswith(".git"):
        path = path[:-4]

    segments = path.split("/")
    if len(segments) < 2:
        raise ValueError("URL must contain owner and repository name")

    owner = segments[0]
    repo = segments[1]

    if not owner or not repo:
        raise ValueError("Owner and repository name cannot be empty")

    # Validate characters (GitHub allows alphanumeric, hyphens, underscores, dots)
    pattern = re.compile(r"^[a-zA-Z0-9._-]+$")
    if not pattern.match(owner):
        raise ValueError(f"Invalid owner name: {owner}")
    if not pattern.match(repo):
        raise ValueError(f"Invalid repository name: {repo}")

    normalized_url = f"https://github.com/{owner}/{repo}"
    return normalized_url, owner, repo


def sanitize_clone_dir_name(owner: str, repo: str) -> str:
    """
    Generate a safe directory name for cloning.

    Args:
        owner: Repository owner
        repo: Repository name

    Returns:
        Safe directory name in format '{owner}-{repo}'
    """
    safe = re.sub(r"[^a-zA-Z0-9._-]", "_", f"{owner}-{repo}")
    return safe
