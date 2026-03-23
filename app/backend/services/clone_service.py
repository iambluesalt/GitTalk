"""
GitHub repository clone service with progress streaming.
"""
import asyncio
import json
import os
import stat
import shutil
import re
from pathlib import Path
from typing import AsyncGenerator, Optional

import httpx

from config import settings
from logger import logger
from models import SSEEvent


def _rm_readonly(func, path, _exc_info):
    """Error handler for shutil.rmtree on Windows — clears read-only flag and retries."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def safe_rmtree(path: Path | None):
    """Remove a directory tree, handling Windows read-only .git files."""
    if path and path.exists():
        shutil.rmtree(str(path), onerror=_rm_readonly)


class CloneService:
    """Handles cloning GitHub repositories with progress reporting."""

    async def check_repo_size(self, owner: str, repo: str, token: Optional[str] = None) -> int:
        """
        Check repository size via GitHub REST API.

        Args:
            owner: Repository owner
            repo: Repository name
            token: Optional GitHub token

        Returns:
            Repository size in KB

        Raises:
            httpx.HTTPStatusError: If the API request fails
        """
        headers = {"Accept": "application/vnd.github.v3+json"}
        if token:
            headers["Authorization"] = f"token {token}"

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://api.github.com/repos/{owner}/{repo}",
                headers=headers,
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("size", 0)  # size in KB

    def check_disk_space(self, path: Path) -> tuple[int, int]:
        """
        Check available disk space.

        Args:
            path: Path to check disk space for

        Returns:
            Tuple of (free_mb, required_mb) where required_mb is MAX_REPO_SIZE_MB
        """
        usage = shutil.disk_usage(str(path))
        free_mb = usage.free // (1024 * 1024)
        return free_mb, settings.MAX_REPO_SIZE_MB

    async def clone_repository(
        self,
        github_url: str,
        clone_dir: Path,
        token: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Clone a repository with progress streaming via SSE events.

        Args:
            github_url: GitHub repository URL
            clone_dir: Target directory for clone
            token: Optional GitHub token for private repos
            timeout: Clone timeout in seconds

        Yields:
            SSE-formatted event strings
        """
        timeout = timeout or settings.CLONE_TIMEOUT_SECONDS
        clone_url = self._build_clone_url(github_url, token)

        # Ensure parent directory exists
        clone_dir.parent.mkdir(parents=True, exist_ok=True)

        # Remove existing directory if present
        if clone_dir.exists():
            safe_rmtree(clone_dir)

        yield SSEEvent(
            event="status",
            data={"message": "Starting clone...", "phase": "clone"}
        ).format()

        process = None
        try:
            process = await asyncio.create_subprocess_exec(
                "git", "clone", "--depth", "1", "--progress", clone_url, str(clone_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # Git writes progress to stderr
            last_percent = -1
            async for line in self._read_stderr(process):
                progress = self._parse_git_progress(line)
                if progress:
                    # Only yield if percentage changed to avoid flooding
                    pct = progress.get("percent", -1)
                    if pct != last_percent:
                        last_percent = pct
                        yield SSEEvent(event="progress", data=progress).format()

            try:
                await asyncio.wait_for(process.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                # Cleanup partial clone
                if clone_dir.exists():
                    safe_rmtree(clone_dir)
                yield SSEEvent(
                    event="error",
                    data={"message": f"Clone timed out after {timeout}s"}
                ).format()
                return

            if process.returncode != 0:
                stderr_remaining = ""
                if process.stderr:
                    remaining = await process.stderr.read()
                    stderr_remaining = remaining.decode("utf-8", errors="replace")
                # Cleanup partial clone
                if clone_dir.exists():
                    safe_rmtree(clone_dir)
                yield SSEEvent(
                    event="error",
                    data={"message": f"Git clone failed: {stderr_remaining.strip()}"}
                ).format()
                return

            yield SSEEvent(
                event="status",
                data={"message": "Clone completed", "phase": "clone_done"}
            ).format()

        except FileNotFoundError:
            yield SSEEvent(
                event="error",
                data={"message": "git is not installed or not in PATH"}
            ).format()
        except Exception as e:
            logger.error(f"Clone error: {e}")
            # Cleanup partial clone
            if clone_dir.exists():
                safe_rmtree(clone_dir)
            yield SSEEvent(
                event="error",
                data={"message": f"Clone failed: {str(e)}"}
            ).format()

    async def _read_stderr(self, process: asyncio.subprocess.Process) -> AsyncGenerator[str, None]:
        """Read lines from process stderr, handling git's carriage-return progress."""
        if not process.stderr:
            return
        buffer = b""
        while True:
            chunk = await process.stderr.read(256)
            if not chunk:
                if buffer:
                    yield buffer.decode("utf-8", errors="replace")
                break
            buffer += chunk
            # Git uses \r for progress updates and \n for final messages
            while b"\r" in buffer or b"\n" in buffer:
                # Find the earliest line break
                r_idx = buffer.find(b"\r")
                n_idx = buffer.find(b"\n")
                if r_idx == -1:
                    idx = n_idx
                elif n_idx == -1:
                    idx = r_idx
                else:
                    idx = min(r_idx, n_idx)
                line = buffer[:idx].decode("utf-8", errors="replace")
                buffer = buffer[idx + 1:]
                if line.strip():
                    yield line.strip()

    def _parse_git_progress(self, line: str) -> Optional[dict]:
        """
        Parse git stderr progress lines into structured data.

        Examples:
            "Receiving objects:  45% (900/2000)"
            "Resolving deltas: 100% (150/150), done."
        """
        # Match patterns like "Phase:  XX% (n/total)"
        match = re.search(r"([\w\s]+):\s+(\d+)%\s+\((\d+)/(\d+)\)", line)
        if match:
            return {
                "phase": match.group(1).strip(),
                "percent": int(match.group(2)),
                "current": int(match.group(3)),
                "total": int(match.group(4)),
            }

        # Match "done." lines
        if "done" in line.lower():
            match_done = re.search(r"([\w\s]+):", line)
            if match_done:
                return {
                    "phase": match_done.group(1).strip(),
                    "percent": 100,
                    "current": 0,
                    "total": 0,
                }

        return None

    def _build_clone_url(self, github_url: str, token: Optional[str] = None) -> str:
        """
        Build the clone URL, injecting token for private repos.

        Args:
            github_url: The public GitHub URL
            token: Optional GitHub token

        Returns:
            Clone URL (with embedded token if provided)
        """
        if token:
            # https://TOKEN@github.com/owner/repo.git
            return github_url.replace("https://", f"https://{token}@") + ".git"
        return github_url + ".git"
