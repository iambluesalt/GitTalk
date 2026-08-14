"""
GitHub repository clone service with progress streaming.
"""
import asyncio
import collections
import json
import os
import queue
import stat
import shutil
import re
import subprocess
import threading
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

        # Run git as a blocking subprocess in a worker thread rather than via
        # asyncio.create_subprocess_exec — on Windows, uvicorn's ProactorEventLoop
        # can hard-crash the interpreter (SIGSEGV) reading overlapped subprocess
        # pipes under git's rapid \r-progress output. A plain thread + blocking
        # reads sidesteps IOCP entirely.
        loop = asyncio.get_running_loop()
        events: "queue.Queue[tuple[str, object]]" = queue.Queue()

        worker = threading.Thread(
            target=self._run_git_clone,
            args=(clone_url, clone_dir, timeout, events),
            daemon=True,
        )
        worker.start()

        last_percent = -1
        try:
            while True:
                kind, payload = await loop.run_in_executor(None, events.get)

                if kind == "line":
                    progress = self._parse_git_progress(payload)
                    if progress:
                        pct = progress.get("percent", -1)
                        if pct != last_percent:
                            last_percent = pct
                            yield SSEEvent(event="progress", data=progress).format()

                elif kind == "not_found":
                    yield SSEEvent(
                        event="error",
                        data={"message": "git is not installed or not in PATH"}
                    ).format()
                    return

                elif kind == "timeout":
                    if clone_dir.exists():
                        safe_rmtree(clone_dir)
                    yield SSEEvent(
                        event="error",
                        data={"message": f"Clone timed out after {timeout}s"}
                    ).format()
                    return

                elif kind == "exception":
                    logger.error(f"Clone error: {payload}")
                    if clone_dir.exists():
                        safe_rmtree(clone_dir)
                    yield SSEEvent(
                        event="error",
                        data={"message": f"Clone failed: {payload}"}
                    ).format()
                    return

                elif kind == "done":
                    returncode, tail = payload
                    if returncode != 0:
                        if clone_dir.exists():
                            safe_rmtree(clone_dir)
                        yield SSEEvent(
                            event="error",
                            data={"message": f"Git clone failed: {tail}"}
                        ).format()
                        return

                    yield SSEEvent(
                        event="status",
                        data={"message": "Clone completed", "phase": "clone_done"}
                    ).format()
                    return
        finally:
            worker.join(timeout=5)

    def _run_git_clone(
        self,
        clone_url: str,
        clone_dir: Path,
        timeout: int,
        events: "queue.Queue[tuple[str, object]]",
    ) -> None:
        """Run `git clone` synchronously in a worker thread, pushing progress lines
        and the final result onto `events` for the async generator to consume."""
        try:
            process = subprocess.Popen(
                ["git", "clone", "--depth", "1", "--progress", clone_url, str(clone_dir)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError:
            events.put(("not_found", None))
            return
        except Exception as e:
            events.put(("exception", str(e)))
            return

        tail_lines: collections.deque = collections.deque(maxlen=20)
        buffer = b""
        try:
            while True:
                chunk = process.stderr.read(256)
                if not chunk:
                    break
                buffer += chunk
                # Git uses \r for progress updates and \n for final messages
                while b"\r" in buffer or b"\n" in buffer:
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
                        tail_lines.append(line.strip())
                        events.put(("line", line.strip()))
            if buffer.strip():
                tail_lines.append(buffer.decode("utf-8", errors="replace").strip())
        except Exception as e:
            events.put(("exception", str(e)))
            process.kill()
            process.wait()
            return

        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            events.put(("timeout", None))
            return

        events.put(("done", (returncode, "\n".join(tail_lines))))

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
