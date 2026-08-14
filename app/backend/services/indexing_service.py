"""
Indexing service orchestrator.
Ties parsing, chunking, embedding, and vector storage together.
Supports incremental re-indexing via file hash tracking.
"""
import asyncio
import hashlib
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import AsyncGenerator

from config import settings
from logger import logger
from models import SSEEvent, ProjectStatus
from storage.metadata_db import db
from storage.vector_db import vector_db
from services.treesitter_service import treesitter_service
from services.chunker_service import code_chunker, CodeChunk
from services.embedding_service import embedding_service
from services.repomap_service import repomap_service
from utils.exclusions import load_gitignore_patterns, should_exclude


def _parse_and_chunk_worker(file_path_str: str, clone_dir_str: str) -> list[CodeChunk]:
    """Parse and chunk one file. Runs in a worker subprocess.

    Module-level (not a bound method) so it can be pickled and sent to a
    ProcessPoolExecutor. Some tree-sitter grammars can hard-crash (native
    access violation, not a catchable Python exception) on pathological
    input — running this in a subprocess means only that subprocess dies,
    not the whole server.
    """
    file_path = Path(file_path_str)
    clone_dir = Path(clone_dir_str)
    rel_path = file_path.relative_to(clone_dir).as_posix()
    ext = file_path.suffix.lower()

    try:
        source_bytes = file_path.read_bytes()
    except Exception as e:
        logger.debug(f"Cannot read {rel_path}: {e}")
        return []

    if not source_bytes.strip():
        return []

    # Try tree-sitter parsing
    tree = treesitter_service.parse_file(str(file_path), source_bytes, ext)

    # Chunk (tree may be None for non-parseable files — chunker handles this)
    return code_chunker.chunk_file(rel_path, ext, source_bytes, tree)


def _repo_map_entry_worker(file_path_str: str, clone_dir_str: str) -> tuple[str, list[tuple[str, str]]]:
    """Parse one file's repo-map entry in a worker subprocess (see
    _parse_and_chunk_worker docstring — same tree-sitter crash risk applies
    here, since repomap_service.process_file() also does real tree-sitter
    parsing, just for a signature summary instead of chunks).
    """
    from services.repomap_service import repomap_service
    return repomap_service.process_file(Path(file_path_str), Path(clone_dir_str))


# Lazily-created pool for isolated parsing. Kept at module level (like the
# other service singletons) rather than per-IndexingService-instance.
_parse_pool: ProcessPoolExecutor | None = None


def _get_parse_pool() -> ProcessPoolExecutor:
    global _parse_pool
    if _parse_pool is None:
        _parse_pool = ProcessPoolExecutor(max_workers=1)
    return _parse_pool


class IndexingService:
    """Orchestrates the full indexing pipeline."""

    async def index_project(
        self,
        project_id: str,
        clone_path: str,
        force_reindex: bool = False,
    ) -> AsyncGenerator[str, None]:
        """
        Index a project: parse, chunk, embed, store.
        Yields SSE event strings for progress reporting.
        """
        start_time = time.time()
        clone_dir = Path(clone_path)

        if not clone_dir.exists():
            yield SSEEvent(
                event="error",
                data={"message": f"Clone path not found: {clone_path}"},
            ).format()
            return

        # Update status to INDEXING
        db.update_project_status(project_id, ProjectStatus.INDEXING)
        yield SSEEvent(
            event="indexing_start",
            data={"message": "Starting indexing...", "project_id": project_id},
        ).format()

        try:
            # 1. Collect eligible files
            yield SSEEvent(
                event="status",
                data={"message": "Scanning files...", "phase": "scan"},
            ).format()

            gitignore_patterns = load_gitignore_patterns(clone_dir)
            all_files = await asyncio.to_thread(
                self._collect_files, clone_dir, gitignore_patterns
            )
            total_files = len(all_files)

            yield SSEEvent(
                event="status",
                data={
                    "message": f"Found {total_files} files to process",
                    "phase": "scan",
                    "total_files": total_files,
                },
            ).format()

            # 2. Compute file hashes and determine what needs indexing
            existing_hashes = {}
            if not force_reindex:
                existing_hashes = db.get_file_hashes(project_id)

            files_to_index: list[tuple[Path, str]] = []  # (path, hash)
            current_file_set: set[str] = set()

            for file_path in all_files:
                rel_path = file_path.relative_to(clone_dir).as_posix()
                current_file_set.add(rel_path)
                file_hash = await asyncio.to_thread(self._compute_file_hash, file_path)

                if not force_reindex and rel_path in existing_hashes:
                    old_hash, _ = existing_hashes[rel_path]
                    if old_hash == file_hash:
                        continue  # File unchanged, skip

                files_to_index.append((file_path, file_hash))

            # 3. Delete stale files (removed from disk)
            stale_files = db.get_stale_files(project_id, current_file_set)
            if stale_files:
                yield SSEEvent(
                    event="status",
                    data={
                        "message": f"Removing {len(stale_files)} stale files...",
                        "phase": "cleanup",
                    },
                ).format()
                for stale_path in stale_files:
                    old_data = existing_hashes.get(stale_path)
                    if old_data:
                        _, old_chunk_ids = old_data
                        if old_chunk_ids:
                            vector_db.delete_by_ids(project_id, old_chunk_ids)
                    db.delete_file_from_index(project_id, stale_path)

            if not files_to_index:
                yield SSEEvent(
                    event="status",
                    data={
                        "message": "All files up to date, no changes detected",
                        "phase": "skip",
                    },
                ).format()

                # Still regenerate repo map and update status
                await self._finalize(project_id, clone_dir, start_time, 0, 0)
                yield self._complete_event(project_id, 0, 0, start_time)
                return

            # 4. Check embedding service availability
            embed_available = await embedding_service.is_available()
            if not embed_available:
                yield SSEEvent(
                    event="error",
                    data={
                        "message": (
                            f"Embedding model '{settings.OLLAMA_EMBED_MODEL}' "
                            f"not available at {settings.OLLAMA_BASE_URL}. "
                            "Please ensure Ollama is running and the model is pulled."
                        )
                    },
                ).format()
                db.update_project_error(project_id, "Embedding model not available")
                return

            # 5. Process files: parse, chunk, embed, store
            total_chunks_created = 0
            files_processed = 0

            # Delete old chunks for files that changed
            for file_path, file_hash in files_to_index:
                rel_path = file_path.relative_to(clone_dir).as_posix()
                if rel_path in existing_hashes:
                    _, old_chunk_ids = existing_hashes[rel_path]
                    if old_chunk_ids:
                        vector_db.delete_by_ids(project_id, old_chunk_ids)

            # Process in batches
            chunk_buffer: list[CodeChunk] = []
            file_chunk_map: dict[str, list[str]] = {}  # rel_path → [chunk_ids]
            file_hash_map: dict[str, str] = {}  # rel_path → hash
            embed_batch_num = 0

            yield SSEEvent(
                event="status",
                data={
                    "message": f"Parsing & chunking {len(files_to_index)} files...",
                    "phase": "parse",
                },
            ).format()

            for file_path, file_hash in files_to_index:
                rel_path = file_path.relative_to(clone_dir).as_posix()
                files_processed += 1

                # Progress event every 3 files or for last file
                if files_processed % 3 == 0 or files_processed == len(files_to_index):
                    yield SSEEvent(
                        event="indexing_progress",
                        data={
                            "files_processed": files_processed,
                            "total_files": len(files_to_index),
                            "current_file": rel_path,
                            "percent": round(
                                files_processed / len(files_to_index) * 100, 1
                            ),
                        },
                    ).format()

                # Parse and chunk the file, isolated in a subprocess (see
                # _parse_and_chunk_worker docstring)
                chunks = await self._parse_and_chunk_isolated(file_path, clone_dir)
                if not chunks:
                    file_hash_map[rel_path] = file_hash
                    file_chunk_map[rel_path] = []
                    continue

                file_chunk_map[rel_path] = [c.id for c in chunks]
                file_hash_map[rel_path] = file_hash
                chunk_buffer.extend(chunks)

                # Embed and store when buffer is large enough
                if len(chunk_buffer) >= settings.EMBEDDING_BATCH_SIZE:
                    embed_batch_num += 1
                    yield SSEEvent(
                        event="status",
                        data={
                            "message": f"Embedding batch {embed_batch_num} ({len(chunk_buffer)} chunks)...",
                            "phase": "embed",
                        },
                    ).format()
                    stored = await self._embed_and_store(
                        project_id, chunk_buffer
                    )
                    total_chunks_created += stored
                    chunk_buffer = []

            # Flush remaining chunks
            if chunk_buffer:
                embed_batch_num += 1
                yield SSEEvent(
                    event="status",
                    data={
                        "message": f"Embedding final batch ({len(chunk_buffer)} chunks)...",
                        "phase": "embed",
                    },
                ).format()
                stored = await self._embed_and_store(project_id, chunk_buffer)
                total_chunks_created += stored

            # 6. Update file hashes and chunk IDs in SQLite
            for rel_path, chunk_ids in file_chunk_map.items():
                file_hash = file_hash_map[rel_path]
                db.update_file_hash(project_id, rel_path, file_hash, chunk_ids)

            # 7. Finalize: FTS index, repo map, status update
            yield SSEEvent(
                event="status",
                data={"message": "Finalizing index...", "phase": "finalize"},
            ).format()

            await self._finalize(
                project_id, clone_dir, start_time,
                files_processed, total_chunks_created,
            )

            yield self._complete_event(
                project_id, files_processed, total_chunks_created, start_time
            )

        except Exception as e:
            logger.error(f"Indexing failed for project {project_id}: {e}")
            db.update_project_error(project_id, str(e))
            yield SSEEvent(
                event="error",
                data={"message": f"Indexing failed: {str(e)}"},
            ).format()
        finally:
            # If the generator is torn down some other way — client
            # disconnect, server shutdown — neither the success path nor the
            # `except` above runs, and the project would stay stuck on
            # INDEXING forever (blocking any future re-index attempt). Only
            # touch it if it's still unresolved; success already moved it to
            # INDEXED and the except above already moved it to ERROR.
            project = db.get_project(project_id)
            if project and project.status == ProjectStatus.INDEXING:
                logger.warning(f"Indexing for {project_id} ended without resolving status — marking as error")
                db.update_project_error(project_id, "Indexing was interrupted (connection closed or server stopped)")

    def _collect_files(
        self, clone_dir: Path, gitignore_patterns: list[str]
    ) -> list[Path]:
        """Walk directory tree and collect eligible files."""
        files = []
        for file_path in clone_dir.rglob("*"):
            if not file_path.is_file():
                continue
            if should_exclude(file_path, clone_dir, gitignore_patterns):
                continue
            # Skip very large files (> 1MB)
            try:
                if file_path.stat().st_size > 1_000_000:
                    continue
            except OSError:
                continue
            files.append(file_path)
        return files

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA-256 hash of a file."""
        try:
            hasher = hashlib.sha256()
            with open(file_path, "rb") as f:
                for block in iter(lambda: f.read(8192), b""):
                    hasher.update(block)
            return hasher.hexdigest()
        except Exception:
            return ""

    async def _parse_and_chunk_isolated(
        self, file_path: Path, clone_dir: Path
    ) -> list[CodeChunk]:
        """Parse and chunk one file in the isolated worker subprocess.

        If the worker process crashes (e.g. a native tree-sitter grammar
        bug), the pool is marked broken — log it, drop that file, and
        rebuild the pool so subsequent files aren't affected.
        """
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(
                _get_parse_pool(),
                _parse_and_chunk_worker,
                str(file_path),
                str(clone_dir),
            )
        except BrokenProcessPool:
            rel_path = file_path.relative_to(clone_dir).as_posix()
            logger.error(
                f"Parser crashed on '{rel_path}' — skipping this file. "
                "Likely a native tree-sitter grammar bug on this file's content."
            )
            global _parse_pool
            _parse_pool = None  # rebuilt lazily on next call
            return []

    async def _embed_and_store(
        self, project_id: str, chunks: list[CodeChunk]
    ) -> int:
        """Embed a batch of chunks and store in LanceDB."""
        if not chunks:
            return 0

        # Embed using contextualized text (with headers for better retrieval)
        embed_texts = [c.embedding_text for c in chunks]

        try:
            embeddings = await embedding_service.embed_texts(embed_texts)
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return 0

        if len(embeddings) != len(chunks):
            logger.error(
                f"Embedding count mismatch: {len(embeddings)} vs {len(chunks)} chunks"
            )
            return 0

        ids = [c.id for c in chunks]
        # Store raw text (no headers) for BM25 search and LLM display
        documents = [c.text for c in chunks]
        metadatas = [
            {
                "file_path": c.file_path,
                "language": c.language,
                "function_name": c.function_name or "",
                "class_name": c.class_name or "",
                "line_start": c.line_start,
                "line_end": c.line_end,
                "chunk_type": c.chunk_type,
            }
            for c in chunks
        ]

        success = vector_db.add_embeddings(
            project_id, embeddings, documents, metadatas, ids
        )
        return len(chunks) if success else 0

    async def _finalize(
        self,
        project_id: str,
        clone_dir: Path,
        start_time: float,
        files_processed: int,
        chunks_created: int,
    ):
        """Create FTS index, generate repo map, update project status."""
        # Create FTS index
        vector_db.create_fts_index(project_id)

        # Generate repo map
        try:
            repo_map = await self._generate_repo_map_isolated(project_id, clone_dir)
            db.update_project_repo_map(project_id, repo_map)
        except Exception as e:
            logger.warning(f"Failed to generate repo map: {e}")

        # Update project status
        db.update_project_status(project_id, ProjectStatus.INDEXED)
        db.update_last_indexed(project_id)

        duration = round(time.time() - start_time, 2)
        logger.info(
            f"Indexing complete for {project_id}: "
            f"{files_processed} files, {chunks_created} chunks, {duration}s"
        )

    async def _generate_repo_map_isolated(self, project_id: str, clone_dir: Path) -> str:
        """Same output as repomap_service.generate_repo_map(), but each file's
        tree-sitter parse runs in the isolated worker subprocess. If a file
        crashes the worker, only that file's entry is dropped — the pool is
        rebuilt and the rest of the repo map still completes.
        """
        loop = asyncio.get_running_loop()
        entries: list[tuple[str, list[tuple[str, str]]]] = []

        for file_path in repomap_service.iter_files(clone_dir):
            try:
                entry = await loop.run_in_executor(
                    _get_parse_pool(), _repo_map_entry_worker,
                    str(file_path), str(clone_dir),
                )
            except BrokenProcessPool:
                rel_path = file_path.relative_to(clone_dir).as_posix()
                logger.error(
                    f"Repo map parser crashed on '{rel_path}' — skipping this file's entry."
                )
                global _parse_pool
                _parse_pool = None
                entry = (rel_path, [])
            entries.append(entry)

        return repomap_service.format_entries(project_id, entries)

    def _complete_event(
        self,
        project_id: str,
        files_processed: int,
        chunks_created: int,
        start_time: float,
    ) -> str:
        """Generate the completion SSE event."""
        duration = round(time.time() - start_time, 2)
        return SSEEvent(
            event="indexing_complete",
            data={
                "message": "Indexing complete",
                "project_id": project_id,
                "files_indexed": files_processed,
                "chunks_created": chunks_created,
                "duration_seconds": duration,
            },
        ).format()


# Global instance
indexing_service = IndexingService()
