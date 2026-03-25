"""
SQLite metadata database for project and code index management.
Stores project metadata, file relationships, and code structure.
"""
import sqlite3
import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

from config import settings
from logger import logger
from models import ProjectMetadata, ProjectStatus, RepositoryAnalysis, ConversationTurn


class MetadataDB:
    """SQLite database wrapper for metadata storage."""

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize database connection."""
        self.db_path = db_path or settings.METADATA_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()

    def _init_database(self):
        """Initialize database schema."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Projects table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    github_url TEXT NOT NULL UNIQUE,
                    clone_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    cloned_at TIMESTAMP NOT NULL,
                    last_indexed TIMESTAMP,
                    last_used TIMESTAMP,
                    analysis_json TEXT,
                    repo_map TEXT,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Code index table - stores file-level metadata
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS code_index (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_type TEXT,
                    language TEXT,
                    lines_count INTEGER,
                    functions TEXT,
                    classes TEXT,
                    imports TEXT,
                    file_hash TEXT,
                    chunk_ids TEXT,
                    last_indexed TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
                    UNIQUE(project_id, file_path)
                )
            """)

            # Conversations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
                )
            """)

            # Messages table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    sources_json TEXT,
                    response_time_ms REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                )
            """)

            # Create indexes for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_projects_status
                ON projects(status)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_code_index_project
                ON code_index(project_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_code_index_file_path
                ON code_index(file_path)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversations_project
                ON conversations(project_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_conversation
                ON messages(conversation_id)
            """)

            # ALTER TABLE for existing databases — add new columns safely
            self._safe_add_column(cursor, "projects", "repo_map", "TEXT")
            self._safe_add_column(cursor, "projects", "error_message", "TEXT")
            self._safe_add_column(cursor, "code_index", "file_hash", "TEXT")
            self._safe_add_column(cursor, "code_index", "chunk_ids", "TEXT")
            self._safe_add_column(cursor, "conversations", "summary", "TEXT")
            self._safe_add_column(cursor, "conversations", "summarized_up_to", "INTEGER DEFAULT 0")

            conn.commit()
            logger.info("Database schema initialized successfully")

    def _safe_add_column(self, cursor, table: str, column: str, col_type: str):
        """Add a column to a table if it doesn't already exist."""
        try:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
            logger.info(f"Added column {column} to {table}")
        except sqlite3.OperationalError:
            # Column already exists
            pass

    # ========================================================================
    # Project Operations
    # ========================================================================

    def create_project(self, project: ProjectMetadata) -> bool:
        """Create a new project entry."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                analysis_json = (
                    project.analysis.model_dump_json() if project.analysis else None
                )
                cursor.execute("""
                    INSERT INTO projects
                    (id, name, github_url, clone_path, status, cloned_at,
                     last_indexed, last_used, analysis_json, repo_map, error_message)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    project.id,
                    project.name,
                    project.github_url,
                    project.clone_path,
                    project.status.value,
                    project.cloned_at,
                    project.last_indexed,
                    project.last_used,
                    analysis_json,
                    project.repo_map,
                    project.error_message,
                ))
                logger.info(f"Created project: {project.name} ({project.id})")
                return True
        except sqlite3.IntegrityError as e:
            logger.warning(f"Project already exists: {project.github_url}")
            return False
        except Exception as e:
            logger.error(f"Failed to create project: {e}")
            raise

    def get_project(self, project_id: str) -> Optional[ProjectMetadata]:
        """Retrieve a project by ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
            row = cursor.fetchone()

            if row:
                return self._row_to_project(row)
            return None

    def get_project_by_url(self, github_url: str) -> Optional[ProjectMetadata]:
        """Check if a project with this URL already exists."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM projects WHERE github_url = ?", (github_url,))
            row = cursor.fetchone()

            if row:
                return self._row_to_project(row)
            return None

    def update_project_status(self, project_id: str, status: ProjectStatus) -> bool:
        """Update project status."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE projects
                SET status = ?, last_used = ?
                WHERE id = ?
            """, (status.value, datetime.now(), project_id))
            return cursor.rowcount > 0

    def update_project_analysis(self, project_id: str, analysis: RepositoryAnalysis) -> bool:
        """Update project analysis data."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE projects
                SET analysis_json = ?
                WHERE id = ?
            """, (analysis.model_dump_json(), project_id))
            return cursor.rowcount > 0

    def update_project_error(self, project_id: str, error_message: str) -> bool:
        """Update project error message and set status to ERROR."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE projects
                SET error_message = ?, status = ?
                WHERE id = ?
            """, (error_message, ProjectStatus.ERROR.value, project_id))
            return cursor.rowcount > 0

    def update_project_repo_map(self, project_id: str, repo_map: str) -> bool:
        """Update project repository map."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE projects
                SET repo_map = ?
                WHERE id = ?
            """, (repo_map, project_id))
            return cursor.rowcount > 0

    def update_last_indexed(self, project_id: str) -> bool:
        """Update last indexed timestamp."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE projects
                SET last_indexed = ?
                WHERE id = ?
            """, (datetime.now(), project_id))
            return cursor.rowcount > 0

    def list_projects(self, limit: int = 50) -> List[ProjectMetadata]:
        """List all projects."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM projects
                ORDER BY last_used DESC, cloned_at DESC
                LIMIT ?
            """, (limit,))
            return [self._row_to_project(row) for row in cursor.fetchall()]

    def get_recent_projects(self, limit: int = 5) -> List[ProjectMetadata]:
        """Get recently used projects."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM projects
                WHERE last_used IS NOT NULL
                ORDER BY last_used DESC
                LIMIT ?
            """, (limit,))
            return [self._row_to_project(row) for row in cursor.fetchall()]

    def delete_project(self, project_id: str) -> bool:
        """Delete a project and all related data (code_index, conversations, messages)."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # Get conversation IDs for this project to delete their messages
            cursor.execute(
                "SELECT id FROM conversations WHERE project_id = ?", (project_id,)
            )
            conv_ids = [row["id"] for row in cursor.fetchall()]
            for conv_id in conv_ids:
                cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
            cursor.execute("DELETE FROM conversations WHERE project_id = ?", (project_id,))
            cursor.execute("DELETE FROM code_index WHERE project_id = ?", (project_id,))
            cursor.execute("DELETE FROM projects WHERE id = ?", (project_id,))
            deleted = cursor.rowcount > 0
            if deleted:
                logger.info(f"Deleted project and related data: {project_id}")
            return deleted

    def delete_all_projects(self) -> int:
        """Delete all projects and related data. Returns count of deleted projects."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM projects")
            count = cursor.fetchone()[0]
            cursor.execute("DELETE FROM messages")
            cursor.execute("DELETE FROM conversations")
            cursor.execute("DELETE FROM code_index")
            cursor.execute("DELETE FROM projects")
            logger.info(f"Deleted all projects ({count} total)")
            return count

    # ========================================================================
    # Code Index Operations
    # ========================================================================

    def add_file_to_index(
        self,
        project_id: str,
        file_path: str,
        file_type: str,
        language: str,
        lines_count: int,
        functions: List[str],
        classes: List[str],
        imports: List[str]
    ) -> bool:
        """Add or update a file in the code index."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO code_index
                    (project_id, file_path, file_type, language, lines_count,
                     functions, classes, imports, last_indexed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    project_id,
                    file_path,
                    file_type,
                    language,
                    lines_count,
                    json.dumps(functions),
                    json.dumps(classes),
                    json.dumps(imports),
                    datetime.now()
                ))
                return True
        except Exception as e:
            logger.error(f"Failed to add file to index: {e}")
            return False

    def get_index_stats(self, project_id: str) -> Dict[str, Any]:
        """Get aggregate index statistics for a project."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    COUNT(*) as files_indexed,
                    COALESCE(SUM(lines_count), 0) as total_lines,
                    COALESCE(SUM(json_array_length(functions)), 0) as total_functions,
                    COALESCE(SUM(json_array_length(classes)), 0) as total_classes,
                    COALESCE(SUM(json_array_length(imports)), 0) as total_imports,
                    COUNT(DISTINCT language) as languages_count
                FROM code_index WHERE project_id = ?
            """, (project_id,))
            row = cursor.fetchone()
            if not row or row["files_indexed"] == 0:
                return {
                    "files_indexed": 0,
                    "total_lines": 0,
                    "total_functions": 0,
                    "total_classes": 0,
                    "total_imports": 0,
                    "languages_count": 0,
                    "chunks_created": 0,
                }

            # Count chunks from chunk_ids JSON arrays
            cursor.execute("""
                SELECT chunk_ids FROM code_index
                WHERE project_id = ? AND chunk_ids IS NOT NULL AND chunk_ids != ''
            """, (project_id,))
            total_chunks = 0
            for r in cursor.fetchall():
                try:
                    chunks = json.loads(r["chunk_ids"])
                    total_chunks += len(chunks) if isinstance(chunks, list) else 0
                except (json.JSONDecodeError, TypeError):
                    pass

            return {
                "files_indexed": row["files_indexed"],
                "total_lines": row["total_lines"],
                "total_functions": row["total_functions"],
                "total_classes": row["total_classes"],
                "total_imports": row["total_imports"],
                "languages_count": row["languages_count"],
                "chunks_created": total_chunks,
            }

    def get_bulk_index_counts(self) -> Dict[str, Dict[str, int]]:
        """Get indexed file count and chunk count for all projects (single query)."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT project_id,
                       COUNT(*) as files_indexed,
                       COALESCE(SUM(json_array_length(CASE WHEN chunk_ids IS NOT NULL AND chunk_ids != '' THEN chunk_ids ELSE '[]' END)), 0) as chunks_created
                FROM code_index
                GROUP BY project_id
            """)
            result = {}
            for row in cursor.fetchall():
                result[row["project_id"]] = {
                    "files_indexed": row["files_indexed"],
                    "chunks_created": row["chunks_created"],
                }
            return result

    def get_project_files(self, project_id: str) -> List[Dict[str, Any]]:
        """Get all indexed files for a project."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM code_index WHERE project_id = ?
            """, (project_id,))

            files = []
            for row in cursor.fetchall():
                files.append({
                    "file_path": row["file_path"],
                    "file_type": row["file_type"],
                    "language": row["language"],
                    "lines_count": row["lines_count"],
                    "functions": json.loads(row["functions"]) if row["functions"] else [],
                    "classes": json.loads(row["classes"]) if row["classes"] else [],
                    "imports": json.loads(row["imports"]) if row["imports"] else [],
                    "last_indexed": row["last_indexed"]
                })
            return files

    def update_file_hash(
        self, project_id: str, file_path: str, file_hash: str, chunk_ids: List[str]
    ) -> bool:
        """Update or insert file hash and chunk IDs for incremental indexing."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO code_index (project_id, file_path, file_hash, chunk_ids, last_indexed)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(project_id, file_path) DO UPDATE SET
                        file_hash = excluded.file_hash,
                        chunk_ids = excluded.chunk_ids,
                        last_indexed = excluded.last_indexed
                """, (
                    project_id,
                    file_path,
                    file_hash,
                    json.dumps(chunk_ids),
                    datetime.now(),
                ))
                return True
        except Exception as e:
            logger.error(f"Failed to update file hash: {e}")
            return False

    def get_file_hashes(self, project_id: str) -> Dict[str, tuple]:
        """
        Get file hashes and chunk IDs for a project.

        Returns:
            Dict of file_path → (file_hash, list_of_chunk_ids)
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT file_path, file_hash, chunk_ids
                FROM code_index WHERE project_id = ?
            """, (project_id,))

            result = {}
            for row in cursor.fetchall():
                file_hash = row["file_hash"] or ""
                chunk_ids_raw = row["chunk_ids"]
                chunk_ids = json.loads(chunk_ids_raw) if chunk_ids_raw else []
                result[row["file_path"]] = (file_hash, chunk_ids)
            return result

    def delete_file_from_index(self, project_id: str, file_path: str) -> bool:
        """Delete a file entry from the code index."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    DELETE FROM code_index
                    WHERE project_id = ? AND file_path = ?
                """, (project_id, file_path))
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to delete file from index: {e}")
            return False

    def get_stale_files(self, project_id: str, current_files: set) -> List[str]:
        """
        Find files in the DB index that are no longer on disk.

        Args:
            project_id: Project identifier
            current_files: Set of relative file paths currently on disk

        Returns:
            List of stale file paths
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT file_path FROM code_index WHERE project_id = ?
            """, (project_id,))

            stale = []
            for row in cursor.fetchall():
                if row["file_path"] not in current_files:
                    stale.append(row["file_path"])
            return stale

    # ========================================================================
    # Conversation Operations
    # ========================================================================

    def create_conversation(self, project_id: str, title: str = "New Chat") -> str:
        """Create a new conversation and return its ID."""
        conv_id = uuid.uuid4().hex[:12]
        now = datetime.now()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO conversations (id, project_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
            """, (conv_id, project_id, title, now, now))
        logger.debug(f"Created conversation {conv_id} for project {project_id}")
        return conv_id

    def get_latest_conversation(self, project_id: str) -> Optional[str]:
        """Get the most recent conversation ID for a project."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id FROM conversations
                WHERE project_id = ?
                ORDER BY updated_at DESC
                LIMIT 1
            """, (project_id,))
            row = cursor.fetchone()
            return row["id"] if row else None

    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get a single conversation by ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, project_id, title, created_at, updated_at
                FROM conversations WHERE id = ?
            """, (conversation_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def list_conversations(self, project_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """List conversations for a project, newest first."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, project_id, title, created_at, updated_at
                FROM conversations
                WHERE project_id = ?
                ORDER BY updated_at DESC
                LIMIT ?
            """, (project_id, limit))
            return [dict(row) for row in cursor.fetchall()]

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        sources_json: Optional[str] = None,
        response_time_ms: Optional[float] = None,
    ) -> str:
        """Add a message to a conversation and return its ID."""
        msg_id = uuid.uuid4().hex[:12]
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO messages
                (id, conversation_id, role, content, sources_json, response_time_ms, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (msg_id, conversation_id, role, content, sources_json, response_time_ms, datetime.now()))
            # Touch the conversation's updated_at
            cursor.execute("""
                UPDATE conversations SET updated_at = ? WHERE id = ?
            """, (datetime.now(), conversation_id))
        return msg_id

    def get_conversation_messages(
        self, conversation_id: str, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get messages for a conversation, ordered chronologically.

        Returns list of dicts with: id, role, content, sources_json, response_time_ms, created_at
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, role, content, sources_json, response_time_ms, created_at
                FROM messages
                WHERE conversation_id = ?
                ORDER BY created_at ASC
                LIMIT ?
            """, (conversation_id, limit))
            return [dict(row) for row in cursor.fetchall()]

    def get_conversation_history(
        self, conversation_id: str, limit: int = 20
    ) -> List[ConversationTurn]:
        """Get conversation history as ConversationTurn objects for the RAG pipeline."""
        rows = self.get_conversation_messages(conversation_id, limit)
        return [
            ConversationTurn(role=row["role"], content=row["content"])
            for row in rows
        ]

    def get_conversation_summary(self, conversation_id: str) -> tuple[str | None, int]:
        """
        Get the rolling summary and the message count it covers.

        Returns:
            (summary_text or None, summarized_up_to count)
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT summary, summarized_up_to FROM conversations WHERE id = ?",
                (conversation_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None, 0
            return row["summary"], row["summarized_up_to"] or 0

    def update_conversation_summary(
        self, conversation_id: str, summary: str, summarized_up_to: int
    ) -> bool:
        """Store a rolling conversation summary."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """UPDATE conversations
                   SET summary = ?, summarized_up_to = ?, updated_at = ?
                   WHERE id = ?""",
                (summary, summarized_up_to, datetime.now(), conversation_id),
            )
            return cursor.rowcount > 0

    def get_all_conversation_messages(
        self, conversation_id: str
    ) -> List[Dict[str, Any]]:
        """Get ALL messages for a conversation (no limit), for summarization."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT id, role, content, sources_json, response_time_ms, created_at
                   FROM messages
                   WHERE conversation_id = ?
                   ORDER BY created_at ASC""",
                (conversation_id,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and its messages."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
            cursor.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
            deleted = cursor.rowcount > 0
            if deleted:
                logger.debug(f"Deleted conversation {conversation_id}")
            return deleted

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _row_to_project(self, row: sqlite3.Row) -> ProjectMetadata:
        """Convert database row to ProjectMetadata model."""
        analysis = None
        if row["analysis_json"]:
            try:
                analysis = RepositoryAnalysis.model_validate_json(row["analysis_json"])
            except Exception as e:
                logger.warning(f"Bad analysis data for project {row['id']}: {e}")

        return ProjectMetadata(
            id=row["id"],
            name=row["name"],
            github_url=row["github_url"],
            clone_path=row["clone_path"],
            status=ProjectStatus(row["status"]),
            cloned_at=datetime.fromisoformat(row["cloned_at"]),
            last_indexed=(
                datetime.fromisoformat(row["last_indexed"])
                if row["last_indexed"] else None
            ),
            last_used=(
                datetime.fromisoformat(row["last_used"])
                if row["last_used"] else None
            ),
            analysis=analysis,
            repo_map=row["repo_map"] if "repo_map" in row.keys() else None,
            error_message=row["error_message"] if "error_message" in row.keys() else None,
        )


# Global database instance
db = MetadataDB()
