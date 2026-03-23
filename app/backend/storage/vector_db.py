"""
LanceDB vector database wrapper for code embeddings.
Handles embedding storage, retrieval, and similarity search.
"""
import lancedb
import pyarrow as pa
from typing import List, Dict, Any, Optional
from pathlib import Path

from config import settings
from logger import logger

# LanceDB table schema for code chunks
TABLE_SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("text", pa.large_string()),
    pa.field("vector", pa.list_(pa.float32(), settings.EMBEDDING_DIMENSIONS)),
    pa.field("file_path", pa.string()),
    pa.field("language", pa.string()),
    pa.field("function_name", pa.string()),
    pa.field("class_name", pa.string()),
    pa.field("line_start", pa.int32()),
    pa.field("line_end", pa.int32()),
    pa.field("chunk_type", pa.string()),
])


def _table_name(project_id: str) -> str:
    """Generate a sanitized table name for a project."""
    return f"project_{project_id.replace('-', '_')}"


class VectorDB:
    """LanceDB wrapper for vector storage and retrieval."""

    def __init__(self, persist_directory: Optional[Path] = None):
        """Initialize LanceDB connection."""
        self.persist_directory = persist_directory or settings.VECTOR_DB_PATH
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        try:
            self.db = lancedb.connect(str(self.persist_directory))
            logger.info(f"LanceDB initialized at: {self.persist_directory}")
        except Exception as e:
            logger.error(f"Failed to initialize LanceDB: {e}")
            raise

    def _list_table_names(self) -> list[str]:
        """Get list of table name strings from LanceDB."""
        result = self.db.list_tables()
        # LanceDB returns a ListTablesResponse object, not a plain list
        if hasattr(result, "tables"):
            return result.tables
        return list(result)

    def get_or_create_table(self, project_id: str) -> lancedb.table.Table:
        """Get or create a table for a project."""
        name = _table_name(project_id)
        try:
            return self.db.open_table(name)
        except Exception:
            pass
        try:
            table = self.db.create_table(name, schema=TABLE_SCHEMA)
            logger.debug(f"Created table: {name}")
            return table
        except Exception:
            # Table was created between our open and create attempts
            return self.db.open_table(name)

    def add_embeddings(
        self,
        project_id: str,
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: List[str],
    ) -> bool:
        """Add embeddings to the table."""
        try:
            table = self.get_or_create_table(project_id)
            records = []
            for i in range(len(ids)):
                meta = metadatas[i] if i < len(metadatas) else {}
                records.append({
                    "id": ids[i],
                    "text": documents[i],
                    "vector": embeddings[i],
                    "file_path": meta.get("file_path", ""),
                    "language": meta.get("language", ""),
                    "function_name": meta.get("function_name", ""),
                    "class_name": meta.get("class_name", ""),
                    "line_start": meta.get("line_start", 0),
                    "line_end": meta.get("line_end", 0),
                    "chunk_type": meta.get("chunk_type", ""),
                })
            table.add(records)
            logger.info(f"Added {len(records)} embeddings to project {project_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to add embeddings: {e}")
            return False

    def search(
        self,
        project_id: str,
        query_embedding: List[float],
        n_results: int = 5,
        where: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Search for similar code chunks."""
        try:
            table = self.get_or_create_table(project_id)
            if table.count_rows() == 0:
                return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

            query = table.search(query_embedding).limit(n_results)
            if where:
                query = query.where(where)
            results = query.to_list()

            documents = []
            metadatas = []
            distances = []
            for row in results:
                documents.append(row.get("text", ""))
                distances.append(row.get("_distance", 0.0))
                metadatas.append({
                    "id": row.get("id", ""),
                    "file_path": row.get("file_path", ""),
                    "language": row.get("language", ""),
                    "function_name": row.get("function_name", ""),
                    "class_name": row.get("class_name", ""),
                    "line_start": row.get("line_start", 0),
                    "line_end": row.get("line_end", 0),
                    "chunk_type": row.get("chunk_type", ""),
                })

            return {
                "documents": [documents],
                "metadatas": [metadatas],
                "distances": [distances],
            }
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    def delete_by_ids(self, project_id: str, ids: List[str]) -> bool:
        """Delete specific chunks by ID."""
        if not ids:
            return True
        try:
            table = self.get_or_create_table(project_id)
            id_list = ", ".join(f"'{id_}'" for id_ in ids)
            table.delete(f"id IN ({id_list})")
            logger.debug(f"Deleted {len(ids)} chunks from project {project_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete chunks: {e}")
            return False

    def delete_table(self, project_id: str) -> bool:
        """Delete a project's table."""
        try:
            name = _table_name(project_id)
            if name in self._list_table_names():
                self.db.drop_table(name)
                logger.info(f"Deleted table: {name}")
            return True
        except Exception as e:
            logger.warning(f"Failed to delete table: {e}")
            return False

    def get_table_count(self, project_id: str) -> int:
        """Get number of embeddings in a table."""
        try:
            name = _table_name(project_id)
            if name not in self._list_table_names():
                return 0
            table = self.db.open_table(name)
            return table.count_rows()
        except Exception as e:
            logger.error(f"Failed to get table count: {e}")
            return 0

    def table_exists(self, project_id: str) -> bool:
        """Check if a table exists for a project."""
        try:
            name = _table_name(project_id)
            return name in self._list_table_names()
        except Exception as e:
            logger.error(f"Failed to check table existence: {e}")
            return False

    def create_fts_index(self, project_id: str) -> bool:
        """Create a full-text search (BM25) index on the text column."""
        try:
            table = self.get_or_create_table(project_id)
            if table.count_rows() == 0:
                return True
            table.create_fts_index("text", replace=True)
            logger.info(f"Created FTS index for project {project_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to create FTS index: {e}")
            return False

    def is_healthy(self) -> bool:
        """Check if LanceDB is accessible."""
        try:
            self._list_table_names()
            return True
        except Exception:
            return False


# Global vector database instance
vector_db = VectorDB()
