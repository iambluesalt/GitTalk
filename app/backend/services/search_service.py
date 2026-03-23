"""
Hybrid search service combining vector similarity and BM25 full-text search.
Uses Reciprocal Rank Fusion (RRF) to merge results from both search methods.
"""
from collections import defaultdict

from config import settings
from logger import logger
from models import SearchResult
from storage.vector_db import vector_db
from services.embedding_service import embedding_service


class SearchService:
    """Hybrid vector + FTS search with RRF re-ranking."""

    async def hybrid_search(
        self,
        project_id: str,
        query: str,
        n_results: int | None = None,
    ) -> list[SearchResult]:
        """
        Search for code chunks relevant to a query using hybrid search.

        Tries hybrid (vector + FTS) first, falls back to vector-only if FTS fails.
        """
        n_results = n_results or settings.MAX_SEARCH_RESULTS
        # Fetch more candidates than needed so RRF has material to work with
        limit = max(n_results * 4, settings.RETRIEVAL_CANDIDATES)

        if not vector_db.table_exists(project_id):
            logger.warning(f"No vector table for project {project_id}")
            return []

        query_vector = await embedding_service.embed_single(query)
        if not query_vector:
            logger.error("Failed to embed query")
            return []

        try:
            results = self._do_hybrid_search(
                project_id, query, query_vector, limit
            )
        except Exception as e:
            logger.warning(f"Hybrid search failed, falling back to vector-only: {e}")
            results = self._do_vector_only_search(project_id, query_vector, limit)

        return results[:n_results]

    def _do_hybrid_search(
        self,
        project_id: str,
        query: str,
        query_vector: list[float],
        limit: int,
    ) -> list[SearchResult]:
        """Run both vector and FTS searches, merge with RRF."""
        table = vector_db.get_or_create_table(project_id)

        # Vector search — returns rows with _distance (lower = more similar)
        vector_rows = table.search(query_vector).limit(limit).to_list()

        # FTS search — returns rows with _score (higher = more relevant)
        fts_rows = table.search(query, query_type="fts").limit(limit).to_list()

        if not fts_rows:
            logger.debug("FTS returned no results, using vector results only")
            return self._rows_to_results(vector_rows, score_field="_distance", invert=True)

        return self._rrf_merge(vector_rows, fts_rows)

    def _rrf_merge(
        self,
        vector_rows: list[dict],
        fts_rows: list[dict],
        k: int = 60,
    ) -> list[SearchResult]:
        """
        Reciprocal Rank Fusion: score[id] = sum(1 / (k + rank + 1)) across lists.

        Both lists are assumed to already be sorted by their respective relevance.
        """
        scores: dict[str, float] = defaultdict(float)
        row_map: dict[str, dict] = {}

        for rank, row in enumerate(vector_rows):
            chunk_id = row.get("id", "")
            scores[chunk_id] += 1.0 / (k + rank + 1)
            row_map[chunk_id] = row

        for rank, row in enumerate(fts_rows):
            chunk_id = row.get("id", "")
            scores[chunk_id] += 1.0 / (k + rank + 1)
            if chunk_id not in row_map:
                row_map[chunk_id] = row

        sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)

        results = []
        for chunk_id in sorted_ids:
            row = row_map[chunk_id]
            results.append(SearchResult(
                chunk_id=chunk_id,
                text=row.get("text", ""),
                file_path=row.get("file_path", ""),
                language=row.get("language", ""),
                function_name=row.get("function_name", ""),
                class_name=row.get("class_name", ""),
                line_start=row.get("line_start", 0),
                line_end=row.get("line_end", 0),
                chunk_type=row.get("chunk_type", ""),
                relevance_score=scores[chunk_id],
            ))

        return results

    def _do_vector_only_search(
        self,
        project_id: str,
        query_vector: list[float],
        limit: int,
    ) -> list[SearchResult]:
        """Fallback: vector-only search via direct LanceDB table API."""
        table = vector_db.get_or_create_table(project_id)
        rows = table.search(query_vector).limit(limit).to_list()
        return self._rows_to_results(rows, score_field="_distance", invert=True)

    def _rows_to_results(
        self,
        rows: list[dict],
        score_field: str = "_distance",
        invert: bool = False,
    ) -> list[SearchResult]:
        """Convert raw LanceDB rows to SearchResult list."""
        results = []
        for row in rows:
            raw_score = row.get(score_field, 0.0)
            relevance = 1.0 / (1.0 + raw_score) if invert else raw_score
            results.append(SearchResult(
                chunk_id=row.get("id", ""),
                text=row.get("text", ""),
                file_path=row.get("file_path", ""),
                language=row.get("language", ""),
                function_name=row.get("function_name", ""),
                class_name=row.get("class_name", ""),
                line_start=row.get("line_start", 0),
                line_end=row.get("line_end", 0),
                chunk_type=row.get("chunk_type", ""),
                relevance_score=relevance,
            ))
        return results


# Global instance
search_service = SearchService()
