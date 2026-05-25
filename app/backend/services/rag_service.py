"""
RAG context assembly service.
Builds structured LLM prompts from search results, repo map, and conversation history.
Routes queries through intent classification to skip retrieval for non-code messages.
"""
from config import settings
from logger import logger
from models import RAGContext, CodeReference, ConversationTurn
from storage.metadata_db import db
from services.search_service import search_service
from services.llm_service import get_prompt_template
from services.router_service import router_service


def _estimate_tokens(text: str) -> int:
    """Estimate token count (~4 chars per token)."""
    return len(text) // 4


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to fit within a token budget."""
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... (truncated)"


class RAGService:
    """Assembles search results and context into LLM-ready messages."""

    async def build_context(
        self,
        project_id: str,
        query: str,
        conversation_id: str | None = None,
        conversation_history: list[ConversationTurn] | None = None,
        template: str = "code_qa",
    ) -> RAGContext:
        """
        Build a complete LLM context for a user query.

        1. Classifies intent (code / general / follow_up)
        2. Skips retrieval for non-code queries
        3. Dynamically allocates token budget across repo map, code chunks, history
        """
        if conversation_id and not conversation_history:
            conversation_history = db.get_conversation_history(conversation_id)
        conversation_history = conversation_history or []

        # Load project metadata
        project = db.get_project(project_id)
        if not project:
            logger.error(f"Project not found: {project_id}")
            return RAGContext(
                messages=[{"role": "user", "content": query}],
                sources=[],
                token_count=_estimate_tokens(query),
                search_results_count=0,
            )

        # --- Classify intent ---
        recent_for_router = [
            {"role": t.role, "content": t.content}
            for t in conversation_history[-4:]
        ]
        intent = await router_service.classify(query, recent_for_router)
        logger.info(f"Query intent: {intent} | query: {query[:80]}")

        # --- Conditional retrieval ---
        search_results = []
        if intent == "code":
            search_results = await search_service.hybrid_search(project_id, query)
        elif intent == "follow_up":
            # Enrich follow-up with context from last assistant turn for better retrieval
            enriched = self._enrich_follow_up_query(query, conversation_history)
            search_results = await search_service.hybrid_search(project_id, enriched)

        # --- Deduplicate overlapping chunks from the same file ---
        if search_results:
            search_results = self._dedup_overlapping(search_results)

        # --- Dynamic token budget based on intent ---
        total_budget = settings.MAX_CONTEXT_TOKENS
        system_fixed = 500
        query_fixed = 200
        remaining = total_budget - system_fixed - query_fixed

        if intent == "general":
            # No code chunks; small repo map for structural context; big history window
            repo_map_budget = int(remaining * 0.15)
            chunks_budget = 0
            history_budget = int(remaining * 0.75)
        elif intent == "follow_up":
            # Some code context, heavier history emphasis
            repo_map_budget = int(remaining * 0.10)
            chunks_budget = int(remaining * 0.30)
            history_budget = int(remaining * 0.50)
        else:  # "code"
            # Prioritise code chunks; modest history
            repo_map_budget = int(remaining * 0.18)
            chunks_budget = int(remaining * 0.57)
            history_budget = int(remaining * 0.15)

        # --- Build system message ---
        system_msg = self._build_system_message(
            project.name,
            project.repo_map,
            repo_map_budget,
            template,
            intent,
        )

        # --- Build history messages ---
        history_msgs = self._build_history_messages(conversation_history, history_budget)

        # --- Build user message with code chunks ---
        user_msg, sources = self._build_user_message(query, search_results, chunks_budget)

        # --- Assemble final messages list ---
        messages: list[dict[str, str]] = [{"role": "system", "content": system_msg}]
        messages.extend(history_msgs)
        messages.append({"role": "user", "content": user_msg})

        total_tokens = sum(_estimate_tokens(m["content"]) for m in messages)

        return RAGContext(
            messages=messages,
            sources=sources,
            token_count=total_tokens,
            search_results_count=len(search_results),
        )

    # ====================================================================
    # Query enrichment
    # ====================================================================

    def _enrich_follow_up_query(
        self,
        query: str,
        history: list[ConversationTurn],
    ) -> str:
        """Enrich a follow-up query with context from the last assistant turn.

        Follow-up queries like "explain that function" have weak semantic signal.
        Appending context from the previous response (file paths, function names,
        technical terms) gives the embedding model a much better chance of finding
        the right code.
        """
        if not history:
            return query

        for turn in reversed(history):
            if turn.role == "assistant":
                context = turn.content[:300]
                return f"{query}\n\nContext: {context}"

        return query

    # ====================================================================
    # Deduplication
    # ====================================================================

    def _dedup_overlapping(self, results: list) -> list:
        """Merge search results from the same file with overlapping or adjacent line ranges.

        When a large function is split into multiple chunks, both halves often
        get retrieved (similar embeddings). This merges them into a single result
        so the LLM sees one continuous block instead of duplicated fragments.
        """
        from collections import defaultdict

        by_file: dict[str, list] = defaultdict(list)
        for r in results:
            by_file[r.file_path].append(r)

        merged: list = []
        for file_results in by_file.values():
            file_results.sort(key=lambda r: r.line_start)

            current = file_results[0]
            for next_r in file_results[1:]:
                # Overlap or adjacent (within 3 lines)
                if next_r.line_start <= current.line_end + 3:
                    current = self._merge_results(current, next_r)
                else:
                    merged.append(current)
                    current = next_r
            merged.append(current)

        merged.sort(key=lambda r: r.relevance_score, reverse=True)
        return merged

    def _merge_results(self, a, b):
        """Merge two overlapping SearchResult objects into one."""
        a_lines = a.text.split("\n")
        b_lines = b.text.split("\n")

        overlap_start = b.line_start - a.line_start
        if overlap_start < len(a_lines):
            b_offset = a.line_end - b.line_start + 1
            if b_offset < len(b_lines):
                merged_text = a.text + "\n" + "\n".join(b_lines[b_offset:])
            else:
                merged_text = a.text
        else:
            merged_text = a.text + "\n" + b.text

        return type(a)(
            chunk_id=a.chunk_id,
            text=merged_text,
            file_path=a.file_path,
            language=a.language,
            function_name=a.function_name or b.function_name,
            class_name=a.class_name or b.class_name,
            line_start=min(a.line_start, b.line_start),
            line_end=max(a.line_end, b.line_end),
            chunk_type=a.chunk_type,
            relevance_score=max(a.relevance_score, b.relevance_score),
        )

    # ====================================================================
    # Builders
    # ====================================================================

    def _build_system_message(
        self,
        project_name: str,
        repo_map: str | None,
        repo_map_budget: int,
        template: str = "code_qa",
        intent: str = "code",
    ) -> str:
        """Build the system prompt, adapting to query intent."""
        if intent == "general":
            parts = [get_prompt_template("general", project_name)]
        elif intent == "follow_up":
            parts = [get_prompt_template(template, project_name)]
            parts.append(
                "\nThe user is following up on a previous topic. Use conversation "
                "history to understand context. Reference code if relevant."
            )
        else:
            parts = [get_prompt_template(template, project_name)]

        if repo_map:
            truncated_map = _truncate_to_tokens(repo_map, repo_map_budget)
            parts.append("")
            parts.append("## Repository Structure")
            parts.append(truncated_map)

        return "\n".join(parts)

    def _build_history_messages(
        self,
        history: list[ConversationTurn],
        budget: int,
    ) -> list[dict[str, str]]:
        """Build history messages within token budget, prioritising recent turns."""
        if not history:
            return []

        msgs: list[dict[str, str]] = []
        tokens_used = 0

        # Walk backwards (newest first) to keep recent context
        selected: list[dict[str, str]] = []
        for turn in reversed(history):
            turn_tokens = _estimate_tokens(turn.content)
            if tokens_used + turn_tokens > budget:
                break
            selected.append({"role": turn.role, "content": turn.content})
            tokens_used += turn_tokens

        selected.reverse()
        msgs.extend(selected)
        return msgs

    def _build_user_message(
        self,
        query: str,
        search_results: list,
        chunks_budget: int,
    ) -> tuple[str, list[CodeReference]]:
        """Build the user message with code chunks and the query."""
        parts: list[str] = []
        sources: list[CodeReference] = []
        tokens_used = 0

        if search_results and chunks_budget > 0:
            parts.append("## Relevant Code")
            parts.append(
                "The following code snippets were retrieved from the repository. "
                "Use them to answer the question. Cite sources by file path and line numbers.\n"
            )
            for i, result in enumerate(search_results, 1):
                location = f"{result.file_path}:{result.line_start}-{result.line_end}"
                label = result.function_name or result.class_name or result.chunk_type
                header = f"### [{i}] {location}"
                if label:
                    header += f" — {label}"

                chunk_text = f"{header}\n```{result.language}\n{result.text}\n```\n"
                chunk_tokens = _estimate_tokens(chunk_text)

                if tokens_used + chunk_tokens > chunks_budget:
                    remaining = chunks_budget - tokens_used
                    if remaining > 100:
                        truncated = _truncate_to_tokens(result.text, remaining - 50)
                        chunk_text = (
                            f"{header}\n```{result.language}\n{truncated}\n```\n"
                        )
                        parts.append(chunk_text)
                        sources.append(self._result_to_reference(result))
                    break

                parts.append(chunk_text)
                tokens_used += chunk_tokens
                sources.append(self._result_to_reference(result))

        parts.append("\n## Question")
        parts.append(query)

        return "\n".join(parts), sources

    def _result_to_reference(self, result) -> CodeReference:
        """Convert a SearchResult to a CodeReference for source tracking."""
        return CodeReference(
            file_path=result.file_path,
            line_start=result.line_start,
            line_end=result.line_end,
            code_snippet=result.text[:500],
            relevance_score=result.relevance_score,
        )


# Global instance
rag_service = RAGService()
