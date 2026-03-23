"""
RAG context assembly service.
Builds structured LLM prompts from search results, repo map, and conversation history.
"""
from config import settings
from logger import logger
from models import RAGContext, CodeReference, ConversationTurn
from storage.metadata_db import db
from services.search_service import search_service
from services.llm_service import get_prompt_template


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

        Args:
            project_id: Project to search in.
            query: User's question.
            conversation_id: If provided, loads history from DB (overrides conversation_history).
            conversation_history: Explicit history turns (used when conversation_id is None).
            template: Prompt template name (code_qa, bug_detection, code_navigation, code_explanation).

        Returns RAGContext with messages in OpenAI/Ollama chat format:
          [system, ...history, user]
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

        # Search for relevant code chunks
        search_results = await search_service.hybrid_search(project_id, query)

        # --- Token budget allocation ---
        total_budget = settings.MAX_CONTEXT_TOKENS
        system_fixed = 500
        query_fixed = 200
        remaining = total_budget - system_fixed - query_fixed
        repo_map_budget = int(remaining * 0.20)
        chunks_budget = int(remaining * 0.55)
        history_budget = int(remaining * 0.15)

        # --- Build system message ---
        system_msg = self._build_system_message(
            project.name, project.repo_map, repo_map_budget, template
        )

        # --- Build history messages ---
        history_msgs = self._build_history_messages(
            conversation_history, history_budget
        )

        # --- Build user message with code chunks ---
        user_msg, sources = self._build_user_message(
            query, search_results, chunks_budget
        )

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

    def _build_system_message(
        self,
        project_name: str,
        repo_map: str | None,
        repo_map_budget: int,
        template: str = "code_qa",
    ) -> str:
        """Build the system prompt from a template with optional repo map."""
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
        """
        Select recent conversation turns that fit within the token budget.

        Selects from newest to oldest, then outputs in chronological order.
        """
        if not history:
            return []

        selected: list[dict[str, str]] = []
        tokens_used = 0

        # Walk backwards (newest first) to prioritize recent context
        for turn in reversed(history):
            turn_tokens = _estimate_tokens(turn.content)
            if tokens_used + turn_tokens > budget:
                break
            selected.append({"role": turn.role, "content": turn.content})
            tokens_used += turn_tokens

        # Reverse to chronological order
        selected.reverse()
        return selected

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

        if search_results:
            parts.append("## Relevant Code\n")
            for i, result in enumerate(search_results, 1):
                # Format chunk header
                location = f"{result.file_path}:{result.line_start}-{result.line_end}"
                label = result.function_name or result.class_name or result.chunk_type
                header = f"### [{i}] {location}"
                if label:
                    header += f" ({label})"

                chunk_text = f"{header}\n```{result.language}\n{result.text}\n```\n"
                chunk_tokens = _estimate_tokens(chunk_text)

                if tokens_used + chunk_tokens > chunks_budget:
                    # Try to fit a truncated version
                    remaining = chunks_budget - tokens_used
                    if remaining > 100:
                        truncated = _truncate_to_tokens(result.text, remaining - 50)
                        chunk_text = f"{header}\n```{result.language}\n{truncated}\n```\n"
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
