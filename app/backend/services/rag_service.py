"""
RAG context assembly service.
Builds structured LLM prompts from search results, repo map, and conversation history.
Routes queries through intent classification to skip retrieval for non-code messages.
Uses rolling conversation summaries to maintain long-term memory.
"""
from config import settings
from logger import logger
from models import RAGContext, CodeReference, ConversationTurn
from storage.metadata_db import db
from services.search_service import search_service
from services.llm_service import get_prompt_template
from services.router_service import router_service

# After this many messages, trigger summarization of older ones
SUMMARIZE_THRESHOLD = 10


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
        3. Dynamically allocates token budget
        4. Injects rolling conversation summary for long-term memory
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
            search_results = await search_service.hybrid_search(
                project_id, enriched
            )

        # --- Dynamic token budget based on intent ---
        total_budget = settings.MAX_CONTEXT_TOKENS
        system_fixed = 500
        query_fixed = 200
        remaining = total_budget - system_fixed - query_fixed

        if intent == "general":
            # No code chunks but still include repo map so the LLM
            # has structural context and won't hallucinate about the project
            repo_map_budget = int(remaining * 0.15)
            chunks_budget = 0
            history_budget = int(remaining * 0.60)
            summary_budget = int(remaining * 0.15)
        elif intent == "follow_up":
            # Some code, more history emphasis
            repo_map_budget = int(remaining * 0.10)
            chunks_budget = int(remaining * 0.25)
            history_budget = int(remaining * 0.40)
            summary_budget = int(remaining * 0.15)
        else:  # "code"
            # Original allocation, plus summary budget carved from remaining
            repo_map_budget = int(remaining * 0.18)
            chunks_budget = int(remaining * 0.47)
            history_budget = int(remaining * 0.15)
            summary_budget = int(remaining * 0.10)

        # --- Rolling summary for long-term memory ---
        summary_block = ""
        if conversation_id:
            summary_block = await self._maybe_summarize(
                conversation_id, conversation_history, summary_budget
            )

        # --- Build system message ---
        # Always include repo map so the LLM has structural context
        system_msg = self._build_system_message(
            project.name,
            project.repo_map,
            repo_map_budget,
            template,
            intent,
        )

        # --- Build history messages (with summary prepended) ---
        history_msgs = self._build_history_messages(
            conversation_history, history_budget, summary_block
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

    # ====================================================================
    # Summarization
    # ====================================================================

    async def _maybe_summarize(
        self,
        conversation_id: str,
        history: list[ConversationTurn],
        summary_budget: int,
    ) -> str:
        """
        Check if conversation needs summarization and return the summary block.

        Summarization triggers when total messages exceed SUMMARIZE_THRESHOLD.
        Only unsummarized messages get sent to the summarizer.
        The summary is stored in the DB and reused until more messages arrive.
        """
        total_messages = len(history)

        # Load existing summary state
        existing_summary, summarized_up_to = db.get_conversation_summary(
            conversation_id
        )

        if total_messages < SUMMARIZE_THRESHOLD:
            # Not enough messages to bother summarizing
            if existing_summary:
                return _truncate_to_tokens(
                    f"## Earlier in this conversation\n{existing_summary}",
                    summary_budget,
                )
            return ""

        # Check if we need to re-summarize (new messages since last summary)
        unsummarized_count = total_messages - summarized_up_to
        if unsummarized_count >= SUMMARIZE_THRESHOLD // 2 or not existing_summary:
            # Summarize everything except the most recent 6 messages
            keep_recent = 6
            messages_to_summarize = history[: total_messages - keep_recent]

            if messages_to_summarize:
                msgs_for_llm = [
                    {"role": m.role, "content": m.content}
                    for m in messages_to_summarize
                ]

                # If there's an existing summary, include it as context
                if existing_summary:
                    msgs_for_llm.insert(
                        0,
                        {
                            "role": "system",
                            "content": f"Previous summary: {existing_summary}",
                        },
                    )

                new_summary = await router_service.summarize(msgs_for_llm)
                new_up_to = total_messages - keep_recent

                db.update_conversation_summary(
                    conversation_id, new_summary, new_up_to
                )
                existing_summary = new_summary
                logger.info(
                    f"Summarized conversation {conversation_id}: "
                    f"{new_up_to} messages condensed"
                )

        if existing_summary:
            return _truncate_to_tokens(
                f"## Earlier in this conversation\n{existing_summary}",
                summary_budget,
            )
        return ""

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
        Appending context from the previous response (which contains file paths,
        function names, and technical terms) gives the embedding model a much
        better chance of finding the right code.
        """
        if len(history) < 2:
            return query

        # Skip the current query (last in history) and find previous assistant response
        prev_turns = history[:-1]
        for turn in reversed(prev_turns):
            if turn.role == "assistant":
                # Take first ~300 chars of the response for search grounding
                context = turn.content[:300]
                return f"{query}\n\nContext: {context}"

        return query

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
        parts = [get_prompt_template(template, project_name)]

        if intent in ("general", "follow_up"):
            parts.append(
                "\nThe user may be having a general conversation or following up "
                "on a previous topic. Respond naturally. You don't need to reference "
                "code unless it's relevant to what they're asking."
            )

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
        summary_block: str = "",
    ) -> list[dict[str, str]]:
        """
        Build history messages with optional summary prepended.

        If a summary exists, it's injected as a system message before
        the recent conversation turns.
        """
        msgs: list[dict[str, str]] = []
        tokens_used = 0

        # Inject summary as context
        if summary_block:
            summary_tokens = _estimate_tokens(summary_block)
            if summary_tokens < budget:
                msgs.append({"role": "system", "content": summary_block})
                tokens_used += summary_tokens

        if not history:
            return msgs

        # Walk backwards (newest first) to prioritize recent context
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
