"""
Query router service — classifies user messages to decide whether code retrieval is needed.
Uses a small/fast LLM (e.g. lfm2.5-thinking) via Ollama for accurate intent classification.
"""
import json
import re

import httpx

from config import settings
from logger import logger


CLASSIFY_PROMPT = """Classify this chat message into ONE category. The chat is about a code repository.

Categories:
- "code": asks about code, files, functions, bugs, the project/repo itself, its purpose, architecture, dependencies, or ANYTHING that needs codebase context to answer properly
- "follow_up": continues or clarifies a previous topic from the conversation (e.g. "explain that again", "what about X part", "tell me more", "can you simplify that")
- "general": ONLY casual talk, greetings, thanks, or general knowledge questions completely unrelated to THIS codebase

If in doubt, choose "code". Questions like "what is this project", "what does this repo do", "how is this structured" are ALWAYS "code".

Recent conversation:
{history_snippet}

Message: {message}

Reply with ONLY: {{"intent": "<category>"}}"""

SUMMARIZE_PROMPT = """Summarize this conversation concisely. Preserve:
- Key topics discussed (specific files, functions, concepts)
- Decisions or conclusions reached
- Any user preferences or corrections mentioned

Keep it under 300 words. Write in third person ("The user asked about...", "The assistant explained...").

Conversation:
{conversation}"""


class RouterService:
    """Classifies queries and summarizes conversations using a small local LLM."""

    async def classify(
        self,
        message: str,
        recent_history: list[dict[str, str]] | None = None,
    ) -> str:
        """
        Classify a user message intent.

        Returns: "code", "general", or "follow_up"
        """
        # Build a short history snippet (last 2 messages) for context
        history_snippet = "None"
        if recent_history:
            last_two = recent_history[-2:]
            lines = [f"{m['role']}: {m['content'][:150]}" for m in last_two]
            history_snippet = "\n".join(lines)

        prompt = CLASSIFY_PROMPT.format(
            history_snippet=history_snippet,
            message=message,
        )

        try:
            raw = await self._call_small_llm(prompt, max_tokens=50)
            return self._parse_intent(raw)
        except Exception as e:
            logger.warning(f"Router classification failed, defaulting to 'code': {e}")
            return "code"  # Safe fallback — always search if unsure

    async def summarize(
        self,
        messages: list[dict[str, str]],
    ) -> str:
        """
        Summarize a list of conversation messages into a compact paragraph.

        Args:
            messages: List of {role, content} dicts to summarize.

        Returns:
            A concise summary string.
        """
        conversation = "\n".join(
            f"{m['role'].title()}: {m['content']}" for m in messages
        )
        prompt = SUMMARIZE_PROMPT.format(conversation=conversation)

        try:
            summary = await self._call_small_llm(prompt, max_tokens=500)
            return summary.strip()
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            # Fallback: just keep the last few messages as-is
            fallback_lines = []
            for m in messages[-4:]:
                fallback_lines.append(f"{m['role']}: {m['content'][:100]}")
            return "\n".join(fallback_lines)

    # ====================================================================
    # Internals
    # ====================================================================

    async def _call_small_llm(self, prompt: str, max_tokens: int = 200) -> str:
        """Call the small router model via Ollama /api/generate."""
        url = f"{settings.OLLAMA_BASE_URL}/api/generate"
        # Thinking models need extra tokens for their <think> reasoning
        predict_tokens = max_tokens + 300
        payload = {
            "model": settings.ROUTER_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": predict_tokens,
                "temperature": 0.1,  # Low temp for deterministic classification
            },
        }

        timeout = httpx.Timeout(30.0, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            raw = data.get("response", "")
            return self._strip_thinking(raw)

    def _strip_thinking(self, text: str) -> str:
        """Strip <think>...</think> blocks from thinking-model output."""
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return cleaned.strip()

    def _parse_intent(self, raw: str) -> str:
        """Extract intent from LLM response, handling various formats."""
        text = raw.strip().lower()

        # Try JSON parse first
        try:
            # Find JSON object in response (model might wrap it in markdown etc.)
            json_match = re.search(r'\{[^}]+\}', text)
            if json_match:
                parsed = json.loads(json_match.group())
                intent = parsed.get("intent", "").strip().lower()
                if intent in ("code", "general", "follow_up"):
                    return intent
        except (json.JSONDecodeError, AttributeError):
            pass

        # Fallback: look for keywords in raw text
        if "general" in text:
            return "general"
        if "follow_up" in text or "follow-up" in text:
            return "follow_up"
        if "code" in text:
            return "code"

        return "code"  # Default to code (safe — always search)


# Global instance
router_service = RouterService()
