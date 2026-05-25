"""
Query router service — classifies user messages to decide whether code retrieval is needed.
Uses fast heuristics; no LLM call needed per query.

Classification tiers (evaluated in order, first match wins):
  1. Tiny messages with no code keywords → "general"  (length guard, O(1))
  2. General/conversational regex patterns → "general"
  3. Code keyword fast-path → "code"                  (skip follow-up check)
  4. Follow-up regex patterns (only when history exists) → "follow_up"
  5. Default → "code"

The default is always "code" — it's safe to search unnecessarily, but it
costs ~100ms for the embedding call, so we try to avoid it for messages that
are clearly conversational.
"""
import re

from logger import logger


# ============================================================================
# Fast-path sets (O(1) lookup before running any regex)
# ============================================================================

# If a message contains any of these words it is almost certainly about code.
# Checked before follow-up patterns so a message like "what does that function
# do?" doesn't get misclassified as follow_up.
_CODE_KEYWORDS = frozenset({
    "function", "method", "class", "module", "file", "import", "export",
    "variable", "constant", "type", "interface", "struct", "enum", "trait",
    "error", "exception", "bug", "fix", "test", "api", "endpoint", "route",
    "database", "query", "schema", "model", "service", "config", "env",
    "dependency", "package", "library", "framework", "component", "hook",
    "middleware", "decorator", "annotation", "async", "await", "callback",
    "loop", "recursion", "algorithm", "data structure", "implement", "return",
    "parameter", "argument", "constructor", "inherit", "override", "abstract",
    "static", "public", "private", "protected", "main", "init", "setup",
    "deploy", "build", "compile", "lint", "format", "parse", "serialize",
    "authenticate", "authorize", "token", "session", "cache", "queue",
    # Common derived forms that appear in natural queries
    "authentication", "authorization", "configuration", "implementation",
    "endpoints", "routes", "components", "handlers", "middleware",
    "functions", "methods", "classes", "modules", "variables", "constants",
    "errors", "exceptions", "tests", "models", "services", "schemas",
})

# ============================================================================
# Regex patterns
# ============================================================================

# Follow-up signals: user is continuing or clarifying a previous topic
_FOLLOW_UP_PATTERNS = [
    r"\b(tell me more|elaborate|continue|go on|keep going)\b",
    r"\bexplain (that|this|it|above|more|further)\b",
    r"\bwhat (about|does that|do you mean(?: by that)?|did you mean)\b",
    r"\b(can you|could you)\s+(clarify|expand|simplify|rephrase|reword)\b",
    r"\bmore (details?|info|information|context|examples?)\b",
    r"\bhow (so|does that work|come)\b",
    r"\bi (don'?t|still don'?t|didn'?t) (understand|get it|follow)\b",
    r"\bwhat'?s (that|this) mean\b",
    r"\bgive me an? example\b",
    # Short pronoun-anchored follow-ups ("and the X?", "what about X?")
    r"^and (the|that|this|those|these|its|their)\b",
    r"\bshow (that|it|this) again\b",
    r"\band how does that\b",
    r"\b(so|and) (why|how|what|when|where) (is|does|did|was|are|were) (that|this|it)\b",
    r"\b(why|how) (so|is that|does that|would that)\b",
]

# General/conversational signals: no codebase context needed at all
_GENERAL_PATTERNS = [
    r"^(hi+|hello+|hey+|howdy|sup|yo|greetings)\W*$",
    r"^(thanks?|thank you|ty|thx|cheers)\W*$",
    r"^(ok+|okay|got it|i see|understood|makes sense|sounds good|cool|nice|great|awesome|perfect|sure|alright)\W*$",
    r"^(good (morning|afternoon|evening|night|day))\W*$",
    r"\b(how are you|what'?s up|how'?s it going|how'?s your day)\b",
    r"^(bye|goodbye|see you|later|cya)\W*$",
    r"^(lol|lmao|haha|hehe)\W*$",
    # Short acknowledgements not caught by the anchored patterns above
    r"^(interesting|noted|fair enough|makes sense|right|yep|nope|yup|nah)\W*$",
    r"^(that('?s| is) (helpful|great|clear|perfect|good|interesting))\W*$",
]

# Max word count for the length guard. Messages this short with no code
# keywords are almost certainly conversational. Avoids running all the regex.
# Kept at 3 so 4-word queries like "how does X work?" are NOT short-circuited
# — they fall through to the full regex + keyword tiers instead.
_SHORT_MSG_WORD_LIMIT = 3


class RouterService:
    """Classifies queries using fast heuristics — no per-query LLM call."""

    async def classify(
        self,
        message: str,
        recent_history: list[dict] | None = None,
    ) -> str:
        """
        Classify a user message intent.

        Returns: "code", "general", or "follow_up"
        Default is "code" — always search if uncertain.
        """
        msg = message.lower().strip()
        has_history = bool(recent_history)
        words = msg.split()

        # ── Tier 1: length guard ─────────────────────────────────────────────
        # Very short messages with zero code keywords are conversational.
        # This saves all subsequent regex work for the common "ok / thanks /
        # got it" cases that appear between real code questions.
        if len(words) <= _SHORT_MSG_WORD_LIMIT:
            has_code_kw = any(w in _CODE_KEYWORDS for w in words)
            if not has_code_kw:
                # Still run general patterns to confirm (they're anchored, fast)
                for pattern in _GENERAL_PATTERNS:
                    if re.search(pattern, msg):
                        logger.debug(f"Intent: general (short+pattern) | {message[:60]}")
                        return "general"
                # Short, no code keywords, no strong follow-up signal → general
                # UNLESS there's history and it looks like a follow-up
                if has_history:
                    for pattern in _FOLLOW_UP_PATTERNS:
                        if re.search(pattern, msg):
                            logger.debug(f"Intent: follow_up (short) | {message[:60]}")
                            return "follow_up"
                # Short message, no code keywords, no follow-up pattern → general
                logger.debug(f"Intent: general (short, no code kw) | {message[:60]}")
                return "general"

        # ── Tier 2: general pattern match (longer messages) ──────────────────
        for pattern in _GENERAL_PATTERNS:
            if re.search(pattern, msg):
                logger.debug(f"Intent: general | {message[:60]}")
                return "general"

        # ── Tier 3: code keyword fast-path ───────────────────────────────────
        # If a known code term appears, skip the follow-up check entirely.
        # "what does that function return?" has "function" → code, not follow_up.
        if any(w in _CODE_KEYWORDS for w in words):
            logger.debug(f"Intent: code (keyword) | {message[:60]}")
            return "code"

        # ── Tier 4: follow-up pattern match ──────────────────────────────────
        if has_history:
            for pattern in _FOLLOW_UP_PATTERNS:
                if re.search(pattern, msg):
                    logger.debug(f"Intent: follow_up | {message[:60]}")
                    return "follow_up"

        # ── Tier 5: default ──────────────────────────────────────────────────
        logger.debug(f"Intent: code (default) | {message[:60]}")
        return "code"


# Global instance
router_service = RouterService()
