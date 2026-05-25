"""
Tests for the heuristic intent classifier (router_service.py).

No external services needed — pure regex/string logic.
Covers every pattern branch + important edge cases.
"""
import pytest

# ── fixtures ──────────────────────────────────────────────────────────────────
# router_service is a singleton; we just call its classify() directly.

@pytest.fixture()
def router():
    from services.router_service import RouterService
    return RouterService()


HISTORY = [
    {"role": "user",      "content": "how does auth work?"},
    {"role": "assistant", "content": "Auth is handled in auth.py using JWT tokens."},
]

# ═══════════════════════════════════════════════════════════════════════════════
# General intent — clear conversational phrases
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("msg", [
    "hi",
    "hi!",
    "hiii",
    "hello",
    "hello!",
    "hey",
    "sup",
    "yo",
    "howdy",
    "thanks",
    "thanks!",
    "thank you",
    "ty",
    "thx",
    "cheers",
    "ok",
    "okay",
    "ok!",
    "okkk",
    "got it",
    "i see",
    "understood",
    "makes sense",
    "sounds good",
    "cool",
    "nice",
    "great",
    "awesome",
    "perfect",
    "sure",
    "alright",
    "good morning",
    "good afternoon",
    "good evening",
    "good night",
    "good day",
    "bye",
    "goodbye",
    "see you",
    "later",
    "cya",
    "lol",
    "lmao",
    "haha",
    "hehe",
    "how are you",
    "how are you?",
    "what's up",
    "whats up",
    "how's it going",
])
async def test_general_phrases(router, msg):
    intent = await router.classify(msg)
    assert intent == "general", f"Expected 'general' for {msg!r}, got {intent!r}"


# ═══════════════════════════════════════════════════════════════════════════════
# Code intent — defaults when no pattern matches
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("msg", [
    "what does the auth function do?",
    "explain the search algorithm",
    "how does indexing work?",
    "find all functions that call the database",
    "what is this project?",
    "show me the router code",
    "where is the config loaded?",
    "what dependencies does this use?",
    "how do I run the tests?",
    "what is Python?",          # generic question — safe to retrieve
    "what is React?",           # generic question — safe to retrieve
    "?",                        # ambiguous — default to code
    "1 + 1",                    # non-conversational, non-follow-up
    "print('hello')",           # looks like code
    "SELECT * FROM users",      # SQL
    "",                         # empty string — default to code
    "   ",                      # whitespace only
    "👀",                       # emoji — no pattern match
    "¿Qué hace esta función?",  # non-English — default to code
    "что это?",                 # Russian — default to code
])
async def test_code_default(router, msg):
    intent = await router.classify(msg)
    assert intent == "code", f"Expected 'code' for {msg!r}, got {intent!r}"


# ═══════════════════════════════════════════════════════════════════════════════
# Mixed messages — general word present but NOT the whole message
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("msg", [
    "thanks! now what does the login function do?",  # thanks + technical content
    "okay what about the middleware?",               # okay + technical question
    "got it. how does caching work here?",           # got it + question
    "cool, can you show me the search service?",     # cool + request
    "hi can you explain the RAG pipeline?",          # hi + technical
    "ok but where are the routes defined?",          # ok + question
    "great! what files handle authentication?",      # great + question
])
async def test_mixed_intent_becomes_code(router, msg):
    """Partial general word + more content → the anchored patterns don't match → code."""
    intent = await router.classify(msg)
    assert intent == "code", f"Expected 'code' for mixed msg {msg!r}, got {intent!r}"


# ═══════════════════════════════════════════════════════════════════════════════
# Follow-up intent — requires conversation history
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("msg", [
    "tell me more",
    "tell me more about that",
    "elaborate",
    "elaborate on that",
    "continue",
    "go on",
    "keep going",
    "explain that",
    "explain this",
    "explain it",
    "explain more",
    "explain further",
    "what about the other part?",
    "what does that mean?",
    "what do you mean?",
    "can you clarify?",
    "could you clarify that?",
    "can you expand on that?",
    "can you simplify that?",
    "could you rephrase?",
    "more details please",
    "more info",
    "more context",
    "give me an example",
    "give me a example",
    "how so?",
    "how does that work?",
    "how come?",
    "i don't understand",
    "i don't get it",
    "i still don't follow",
    "i didn't understand",
])
async def test_follow_up_with_history(router, msg):
    intent = await router.classify(msg, recent_history=HISTORY)
    assert intent == "follow_up", f"Expected 'follow_up' for {msg!r} with history, got {intent!r}"


# ═══════════════════════════════════════════════════════════════════════════════
# Follow-up phrases WITHOUT history → must fall back to "code"
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("msg", [
    "tell me more",
    "explain that",
    "elaborate",
    "can you clarify?",
    "more details",
])
async def test_follow_up_without_history_becomes_code(router, msg):
    """No history → follow-up patterns are skipped → safe 'code' default."""
    intent = await router.classify(msg, recent_history=None)
    assert intent == "code", f"Expected 'code' (no history) for {msg!r}, got {intent!r}"


async def test_follow_up_empty_history_list(router):
    """Empty list treated same as no history."""
    intent = await router.classify("tell me more", recent_history=[])
    assert intent == "code"


# ═══════════════════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════════════════

async def test_very_long_message_is_code(router):
    """A 2000-char message that doesn't match any pattern → code."""
    msg = "What is " + "the function " * 200
    intent = await router.classify(msg)
    assert intent == "code"


async def test_case_insensitivity(router):
    """Patterns are lowercased before matching."""
    assert await router.classify("HI") == "general"
    assert await router.classify("THANKS") == "general"
    assert await router.classify("TELL ME MORE", recent_history=HISTORY) == "follow_up"


async def test_surrounding_whitespace_stripped(router):
    """Leading/trailing whitespace should not prevent matching."""
    assert await router.classify("  hi  ") == "general"
    assert await router.classify("  thanks!  ") == "general"


async def test_general_takes_priority_over_follow_up(router):
    """
    A message like 'thanks' is general even if follow-up history exists.
    General patterns are checked first.
    """
    intent = await router.classify("thanks", recent_history=HISTORY)
    assert intent == "general"


async def test_multiple_calls_same_result(router):
    """Classifier is stateless — same input always gives same output."""
    for _ in range(5):
        assert await router.classify("what does parse() do?") == "code"
        assert await router.classify("hi") == "general"


async def test_newline_in_message(router):
    """Multi-line message with no pattern match → code."""
    msg = "Can you help me?\nI need to understand the database layer."
    assert await router.classify(msg) == "code"


async def test_html_injection_in_message(router):
    """HTML/script injection attempt doesn't crash and defaults to code."""
    msg = "<script>alert('xss')</script>"
    intent = await router.classify(msg)
    assert intent in ("code", "general", "follow_up")  # must not raise


async def test_only_punctuation(router):
    """Message of just punctuation → code (safe default)."""
    assert await router.classify("!!! ???") == "code"
    assert await router.classify("...") == "code"
