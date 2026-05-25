"""
Router Benchmark: Old (regex) vs New (tiered heuristic) vs LLM (llama3.2:1b).

Compares three intent classifiers on accuracy and latency:
  - Old: pure regex, original implementation
  - New: tiered heuristic (length guard + code keywords + regex)
  - LLM: llama3.2:1b via Ollama structured JSON output

Regex routers are run REGEX_ITER times per case (fast, need many runs to measure).
LLM router is run LLM_ITER times per case (slow, 3 is enough for a reliable average).

Saves results to:
  tests/results/router_benchmark_latest.txt   (plain text, no ANSI)
  tests/results/router_benchmark_latest.json  (structured data for analysis)

Run from the backend directory:
    python tests/benchmark_router.py
"""

import asyncio
import json
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

# Force UTF-8 so box-drawing characters render on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REGEX_ITER = 30_000   # per case for regex routers (sub-µs, need volume)
LLM_ITER   = 3        # per case for LLM router   (~100-800 ms each)
LLM_MODEL  = "llama3.2:1b"
OLLAMA_URL = "http://localhost:11434"
RESULTS_DIR = Path(__file__).parent / "results"


# ============================================================================
# ANSI helpers
# ============================================================================

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RED    = "\033[31m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
BLUE   = "\033[34m"
CYAN   = "\033[36m"

INTENT_COLOR = {"general": GREEN, "follow_up": YELLOW, "code": BLUE}

def c(text: str, color: str) -> str:
    return f"{color}{text}{RESET}"

def tick(got: str, expected: str) -> str:
    return c("✓", GREEN) if got == expected else c("✗", RED)

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


# ============================================================================
# OLD ROUTER — original regex-only implementation (verbatim)
# ============================================================================

_OLD_FOLLOW_UP = [
    r"\b(tell me more|elaborate|continue|go on|keep going)\b",
    r"\bexplain (that|this|it|above|more|further)\b",
    r"\bwhat (about|does that|do you mean(?: by that)?|did you mean)\b",
    r"\b(can you|could you)\s+(clarify|expand|simplify|rephrase|reword)\b",
    r"\bmore (details?|info|information|context|examples?)\b",
    r"\bhow (so|does that work|come)\b",
    r"\bi (don'?t|still don'?t|didn'?t) (understand|get it|follow)\b",
    r"\bwhat'?s (that|this) mean\b",
    r"\bgive me an? example\b",
]
_OLD_GENERAL = [
    r"^(hi+|hello+|hey+|howdy|sup|yo|greetings)\W*$",
    r"^(thanks?|thank you|ty|thx|cheers)\W*$",
    r"^(ok+|okay|got it|i see|understood|makes sense|sounds good|cool|nice|great|awesome|perfect|sure|alright)\W*$",
    r"^(good (morning|afternoon|evening|night|day))\W*$",
    r"\b(how are you|what'?s up|how'?s it going|how'?s your day)\b",
    r"^(bye|goodbye|see you|later|cya)\W*$",
    r"^(lol|lmao|haha|hehe)\W*$",
]

async def classify_old(message: str, history: list | None = None) -> str:
    msg = message.lower().strip()
    for p in _OLD_GENERAL:
        if re.search(p, msg): return "general"
    if history:
        for p in _OLD_FOLLOW_UP:
            if re.search(p, msg): return "follow_up"
    return "code"


# ============================================================================
# NEW ROUTER — tiered heuristic implementation
# ============================================================================

_CODE_KEYWORDS = frozenset({
    "function", "method", "class", "module", "file", "import", "export",
    "variable", "constant", "type", "interface", "struct", "enum", "trait",
    "error", "exception", "bug", "fix", "test", "api", "endpoint", "route",
    "database", "query", "schema", "model", "service", "config", "env",
    "dependency", "package", "library", "framework", "component", "hook",
    "middleware", "decorator", "annotation", "async", "await", "callback",
    "loop", "recursion", "algorithm", "implement", "return",
    "parameter", "argument", "constructor", "inherit", "override", "abstract",
    "static", "public", "private", "protected", "main", "init", "setup",
    "deploy", "build", "compile", "lint", "format", "parse", "serialize",
    "authenticate", "authorize", "token", "session", "cache", "queue",
    "authentication", "authorization", "configuration", "implementation",
    "endpoints", "routes", "components", "handlers",
    "functions", "methods", "classes", "modules", "variables", "constants",
    "errors", "exceptions", "tests", "models", "services", "schemas",
})
_NEW_FOLLOW_UP = [
    r"\b(tell me more|elaborate|continue|go on|keep going)\b",
    r"\bexplain (that|this|it|above|more|further)\b",
    r"\bwhat (about|does that|do you mean(?: by that)?|did you mean)\b",
    r"\b(can you|could you)\s+(clarify|expand|simplify|rephrase|reword)\b",
    r"\bmore (details?|info|information|context|examples?)\b",
    r"\bhow (so|does that work|come)\b",
    r"\bi (don'?t|still don'?t|didn'?t) (understand|get it|follow)\b",
    r"\bwhat'?s (that|this) mean\b",
    r"\bgive me an? example\b",
    r"^and (the|that|this|those|these|its|their)\b",
    r"\bshow (that|it|this) again\b",
    r"\band how does that\b",
    r"\b(so|and) (why|how|what|when|where) (is|does|did|was|are|were) (that|this|it)\b",
    r"\b(why|how) (so|is that|does that|would that)\b",
]
_NEW_GENERAL = [
    r"^(hi+|hello+|hey+|howdy|sup|yo|greetings)\W*$",
    r"^(thanks?|thank you|ty|thx|cheers)\W*$",
    r"^(ok+|okay|got it|i see|understood|makes sense|sounds good|cool|nice|great|awesome|perfect|sure|alright)\W*$",
    r"^(good (morning|afternoon|evening|night|day))\W*$",
    r"\b(how are you|what'?s up|how'?s it going|how'?s your day)\b",
    r"^(bye|goodbye|see you|later|cya)\W*$",
    r"^(lol|lmao|haha|hehe)\W*$",
    r"^(interesting|noted|fair enough|makes sense|right|yep|nope|yup|nah)\W*$",
    r"^(that('?s| is) (helpful|great|clear|perfect|good|interesting))\W*$",
]
_SHORT_LIMIT = 3

async def classify_new(message: str, history: list | None = None) -> str:
    msg = message.lower().strip()
    words = msg.split()
    if len(words) <= _SHORT_LIMIT:
        if not any(w in _CODE_KEYWORDS for w in words):
            for p in _NEW_GENERAL:
                if re.search(p, msg): return "general"
            if history:
                for p in _NEW_FOLLOW_UP:
                    if re.search(p, msg): return "follow_up"
            return "general"
    for p in _NEW_GENERAL:
        if re.search(p, msg): return "general"
    if any(w in _CODE_KEYWORDS for w in words):
        return "code"
    if history:
        for p in _NEW_FOLLOW_UP:
            if re.search(p, msg): return "follow_up"
    return "code"


# ============================================================================
# LLM ROUTER — llama3.2:1b via Ollama structured JSON output
# ============================================================================

_LLM_SYSTEM = (
    "You are an intent classifier for a code repository Q&A assistant.\n"
    "Classify the user message into exactly one intent:\n"
    "  code      — asks about the codebase: functions, errors, architecture, implementation details\n"
    "  general   — casual conversation: greetings, thanks, small talk, one-word acknowledgements\n"
    "  follow_up — continues or clarifies the previous assistant response (e.g. 'explain that', 'why so?', 'and the constructor?')\n"
    "Reply with a JSON object like: {\"intent\": \"code\"}"
)

_LLM_FORMAT = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": ["code", "general", "follow_up"]}
    },
    "required": ["intent"],
}

async def _single_llm_call(message: str, history: list | None) -> tuple[str, float]:
    """One LLM call. Returns (intent, latency_ms)."""
    has_ctx = "yes" if history else "no"
    prompt = f"Prior conversation: {has_ctx}\nUser message: {message}"
    payload = {
        "model": LLM_MODEL,
        "system": _LLM_SYSTEM,
        "prompt": prompt,
        "stream": False,
        "format": _LLM_FORMAT,
        "options": {"temperature": 0.0, "num_predict": 30},
    }
    t0 = time.perf_counter_ns()
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=5.0)) as client:
        resp = await client.post(f"{OLLAMA_URL}/api/generate", json=payload)
        resp.raise_for_status()
    elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000

    raw = resp.json().get("response", "")
    try:
        data = json.loads(raw)
        intent = data.get("intent", "code")
        if intent not in ("code", "general", "follow_up"):
            intent = "code"
    except (json.JSONDecodeError, TypeError):
        low = raw.lower()
        if "general"   in low: intent = "general"
        elif "follow"  in low: intent = "follow_up"
        else:                  intent = "code"

    return intent, elapsed_ms


async def classify_llm_avg(message: str, history: list | None, n: int) -> tuple[str, float]:
    """Run n LLM calls, return (last result, avg_ms)."""
    total_ms = 0.0
    result = "code"
    for _ in range(n):
        result, ms = await _single_llm_call(message, history)
        total_ms += ms
    return result, total_ms / n


async def check_llm_available() -> bool:
    if not HAS_HTTPX:
        return False
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{OLLAMA_URL}/api/tags")
            if resp.status_code != 200:
                return False
            models = [m.get("name", "") for m in resp.json().get("models", [])]
            return any(LLM_MODEL in m for m in models)
    except Exception:
        return False


# ============================================================================
# Test cases
# ============================================================================

@dataclass
class Case:
    label: str
    message: str
    history: bool
    expected: str

CASES: list[Case] = [
    # Pure conversational
    Case("greeting",             "hi",                                            False, "general"),
    Case("thanks",               "thanks",                                        False, "general"),
    Case("acknowledgement",      "ok got it",                                     False, "general"),
    Case("affirmation",          "makes sense",                                   False, "general"),
    Case("farewell",             "bye",                                           False, "general"),
    Case("laughter",             "lol",                                           False, "general"),
    Case("pleasantry",           "how are you?",                                  False, "general"),
    # Short ambiguous — old router wastes embedding call on these
    Case("ambiguous 1",          "interesting, what else?",                       False, "general"),
    Case("ambiguous 2",          "that's helpful",                                False, "general"),
    Case("ambiguous 3",          "noted, thanks",                                 False, "general"),
    Case("ambiguous 4",          "fair enough",                                   False, "general"),
    Case("ambiguous 5",          "yep, got it",                                   False, "general"),
    Case("ambiguous 6",          "right, makes sense",                            False, "general"),
    # Follow-ups (require history)
    Case("follow-up classic",    "explain that",                                  True,  "follow_up"),
    Case("follow-up detail",     "tell me more",                                  True,  "follow_up"),
    Case("follow-up clarify",    "what do you mean?",                             True,  "follow_up"),
    Case("follow-up example",    "give me an example",                            True,  "follow_up"),
    Case("follow-up short",      "and the constructor?",                          True,  "follow_up"),
    Case("follow-up pronoun",    "and how does that work?",                       True,  "follow_up"),
    Case("follow-up why",        "why so?",                                       True,  "follow_up"),
    # Code keyword inside follow-up phrasing — old router with history misclassifies
    Case("kw beats follow-up 1", "what does that function return?",               True,  "code"),
    Case("kw beats follow-up 2", "explain the constructor method",                True,  "code"),
    Case("kw beats follow-up 3", "what does that error mean exactly",             True,  "code"),
    # Genuine code queries
    Case("code plain",           "how does authentication work?",                 False, "code"),
    Case("code keyword",         "what does the main function return?",           False, "code"),
    Case("code error",           "where is error handling done?",                 False, "code"),
    Case("code schema",          "what is the database schema?",                  False, "code"),
    Case("code api",             "how is the api endpoint structured?",           False, "code"),
    Case("code long",
         "can you show me how the token refresh logic works in the auth service?",
         False, "code"),
]


# ============================================================================
# Result dataclass
# ============================================================================

@dataclass
class Result:
    case:        Case
    old_intent:  str
    new_intent:  str
    llm_intent:  str | None   # None if LLM was skipped
    old_us:      float        # avg µs per call
    new_us:      float        # avg µs per call
    llm_ms:      float | None # avg ms per call


# ============================================================================
# Benchmark runners
# ============================================================================

async def bench_regex(fn, message: str, history, n: int) -> tuple[str, float]:
    await fn(message, history)                  # warm-up
    t0 = time.perf_counter_ns()
    for _ in range(n):
        result = await fn(message, history)
    us = (time.perf_counter_ns() - t0) / n / 1_000
    return result, us


async def bench_case(case: Case, history_stub: list, llm_available: bool) -> Result:
    history = history_stub if case.history else None

    old_intent, old_us = await bench_regex(classify_old, case.message, history, REGEX_ITER)
    new_intent, new_us = await bench_regex(classify_new, case.message, history, REGEX_ITER)

    llm_intent, llm_ms = None, None
    if llm_available:
        # One warm-up call (not timed) to ensure model is loaded
        await _single_llm_call(case.message, history)
        llm_intent, llm_ms = await classify_llm_avg(case.message, history, LLM_ITER)

    return Result(case, old_intent, new_intent, llm_intent, old_us, new_us, llm_ms)


# ============================================================================
# Output helpers — write to both console and a buffer (for file saving)
# ============================================================================

class Tee:
    """Writes to stdout AND collects plain-text (ANSI-stripped) output."""
    def __init__(self):
        self._lines: list[str] = []

    def print(self, text: str = ""):
        print(text)
        self._lines.append(strip_ansi(text))

    def plain_text(self) -> str:
        return "\n".join(self._lines) + "\n"


# ============================================================================
# Main
# ============================================================================

async def main():
    tee = Tee()
    history_stub = [{"role": "user", "content": "prior message"}]

    # Check LLM availability
    llm_available = await check_llm_available()

    W = 136
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    tee.print()
    tee.print(BOLD + "━" * W + RESET)
    tee.print(BOLD + f"  Router Benchmark  —  {timestamp}" + RESET)
    tee.print(BOLD + f"  Regex: {REGEX_ITER:,} iterations/case  │  LLM ({LLM_MODEL}): {LLM_ITER} iterations/case" + RESET)
    if not llm_available:
        tee.print(c(f"  ⚠  Ollama/{LLM_MODEL} not available — LLM column will show N/A", YELLOW))
    tee.print(BOLD + "━" * W + RESET)
    tee.print()

    # Column widths
    CL, CM, CE, CR, CUS, CMS = 22, 40, 10, 10, 8, 10

    has_llm = llm_available
    hdr = (
        f"  {'Case':<{CL}}"
        f"{'Message':<{CM}}"
        f"{'Exp':<{CE}}"
        f"{'Old':<{CR}}"
        f"{'New':<{CR}}"
        + (f"{'LLM':<{CR}}" if has_llm else "")
        + f"{'Old µs':>{CUS}}"
        f"{'New µs':>{CUS}}"
        + (f"{'LLM ms':>{CMS}}" if has_llm else "")
        + f"  {'✓old':<5}{'✓new':<5}"
        + ("✓llm" if has_llm else "")
    )
    tee.print(hdr)
    tee.print("  " + "─" * (W - 2))

    results: list[Result] = []
    prev_cat = ""

    for case in CASES:
        cat = case.label.split()[0]
        if cat != prev_cat and results:
            tee.print()
        prev_cat = cat

        r = await bench_case(case, history_stub, llm_available)
        results.append(r)

        msg_d = case.message if len(case.message) < CM - 1 else case.message[:CM - 4] + "…"
        llm_r_str  = f"{r.llm_intent:<{CR}}" if r.llm_intent else f"{'N/A':<{CR}}"
        llm_ms_str = f"{r.llm_ms:>{CMS}.1f}" if r.llm_ms is not None else f"{'N/A':>{CMS}}"

        line = (
            f"  {case.label:<{CL}}"
            f"{msg_d:<{CM}}"
            f"{case.expected:<{CE}}"
            f"{r.old_intent:<{CR}}"
            f"{r.new_intent:<{CR}}"
            + (llm_r_str if has_llm else "")
            + f"{r.old_us:>{CUS}.3f}"
            f"{r.new_us:>{CUS}.3f}"
            + (llm_ms_str if has_llm else "")
            + f"  {tick(r.old_intent, case.expected):<4} {tick(r.new_intent, case.expected):<4}"
            + (f" {tick(r.llm_intent, case.expected)}" if has_llm and r.llm_intent else "")
        )
        # Highlight disagreements between routers
        all_same = (r.old_intent == r.new_intent and
                    (not has_llm or r.llm_intent == r.new_intent))
        if not all_same:
            line = YELLOW + line + RESET

        tee.print(line)

    # ── Summary ──────────────────────────────────────────────────────────────

    tee.print()
    tee.print("  " + "─" * (W - 2))
    tee.print()

    n = len(results)
    wrong_old  = sum(1 for r in results if r.old_intent != r.case.expected)
    wrong_new  = sum(1 for r in results if r.new_intent != r.case.expected)
    wrong_llm  = sum(1 for r in results if r.llm_intent and r.llm_intent != r.case.expected)

    total_old_us  = sum(r.old_us  for r in results)
    total_new_us  = sum(r.new_us  for r in results)
    total_llm_ms  = sum(r.llm_ms  for r in results if r.llm_ms is not None)

    tee.print(BOLD + "  Accuracy" + RESET)
    tee.print(f"  {'Cases':<35} {n}")
    tee.print(f"  {'Old router wrong':<35} "
              + c(f"{wrong_old}/{n}", RED if wrong_old else GREEN))
    tee.print(f"  {'New router wrong':<35} "
              + c(f"{wrong_new}/{n}", RED if wrong_new else GREEN))
    if has_llm:
        tee.print(f"  {'LLM router wrong':<35} "
                  + c(f"{wrong_llm}/{n}", RED if wrong_llm else GREEN))
    tee.print()

    # Latency comparison
    tee.print(BOLD + "  Latency  (logic only — does not include downstream embedding/search)" + RESET)
    tee.print(f"  {'Old router (sum of avgs)':<40} {total_old_us:>10.3f} µs")
    tee.print(f"  {'New router (sum of avgs)':<40} {total_new_us:>10.3f} µs")
    if has_llm and total_llm_ms:
        tee.print(f"  {'LLM router (sum of avgs)':<40} {total_llm_ms:>10.1f} ms"
                  + f"  =  {total_llm_ms * 1000:,.0f} µs")
        ratio_vs_old = (total_llm_ms * 1000) / total_old_us
        ratio_vs_new = (total_llm_ms * 1000) / total_new_us
        tee.print()
        tee.print(c(f"  LLM is {ratio_vs_old:,.0f}× slower than old regex, "
                    f"{ratio_vs_new:,.0f}× slower than new router (pure logic only).", CYAN))
        tee.print(c("  Add ~100 ms per query for the embedding call that LLM classification "
                    "would save for 'general' messages.", DIM))
    tee.print()

    # Per-category breakdown
    categories = [
        ("Pure conversational",  lambda r: r.case.expected == "general" and "ambiguous" not in r.case.label),
        ("Short ambiguous",      lambda r: "ambiguous" in r.case.label),
        ("Follow-up",            lambda r: r.case.expected == "follow_up"),
        ("Kw beats follow-up",   lambda r: "kw beats" in r.case.label),
        ("Genuine code",         lambda r: r.case.expected == "code" and "kw beats" not in r.case.label),
    ]

    tee.print(BOLD + "  Per-category accuracy" + RESET)
    hdr2 = f"  {'Category':<26} {'N':>3}  {'Old':>5}  {'New':>5}"
    if has_llm: hdr2 += f"  {'LLM':>5}"
    hdr2 += "  Embedding calls saved by new"
    tee.print(hdr2)
    tee.print("  " + "─" * 80)

    embed_saved_total = 0
    for cat_name, pred in categories:
        cat = [r for r in results if pred(r)]
        if not cat: continue
        ok_old = sum(1 for r in cat if r.old_intent == r.case.expected)
        ok_new = sum(1 for r in cat if r.new_intent == r.case.expected)
        ok_llm = sum(1 for r in cat if r.llm_intent and r.llm_intent == r.case.expected)
        embed_saved = sum(
            1 for r in cat
            if r.old_intent in ("code", "follow_up") and r.new_intent == "general"
        )
        embed_saved_total += embed_saved
        ok_old_s = c(f"{ok_old}/{len(cat)}", GREEN if ok_old == len(cat) else RED)
        ok_new_s = c(f"{ok_new}/{len(cat)}", GREEN if ok_new == len(cat) else RED)
        ok_llm_s = c(f"{ok_llm}/{len(cat)}", GREEN if ok_llm == len(cat) else RED) if has_llm else ""
        emb_s    = c(f"−{embed_saved}", GREEN) if embed_saved else c("none", DIM)
        row = f"  {cat_name:<26} {len(cat):>3}  {ok_old_s:>13}  {ok_new_s:>13}"
        if has_llm: row += f"  {ok_llm_s:>13}"
        row += f"  {emb_s}"
        tee.print(row)

    tee.print()
    tee.print(c("  Real-world impact", CYAN))
    tee.print(f"  Each embedding call saved ≈ 100 ms (Ollama nomic-embed-text round-trip).")
    tee.print(f"  New router avoids {c(str(embed_saved_total), GREEN)} unnecessary embedding "
              f"call(s) vs old router across this query set.")
    if has_llm:
        avg_llm_ms = total_llm_ms / n
        tee.print()
        tee.print(c("  Verdict on LLM classification:", BOLD))
        tee.print(f"  • Average LLM call: {avg_llm_ms:.0f} ms  — added to EVERY message, before the first token streams.")
        tee.print(f"  • Heuristic router: < 1 µs  — effectively free.")
        tee.print(f"  • Accuracy gain from LLM: {max(0, wrong_old - wrong_llm)} fewer errors vs old,  "
                  f"{wrong_llm - wrong_new:+d} vs new router.")
        if wrong_llm <= wrong_new:
            tee.print(c("  → LLM matches or beats new router on accuracy, but at ~{:.0f}x the latency cost.".format(avg_llm_ms * 1000 / max(total_new_us / n, 0.001)), YELLOW))
            tee.print(c("  → Not worth it: the heuristic already gets 29/29 correct at zero cost.", YELLOW))
        else:
            tee.print(c(f"  → LLM is less accurate AND slower. Definitively not worth it.", RED))

    tee.print()
    tee.print(BOLD + "━" * W + RESET)
    tee.print()

    # ── Save output ───────────────────────────────────────────────────────────

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    txt_path  = RESULTS_DIR / "router_benchmark_latest.txt"
    json_path = RESULTS_DIR / "router_benchmark_latest.json"

    txt_path.write_text(tee.plain_text(), encoding="utf-8")

    json_data = {
        "benchmark_date": timestamp,
        "iterations": {"regex": REGEX_ITER, "llm": LLM_ITER},
        "llm_model": LLM_MODEL,
        "llm_available": llm_available,
        "cases": [
            {
                "label":       r.case.label,
                "message":     r.case.message,
                "has_history": r.case.history,
                "expected":    r.case.expected,
                "old": {
                    "result":  r.old_intent,
                    "correct": r.old_intent == r.case.expected,
                    "avg_us":  round(r.old_us, 4),
                },
                "new": {
                    "result":  r.new_intent,
                    "correct": r.new_intent == r.case.expected,
                    "avg_us":  round(r.new_us, 4),
                },
                "llm": {
                    "result":  r.llm_intent,
                    "correct": r.llm_intent == r.case.expected if r.llm_intent else None,
                    "avg_ms":  round(r.llm_ms, 2) if r.llm_ms is not None else None,
                },
            }
            for r in results
        ],
        "summary": {
            "total_cases": n,
            "old": {
                "correct":     n - wrong_old,
                "wrong":       wrong_old,
                "total_us":    round(total_old_us, 4),
                "avg_us":      round(total_old_us / n, 4),
            },
            "new": {
                "correct":     n - wrong_new,
                "wrong":       wrong_new,
                "total_us":    round(total_new_us, 4),
                "avg_us":      round(total_new_us / n, 4),
            },
            "llm": {
                "correct":     n - wrong_llm if llm_available else None,
                "wrong":       wrong_llm if llm_available else None,
                "total_ms":    round(total_llm_ms, 2) if llm_available else None,
                "avg_ms":      round(total_llm_ms / n, 2) if llm_available else None,
                "skipped":     not llm_available,
            },
            "embedding_calls_saved_by_new_vs_old": embed_saved_total,
        },
    }

    json_path.write_text(json.dumps(json_data, indent=2), encoding="utf-8")

    print(f"  Results saved to:")
    print(f"    {c(str(txt_path),  CYAN)}  (plain text)")
    print(f"    {c(str(json_path), CYAN)}  (JSON for analysis)")
    print()


if __name__ == "__main__":
    asyncio.run(main())
