# GitTalk → LangGraph Migration Plan v1.2

Branch: `langgraph-migration`. Supersedes `LANGGRAPH_MIGRATION_PLAN.md` (v1) — this version reflects actual codebase findings from a full exploration pass plus decisions made with the user. Persisted to disk so it survives context resets.

## Context
v1 laid out the intent: replace the hand-rolled single-pass RAG pipeline with a LangGraph tool-calling agent (multi-hop search/read instead of retrieve-once-then-answer), on Groq (cloud) + Ollama (local) only. A full codebase exploration (3 parallel agents covering model/embedding layer, retrieval/vector storage, and RAG orchestration/routes/memory) confirmed most of v1 but found several places where actual code differs from v1's assumptions — captured below as **deviations**, each with rationale. Two open design questions were resolved with the user:
- Keep `metadata_db` `messages`/`conversations` tables as the source of truth for chat history/UI (sidebar, load past chat, title). LangGraph memory is not used for cross-turn persistence.
- Keep `router_service.classify()` heuristic as a cheap pre-agent gate (skips the model entirely for greetings/small talk).

## Deviations from v1 (found during exploration)
1. **No Groq today.** Current "cloud" provider is a generic OpenAI-compatible HTTP client (`CLOUD_API_KEY`/`CLOUD_API_BASE_URL`/`CLOUD_MODEL`), not Groq-branded. Will introduce real `GROQ_*` config and `langchain_groq.ChatGroq`.
2. **search_service.py has no regex language-prefilter and no Python BM25** (contra v1 Phase 3). It's 167 lines (not 225 as v1 estimated), already does vector search + LanceDB-native FTS + a small hand-rolled RRF merge, with a clean fallback to vector-only. It's small, correct, and already decoupled from `vector_db.py` internals. **Decision: do not force it through `langchain_community`'s `LanceDB`/`BM25Retriever`/`EnsembleRetriever` abstractions** — there's no prefilter logic to preserve, and rewriting a working 167-line RRF into LC retriever classes adds risk for no behavior change. Instead, expose `search_service.hybrid_search` directly as a `@tool` in Phase 5. This drops `langchain_community` from the dependency list.
3. **No SqliteSaver / checkpointer** (contra v1 Phase 4). Since chat history stays in `metadata_db` (per user decision) and gets rebuilt into the initial message list every request (exactly like `rag_service.build_context` does today), a LangGraph checkpointer isn't needed for correctness — the multi-hop tool loop only needs state to survive within a single `graph.astream_events()` call, which LangGraph does in-memory automatically. Drops `langgraph-checkpoint-sqlite` from dependencies.
4. Net LOC removed will be less than v1's ~700-750 estimate, since `metadata_db.py` conversation CRUD (~172 lines) and `search_service.py`'s RRF (~89 lines) are being kept, not deleted.

## New dependencies (revised)
```
langgraph
langchain-core
langchain-groq
langchain-ollama
```

## Phase 0 — Housekeeping
- `git rm requirements.txt` and `git add app/requirements.txt` — this move is already in progress (working tree shows root file deleted, `app/requirements.txt` untracked with identical content), just needs finalizing as a commit.
- No `bash.exe.stackdump`, no stray `COMMANDS.txt`/`PIPELINE_AUDIT.md`/`PROJECT_STATE.md` found in the repo — nothing to do there (v1 assumed these existed; confirmed they don't).
- `.gitignore` already covers `venv`, `build/`, `.react-router/`, `data/`, `cloned_repos/`, `node_modules/` — confirmed, no change needed.

## Phase 1 — Model layer (`app/backend/services/llm_service.py`, 313 lines)
- `app/backend/config.py`: replace generic `CLOUD_API_PROVIDER`/`CLOUD_API_BASE_URL`/`CLOUD_API_KEY`/`CLOUD_MODEL` with `GROQ_API_KEY`, `GROQ_MODEL` (base URL is fixed inside `ChatGroq`). Keep `LLM_PROVIDER: Literal["ollama","cloud","hybrid"]` knob name as-is (v1's explicit choice — `cloud` means Groq).
- `ChatGroq(model=settings.GROQ_MODEL, api_key=settings.GROQ_API_KEY, streaming=True)` and `ChatOllama(model=settings.OLLAMA_MODEL, base_url=settings.OLLAMA_BASE_URL)`.
- Hybrid mode: `primary.with_fallbacks([fallback])`. Verify during implementation that LC's fallback-on-stream semantics match today's "only fallback if no token has been yielded yet, else propagate" behavior (`llm_service.py:98-116`) — LC's `with_fallbacks` triggers on exception raised during stream setup/early iteration; confirm with a quick manual test since a generator can't retract already-yielded chunks either way.
- `PROMPT_TEMPLATES`/`get_prompt_template(name, project_name)` (`llm_service.py:18-43`): keep the exact same function signature (thin wrapper backed by `ChatPromptTemplate.from_template(...).format(...)`) — call sites in `rag_service.py:235,237,243` (moving into the graph's system-prompt builder in Phase 6) don't need to change.
- `generate_title()` (`llm_service.py:118`): `ChatOllama(model=settings.OLLAMA_FAST_MODEL).ainvoke(...)`, same truncation fallback on error/empty result.
- `list_models()`/`check_availability()` (`llm_service.py:155,181`): keep as-is (direct Ollama `/api/tags` calls) — these are introspection/UI-picker helpers, not chat calls, no LC equivalent needed.
- Update `app/backend/tests/test_llm.py` alongside.

## Phase 2 — Embeddings (`app/backend/services/embedding_service.py`, 131 lines)
- `OllamaEmbeddings(model=settings.OLLAMA_EMBED_MODEL, base_url=settings.OLLAMA_BASE_URL)`.
- `langchain_ollama.OllamaEmbeddings` does not inject Nomic's `search_document:`/`search_query:` prefixes — subclass it, override `embed_documents`/`embed_query` to prepend the prefix (mirrors `embedding_service.py:50,73`) before calling `super()`.
- Keep `MAX_EMBED_CHARS` truncation (`embedding_service.py:15,28`) — LC doesn't do this, and nomic-embed-text needs it (8192-token ctx).
- Keep the per-item zero-vector fallback on batch failure (`embedding_service.py:34-68`) — LC will just raise; wrap it.
- Verify during implementation whether `OllamaEmbeddings.embed_documents` already batches via Ollama's `/api/embed` list input; keep manual `EMBEDDING_BATCH_SIZE` chunking only if it doesn't (v1's own hedge, confirmed still open).
- Call sites unaffected: `search_service.py:36` (`embed_single`→`embed_query`), `indexing_service.py:317-320` (`embed_texts`→`embed_documents`).

## Phase 3 — Retrieval
**No rewrite** — see Deviation #2. `search_service.hybrid_search` and `vector_db.py` stay exactly as-is. They get wrapped as a tool in Phase 5, not restructured.

## Phase 4 — Memory/state
**No checkpointer** — see Deviation #3. `metadata_db.py` conversation CRUD (`create_conversation`, `add_message`, `get_conversation_history`, etc., lines 544-712) stays unchanged, used exactly as `rag_service.build_context` uses it today to build the initial message list per request.

## Phase 5 — Tools (new file `app/backend/services/agent_tools.py`)
- `make_tools(project_id: str) -> list` factory (tools close over `project_id` per-request — it must not be an LLM-visible parameter):
  - `search_codebase(query: str) -> str` — `@tool` wrapping `search_service.hybrid_search(project_id, query)`, formats results as text the model can read. The raw `SearchResult`s also need to reach the SSE `sources` event (Phase 7) — return `(content, artifact)` via `response_format="content_and_artifact"` so the formatted text goes to the model and raw results are recoverable from the `ToolMessage.artifact` for event translation.
  - `read_file(path: str, line_start: int | None, line_end: int | None) -> str` — reads from `project.clone_path` (`models.py:121`) on disk, path-traversal guarded (must resolve within `clone_path`).
  - `list_files(dir: str = "") -> str` — check `repomap_service.py` first for existing tree-listing logic to reuse before writing new code.
  - Skip `grep_repo` (v1 marks it stretch/optional) unless requested later.

## Phase 6 — Agent graph (new file `app/backend/services/agent_graph.py`, replaces `rag_service.py`'s retrieval-injection role)
- `StateGraph` over a `TypedDict`/`MessagesState`-based state. Nodes: `agent` (model with `.bind_tools(tools)`) → conditional edge (`tools_condition`) → `tools` node (`ToolNode`) → loop back to `agent` → `END`.
- System prompt assembly: port `rag_service._build_system_message`'s repo-map truncation and `_build_history_messages`'s token-budget history windowing (`rag_service.py:225-276`) into the graph's setup step (run once before invoking the graph, not a node) — same token-budget math, same `settings.MAX_CONTEXT_TOKENS` split.
- The old `_build_user_message`'s manual "## Relevant Code" chunk injection (`rag_service.py:278-323`) is **replaced by tool-calling itself** — the agent now asks for code via `search_codebase` and gets it back as a `ToolMessage`, instead of code being pre-injected before the first model call. This is the actual point of the migration (multi-hop) and a real simplification, not just a port.
- `router_service.classify()` pre-filter (kept per user decision): call before building the graph state. If `"general"` → skip the graph/tools entirely, call the chat model directly with the `general` prompt template (mirrors today's behavior exactly). If `"code"`/`"follow_up"` → run the full tool-calling graph with the `code_qa` template + repo map in the system message.

## Phase 7 — Route integration (`app/backend/routes/chat.py`, 219 lines)
- Replace the `rag_service.build_context` + `llm_service.stream_chat` sequence with: build initial state (system + history, per Phase 6) → `graph.astream_events(state, version="v2")`.
- Event translation to the **existing SSE schema** (`models.py:210-218`, confirmed exactly matched by frontend `chat.tsx`/`api.ts`/`types.ts` — zero frontend changes):
  - Accumulate tool results from `on_tool_end` events across the run; once the graph reaches its final `on_chat_model_stream` sequence (no further tool calls pending), emit one `sources` event with the accumulated `CodeReference[]` — same shape as today, just potentially built from multiple tool calls instead of one retrieval.
  - `on_chat_model_stream` chunks → `token` events, same as today.
  - `done` / `error` events unchanged: still carry `conversation_id`, `response_time_ms`, optional `conversation_title`.
- Unchanged: project validation, `db.add_message` persistence (user + assistant), `generate_title` call for new conversations, conversation CRUD routes (`list_conversations`, `get_conversation`, `delete_conversation`) — these don't touch the agent at all.

## Phase 8 — Testing
- Port `test_rag.py` (intent-gated retrieval, template selection, history/repo-map budget truncation, dedup) and `test_router.py` (unchanged, router isn't touched) against the new structure.
- `test_search.py` unchanged (search_service isn't touched — Deviation #2).
- Add: tool unit tests (`agent_tools.py`), an agent-loop integration test asserting a multi-hop question produces 2+ `search_codebase` calls, and an SSE event-translation test (tool results → one `sources` event, correct `CodeReference` shape).

## Phase 9 — Cutover
- `AGENT_BACKEND: Literal["legacy","langgraph"]` config flag; `routes/chat.py` branches between old (`rag_service`+`llm_service.stream_chat`) and new (`agent_graph`) paths behind it.
- Once validated in real usage with flag defaulted to `langgraph`: delete `rag_service.py`'s retrieval-injection code and `llm_service.py`'s old raw-httpx streaming/routing internals (`_stream_ollama`, `_stream_cloud`, hybrid try/except). **Keep**: `metadata_db.py` conversation CRUD, `search_service.py`, `vector_db.py` (per deviations above — not slated for deletion).
- Do not merge to `main` until the flag defaults to `langgraph` and legacy code is deleted (per original plan).

## Verification
- `pytest app/backend/tests` after each phase (existing suite must keep passing where unchanged; ported tests updated where behavior changed).
- Manual: start backend (`uvicorn` per existing dev workflow), run a known-multi-hop question against an indexed project, confirm ≥2 tool calls in logs and correct final answer; confirm SSE `sources`/`token`/`done` events match frontend expectations by exercising the chat UI in `app/routes/chat.tsx`.
- Confirm `AGENT_BACKEND=legacy` path still works unchanged before flipping default (Phase 9 safety net).

## Execution order / check-ins
Implement phases 0-2 (housekeeping, model layer, embeddings) together, then check in. Then phases 5-7 (tools, graph, routes) together — this is the core risk area — then check in with a working end-to-end demo before Phase 8 (tests) and Phase 9 (cutover/deletion of legacy code).

## Explicit non-goals (unchanged from v1)
- No change to `clone_service.py`, `routes/clone.py`, tree-sitter chunking, `vector_db.py` table lifecycle mgmt — unrelated to chat/RAG.
- No generic OpenAI-compat provider — Groq + Ollama only, per current decision.
