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

## Phase 0 — Housekeeping — ✅ DONE (commit `1dd22ad`)
- `git rm requirements.txt` and `git add app/requirements.txt` — this move is already in progress (working tree shows root file deleted, `app/requirements.txt` untracked with identical content), just needs finalizing as a commit.
- No `bash.exe.stackdump`, no stray `COMMANDS.txt`/`PIPELINE_AUDIT.md`/`PROJECT_STATE.md` found in the repo — nothing to do there (v1 assumed these existed; confirmed they don't).
- `.gitignore` already covers `venv`, `build/`, `.react-router/`, `data/`, `cloned_repos/`, `node_modules/` — confirmed, no change needed.

## Phase 1 — Model layer (`app/backend/services/llm_service.py`) — ✅ DONE
- **Config names kept as `CLOUD_*` (user decision), not renamed to `GROQ_*`.** `.env` already points the generic cloud path at Groq (`https://api.groq.com/openai/v1`, `openai/gpt-oss-120b`), and the settings UI exposes a full 4-field cloud form (`settings.tsx:594-629`, `types.ts:137-140,162-165`, `models.py:57-60`, `errors.ts:41`). Renaming would have forced a frontend rewrite for zero functional gain. `LLM_PROVIDER: Literal["ollama","cloud","hybrid"]` unchanged; `cloud` means Groq.
- `_groq_base_url()` normalises `CLOUD_API_BASE_URL` by stripping a trailing `/openai/v1` — the groq SDK's own default base is `https://api.groq.com` and it appends `/openai/v1` to every path, so passing the stored value through raw would double the path.
- `get_chat_model(model_override)` is the new public seam returning a `BaseChatModel` (Phase 6 needs it for `.bind_tools()`); `stream_chat()` keeps its old signature and is now a thin wrapper over `model.astream()`.
- **Open question resolved:** LC's `RunnableWithFallbacks.astream` pulls the *first* chunk inside its `try` block, so it falls back only when the primary fails before yielding anything and re-raises on mid-stream failure — exactly the old `yield_started` semantics. Verified in source and pinned by two tests.
- `PROMPT_TEMPLATES`/`get_prompt_template(name, project_name)` unchanged. **Deviation:** not backed by `ChatPromptTemplate` — `.format()` on a `ChatPromptTemplate` renders a role-prefixed message string, which is wrong for a function that must return a raw system-prompt string. Plain `str.format()` stays.
- `generate_title()` → `ChatOllama(OLLAMA_FAST_MODEL, temperature=0.3, num_predict=20, timeout=10s).ainvoke()`, same truncation fallback. Now hits `/api/chat` instead of `/api/generate`.
- `list_models()`/`check_availability()` kept as direct httpx `/api/tags` calls.
- Raw-httpx `_stream_ollama`/`_stream_cloud` **deleted now** rather than at Phase 9 — Phase 1 replaces the model layer wholesale, so Phase 9's llm_service deletion is already done.
- `tests/test_llm.py` rewritten (36 tests): transport parsing is LC's problem now; tests cover override parsing, base-URL normalisation, provider routing, chunk-text extraction, fallback semantics, introspection helpers.
- Verified live: streaming through `ChatGroq` against the configured Groq endpoint returns tokens correctly.

## Phase 2 — Embeddings (`app/backend/services/embedding_service.py`) — ✅ DONE
- `PrefixedOllamaEmbeddings(OllamaEmbeddings)` subclass injects per-task prefixes and truncates at `MAX_EMBED_CHARS` (both confirmed absent from LC). `embed_query` calls `super().embed_documents` directly so the query prefix isn't overwritten by our own document prefix.
- **Prefixes are gated on the embed model**, not applied unconditionally as the old code did: `resolve_task_prefixes(model)` looks the model family up in `TASK_PREFIXES` (`nomic-embed*` → `search_document: `/`search_query: `) and returns empty strings otherwise. `ollama_embed_model` is user-editable in the settings UI, and prefixing a model that doesn't expect one silently corrupts its vectors with no error. Current `.env` (`nomic-embed-text:v1.5`) resolves to the same prefixes as before — no reindex needed.
- `MAX_EMBED_CHARS` stays unconditional: it's a safety ceiling, harmless for models with a larger context.
- `OLLAMA_FAST_MODEL` (title generation) added explicitly to `.env` — it was relying on the `config.py` default, so an unpulled `llama3.2:1b` meant every title silently degraded to a truncated user message. Still absent from `_config_dict()`/`ConfigUpdate`, so it is not visible in the settings UI (unchanged, out of migration scope).
- **Open question resolved:** `OllamaEmbeddings.aembed_documents` sends the entire list to `/api/embed` in a single request with no chunking — so manual `EMBEDDING_BATCH_SIZE` batching is **kept** in `EmbeddingService.embed_texts`, along with the per-item retry and zero-vector fallback.
- `EmbeddingService` keeps its public API (`embed_texts`, `embed_single`, `is_available`) — call sites at `search_service.py:36` and `indexing_service.py:317-320` unchanged.
- New `tests/test_embedding.py` (10 tests) covers prefixes, truncation, batch splitting, and both failure paths. Not live-verified: Ollama was not running locally at the time.

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
- **Known-bad baseline (pre-existing at commit `1d161b0`, unrelated to this migration — do not attribute to migration work):**
  - `tests/test_chunker.py::test_large_function_chunks_stay_within_token_limit` hard-crashes the interpreter (tree-sitter native fault in `chunker_service.py:153`) — run the suite with `--ignore=tests/test_chunker.py` to get results.
  - 17 failures: 16 in `tests/test_router.py` (classifier returns `general` where tests expect `code`) + `tests/test_api.py::test_chat_streams_tokens` (fixture mocks `generate_title` with a non-async `MagicMock`). Confirmed identical before and after Phases 1-2.
- Manual: start backend (`uvicorn` per existing dev workflow), run a known-multi-hop question against an indexed project, confirm ≥2 tool calls in logs and correct final answer; confirm SSE `sources`/`token`/`done` events match frontend expectations by exercising the chat UI in `app/routes/chat.tsx`.
- Confirm `AGENT_BACKEND=legacy` path still works unchanged before flipping default (Phase 9 safety net).

## Execution order / check-ins
Implement phases 0-2 (housekeeping, model layer, embeddings) together, then check in. Then phases 5-7 (tools, graph, routes) together — this is the core risk area — then check in with a working end-to-end demo before Phase 8 (tests) and Phase 9 (cutover/deletion of legacy code).

## Explicit non-goals (unchanged from v1)
- No change to `clone_service.py`, `routes/clone.py`, tree-sitter chunking, `vector_db.py` table lifecycle mgmt — unrelated to chat/RAG.
- No generic OpenAI-compat provider — Groq + Ollama only, per current decision.
