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

## Phase 5 — Tools (`app/backend/services/agent_tools.py`) — ✅ DONE
- `make_tools(project_id) -> list[BaseTool]` factory; tools close over `project_id` and the resolved `clone_path`, so neither is LLM-visible. All three use `parse_docstring=True` so arg descriptions reach the model's tool schema.
  - `search_codebase(query)` — `response_format="content_and_artifact"`: formatted numbered snippets go to the model, raw `SearchResult`s ride along in `ToolMessage.artifact` for Phase 7's `sources` event.
  - `read_file(path, line_start=None, line_end=None)` — 1-indexed inclusive, line-numbered output, `READ_FILE_MAX_LINES`/`READ_FILE_MAX_CHARS` capped. Guards verified live: traversal (`../../.env`, `/etc/passwd`, `dir="../.."`), missing files, out-of-range lines, and `should_exclude` (so `.env`/keys/lockfiles are unreadable even inside the clone).
  - `list_files(dir="")` — **deviation:** nothing in `repomap_service.py` was reusable (it does a full `rglob` + tree-sitter signature extraction, not a directory listing), so this is a small new `iterdir` + `EXCLUDED_DIRS`/`should_exclude` filter.
  - `grep_repo` skipped as planned.
- `dedup_overlapping`/`_merge_results` were **moved here** from `rag_service` (applied per search call) rather than imported, so Phase 9 can delete `rag_service.py` outright.
- Per-call output ceilings (`SEARCH_RESULT_MAX_CHARS` etc.) are new: tool results accumulate across hops, so each call has to stay small enough that a 6-hop run still fits the window.

## Phase 6 — Agent graph (`app/backend/services/agent_graph.py`) — ✅ DONE
- `StateGraph(MessagesState)`: `agent` → `tools_condition` → `ToolNode` → back to `agent` → `END`. Compiled per request.
- **The agent node streams (`model.astream()` + chunk summation) instead of `ainvoke`** — with `ainvoke` LangChain emits no `on_chat_model_stream` events, so the route would have had nothing to turn into `token` events. Chunk addition merges partial tool-call arguments back into one `AIMessage`.
- Setup (not a node): `router_service.classify()` gate → repo-map/history token budgets → `SystemMessage` + windowed history + `HumanMessage`. `"general"` returns the bare chat model as the runnable (no graph, no tools); `"code"`/`"follow_up"` return the compiled graph. Both are driven identically by the route via `astream_events`.
- **New `code_agent` prompt template** in `llm_service.PROMPT_TEMPLATES`: `code_qa` tells the model to answer "using ONLY the provided code context below … under '## Relevant Code'", which is false in a tool-calling world. `code_qa` is left untouched for the legacy path.
- Budgets changed shape: there is no up-front chunk budget any more, so `code` reserves only repo-map 18% / history 25% and leaves the rest of the window for tool results (`follow_up` 10%/40%, `general` 15%/75%).
- **Bug fixed in passing:** the route persists the user message *before* context assembly, so `get_conversation_history` returns it as the last turn and legacy sent the question twice. The agent path drops that trailing duplicate.
- `llm_service.get_chat_model` gained a `tools=` kwarg: `RunnableWithFallbacks` has no `bind_tools`, so tools must be bound to each provider *before* fallbacks are composed or hybrid mode breaks.

## Phase 7 — Route integration (`app/backend/routes/chat.py`) — ✅ DONE
- Both pipelines are async generators of `(event_name, data)` pairs — `_run_legacy` and `_run_agent` — selected by `settings.AGENT_BACKEND`. The endpoint body just formats them as SSE and accumulates the response text for persistence. **Deviation:** the `AGENT_BACKEND` flag was pulled forward from Phase 9 so the legacy path stays exercisable during the demo instead of being deleted first; it defaults to `langgraph`.
- Event translation to the **existing SSE schema** — zero frontend changes:
  - `on_tool_end` (name `search_codebase`) → artifact `SearchResult`s → `CodeReference`s, deduped by `file:line_start-line_end` across the whole run.
  - **Deviation:** sources are emitted *cumulatively*, not once. One event fires immediately before the first token (empty if nothing was searched — the frontend only clears `isSearching` on a `sources` event, so `general` turns must still get one), and again whenever a later hop finds new code. Each event carries the full list, which the client already replaces wholesale.
  - `on_chat_model_stream` → `token`. Text the model emits alongside a tool call streams through rather than being buffered; withholding it until the turn ends would stall the final answer.
  - `done`/`error` unchanged.
- Unchanged: project validation, `db.add_message` persistence, `generate_title`, conversation CRUD routes.
- Verified live against Groq (`openai/gpt-oss-120b`) with real files and a stubbed index: one question produced `list_files` → `search_codebase` → `read_file` and a correctly cited answer.

## Phase 8 — Testing — ✅ DONE
- `test_rag.py`/`test_router.py`/`test_search.py` needed no porting — `rag_service.py` and `search_service.py` are untouched (Deviation #2, and legacy path still lives until Phase 9 deletes it), so their existing tests already pass unchanged.
- Tool unit tests: `tests/test_agent_tools.py` (already present, 30 tests — `search_codebase`/`read_file`/`list_files` guards, dedup moved from `rag_service`).
- **New `tests/test_agent_graph.py`** (6 tests): `build_agent_run` gating (`general` skips the graph and binds no tools; `code` compiles a tool-bound graph with `recursion_limit = AGENT_MAX_TOOL_HOPS * 2 + 1`; missing project falls back to `general`; trailing duplicate user turn is dropped from history) plus two graph-loop integration tests driving the real compiled `StateGraph` with a scripted fake model and real `@tool`-decorated stand-ins: one proves a multi-hop question round-trips through `ToolNode` for 2+ `search_codebase` calls before the final answer, the other proves a no-tool-call response ends the graph after a single agent hop.
- SSE event-translation test: done in Phase 7 (`test_chat_agent_backend_translates_events`), asserts `sources` precedes the first `token` and carries `CodeReference`-shaped data.
- **Full suite after Phase 8** (`pytest tests --ignore=tests/test_chunker.py`): **269 passed, 16 failed (all `test_router.py`, pre-existing baseline, unchanged), 11 skipped.**

## Phase 9 — Cutover — ✅ DONE
- ~~`AGENT_BACKEND` config flag; `routes/chat.py` branches between old and new paths behind it.~~ **Done in Phase 7**, defaulted to `langgraph`.
- **Manual live verification (before deletion, per the safety-net gate below):** started `uvicorn` against the real Groq endpoint, cloned+indexed `karpathy/micrograd`, and drove `/api/chat` directly. A multi-hop prompt produced two distinct `search_codebase` calls (`search_codebase('class Neuron')`, `search_codebase('class Value')`) confirmed in server logs, a correctly cited final answer, and the SSE stream had `sources` before the first `token` and a trailing `done` with `conversation_id`/`conversation_title`. Test project and clone were deleted afterward.
- `llm_service.py`'s old raw-httpx internals were already gone (Phase 1). This phase deleted the retrieval-injection code:
  - `services/rag_service.py` deleted outright (`build_context`, `_dedup_overlapping`/`_merge_results` — already duplicated into `agent_tools.py` in Phase 5 — and the `_build_system_message`/`_build_history_messages`/`_build_user_message` chunk-injection builders).
  - `routes/chat.py`: removed `_run_legacy`, the `rag_service` import, and the `AGENT_BACKEND` branch — `chat()` always calls `_run_agent` now.
  - `AGENT_BACKEND` setting removed from `config.py` and `.env` (only one backend exists, so the flag had nothing left to select).
  - `PROMPT_TEMPLATES["code_qa"]` removed from `llm_service.py` — it was the legacy chunk-injection prompt ("using ONLY the provided code context... under '## Relevant Code'"), unreachable now that `agent_graph.py` only ever requests `"general"` or `"code_agent"`. Fallback default in `get_prompt_template` repointed at `"code_agent"`.
  - `models.RAGContext` removed (unused once `rag_service.py` was gone).
  - Tests: `tests/test_rag.py` deleted (tested the deleted service); `tests/test_api.py::test_chat_streams_tokens` (legacy happy-path) deleted — the agent-backend happy path is already covered by the renamed `test_chat_streams_agent_events`; `tests/test_e2e.py`'s fixture no longer imports/patches `rag_svc`.
  - **Kept, per deviations above**: `metadata_db.py` conversation CRUD, `search_service.py`, `vector_db.py`.

## Verification
- `pytest app/backend/tests` after each phase (existing suite must keep passing where unchanged; ported tests updated where behavior changed).
- **Known-bad baseline (pre-existing at commit `1d161b0`, unrelated to this migration — do not attribute to migration work):**
  - `tests/test_chunker.py::test_large_function_chunks_stay_within_token_limit` hard-crashes the interpreter (tree-sitter native fault in `chunker_service.py:153`) — run with `--ignore=tests/test_chunker.py`.
  - `tests/test_e2e.py` can also hard-crash the interpreter (`Windows fatal exception: access violation`, native pathlib fault inside `utils/exclusions.py:128 should_exclude` ← `analyzer_service.py:111 analyze_repository`) when the live-service skip guard doesn't trigger (i.e. git + Ollama + an embed model + a chat model are all actually available, as they were in this environment). Neither file touched by this migration; the analyzer/exclusions code is unrelated to chat/RAG. Run with `--ignore=tests/test_chunker.py --ignore=tests/test_e2e.py` for a clean automated run.
  - Was 17 failures: 16 in `tests/test_router.py` (classifier returns `general` where tests expect `code`) + `tests/test_api.py::test_chat_streams_tokens` (fixture mocks `generate_title` with a non-async `MagicMock`). The `test_api` one was fixed in Phase 7; **baseline is 16 router failures**, confirmed identical before and after Phases 1-9 (`pytest tests --ignore=tests/test_chunker.py --ignore=tests/test_e2e.py`: 248 passed, 16 failed).
- **Pre-existing, not fixed (out of migration scope):** `metadata_db.get_conversation_messages` is `ORDER BY created_at ASC LIMIT 20`, i.e. it returns the *oldest* 20 messages, so long conversations feed the model their opening turns and drop recent ones. The history windowing on top of it can only pick from what it's given. Worth a separate fix.
- **Manual verification: done** — see Phase 9 above.

## Execution order / check-ins
Implement phases 0-2 (housekeeping, model layer, embeddings) together, then check in. Then phases 5-7 (tools, graph, routes) together — this is the core risk area — then check in with a working end-to-end demo before Phase 8 (tests) and Phase 9 (cutover/deletion of legacy code).

## Explicit non-goals (unchanged from v1)
- No change to `clone_service.py`, `routes/clone.py`, tree-sitter chunking, `vector_db.py` table lifecycle mgmt — unrelated to chat/RAG.
- No generic OpenAI-compat provider — Groq + Ollama only, per current decision.
