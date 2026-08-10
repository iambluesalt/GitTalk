# GitTalk → LangGraph/LangChain Migration Plan

Branch: `langgraph-migration`. Persisted to disk (not just chat) so it survives context resets.

## Goal
Replace hand-rolled RAG pipeline with a LangGraph agent that can loop on tools (multi-hop search, file read) instead of single-pass retrieve→answer. Model backends: **Groq** (cloud, fast prototyping) + **Ollama** (local chat + embeddings) only — no generic OpenAI-compat path needed.

## New dependencies
```
langgraph
langgraph-checkpoint-sqlite
langchain-core
langchain-groq
langchain-ollama
langchain-community   # LanceDB vectorstore, EnsembleRetriever, BM25Retriever
```
Check compat against pinned `fastapi==0.141.1` / `pydantic==2.13.4` (app/requirements.txt) before installing — LC pydantic v2 support should be fine but verify.

## Phase 0 — Housekeeping (do first, low risk)
- Remove `bash.exe.stackdump` (crash artifact, junk).
- Decide fate of `COMMANDS.txt`, `PIPELINE_AUDIT.md`, `PROJECT_STATE.md` (legit but untracked scratch docs — not noise, just unfiled). Move to `docs/` or commit as-is.
- Confirm `app/venv`, `build/`, `.react-router/`, `data/`, `cloned_repos/`, `node_modules/` all gitignored (verified — they are).

## Phase 1 — Model layer (replaces `services/llm_service.py`, 315 lines)
- `ChatGroq` + `ChatOllama` wrappers behind existing `LLM_PROVIDER` config knob (`ollama|cloud|hybrid`, cloud = groq now).
- Hybrid fallback (`hybrid` mode) → `RunnableWithFallbacks` (Groq primary, Ollama fallback) instead of manual try/except.
- Keep `PROMPT_TEMPLATES` (code_qa/general) as LC `ChatPromptTemplate`s.
- Title-generation one-off call stays, just via `ChatOllama.invoke()`.

## Phase 2 — Embeddings (replaces `services/embedding_service.py`, 152 lines)
- `OllamaEmbeddings` (nomic-embed-text). Confirm it supports the `search_document:`/`search_query:` prefix injection Nomic needs — if not, thin subclass wrapping `OllamaEmbeddings.embed_documents/embed_query` to inject prefixes. Keep manual batching only if `OllamaEmbeddings` doesn't already batch.

## Phase 3 — Retrieval (replaces most of `services/search_service.py`, 225 lines)
- Wrap `storage/vector_db.py` LanceDB tables with LC's `LanceDB` vectorstore.
- `BM25Retriever` over the same corpus (FTS index already exists in LanceDB — evaluate reusing it vs LC's in-memory BM25).
- `EnsembleRetriever([vector_retriever, bm25_retriever], weights=[...])` replaces the hand-rolled `_rrf_merge`.
- Custom regex language-prefilter + "retry unfiltered if empty" logic **stays custom** — wrap as a pre/post filter step around the ensemble retriever, or as a dedicated `@tool`.

## Phase 4 — Memory/state (replaces conversation-history portion of `storage/metadata_db.py`, ~150 of 663 lines)
- `SqliteSaver` (langgraph-checkpoint-sqlite) checkpointer, thread_id = conversation_id.
- Keep `metadata_db.py` for projects/index/files (unrelated to chat memory) — no change there.

## Phase 5 — Tools
Define as `@tool`:
- `search_codebase(query)` → wraps Phase 3 ensemble retriever + language prefilter
- `read_file(path)` → reads a chunk/file from the cloned repo
- `list_files(dir)` → repo tree listing
- (stretch) `grep_repo(pattern)` for exact-match lookups the retriever might miss

## Phase 6 — Agent graph (replaces `services/rag_service.py` [337 lines] + orchestration in `routes/chat.py`)
- `StateGraph`: `agent` node (model with `.bind_tools()`) → conditional edge → `tools` node → loop back to `agent` → `END` when no tool calls.
- System prompt assembly replaces manual context/token-budget trimming in `rag_service.py`; use `trim_messages` if needed for history windowing.
- Checkpointer from Phase 4 attached to the graph for cross-turn memory.

## Phase 7 — Route integration (`routes/chat.py`, ~150 of 220 lines rewritten)
- Replace manual pipeline with `graph.astream_events(...)`.
- Translate LG event stream → existing SSE event schema (`source`, token deltas, `done`) so **frontend (`app/routes/chat.tsx`, `app/lib/types.ts`) needs zero/minimal changes**.
- Keep project validation, message persistence (user msg still logged to `metadata_db` for the project's message list UI), title-gen call as-is.

## Phase 8 — Testing
- Existing test suite (added in commit `1627e18`) covers RAG/router/search — port/rewrite against new graph, don't just delete.
- Add: tool unit tests, agent-loop integration test (multi-hop question → verify 2+ tool calls), checkpointer persistence test.

## Phase 9 — Cutover strategy
- Add `AGENT_BACKEND=legacy|langgraph` config flag; run both paths side-by-side behind the flag during validation instead of a hard swap.
- Once langgraph path is validated in real usage, remove legacy code: old `llm_service.py` streaming/routing, `rag_service.py`, RRF merge in `search_service.py`, conversation-history CRUD in `metadata_db.py`.
- Do not merge to `main` until flag defaults to `langgraph` and legacy path is deleted.

## Estimated LOC impact
~1035 lines deleted (llm_service, search_service RRF, embedding_service, rag_service, chat.py orchestration, metadata_db conversation CRUD) vs. ~300 added (tools, graph def, checkpointer wiring) → **net ~700-750 LOC eliminated**. See Phase-by-phase breakdown above for per-file detail.

## Explicit non-goals
- No change to `clone_service.py`, `routes/clone.py`, tree-sitter chunking, `vector_db.py` table lifecycle mgmt — unrelated to chat/RAG.
- No generic OpenAI-compat provider — Groq + Ollama only, per current decision.
