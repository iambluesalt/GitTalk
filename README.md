# GitTalk — Chat with any GitHub codebase

> Clone a repo, index it locally, and ask questions about the code in plain English.  
> Everything runs on your machine except the optional cloud LLM call. No code ever leaves your network for indexing or search.

**Pipeline:** GitHub URL → preflight check → shallow clone → AST parsing → hybrid vector + FTS indexing → LangGraph tool-calling agent

---

## Features

- **Privacy-first** — fully local by default; Ollama runs the LLM and embeddings on your hardware
- **Agentic chat** — a [LangGraph](https://www.langchain.com/langgraph) tool-calling agent searches and reads the repo itself, hop by hop, instead of answering from one batch of pre-fetched chunks
- **Hybrid search** — combines vector similarity (LanceDB) and BM25 full-text search, fused with Reciprocal Rank Fusion (RRF), exposed to the agent as a `search_codebase` tool
- **AST-aware chunking** — Tree-sitter parses Python, JavaScript, TypeScript, Go, Java, Rust, C, and C++ into semantic chunks (functions, classes, blocks)
- **Intent-aware routing** — a tiered heuristic router skips the agent (and any embedding calls) entirely for conversational messages, cutting unnecessary latency
- **Streaming everywhere** — clone, indexing, and chat all use Server-Sent Events (SSE) for real-time progress
- **Smart conversation titles** — `llama3.2:1b` generates a concise title after the first message; falls back gracefully if the model isn't available
- **Model switcher** — pick any Ollama model or the configured cloud model per conversation from the chat header
- **Cloud fallback** — [Groq](https://groq.com) via `langchain-groq` when configured; hybrid mode tries Groq first and falls back to Ollama automatically if it fails before streaming starts

---

## Stack

| Layer | Technology |
|---|---|
| Frontend | React 19 · React Router v7 · Vite · Tailwind CSS v4 |
| Backend | FastAPI · Uvicorn · Pydantic v2 |
| Agent | LangGraph (`StateGraph`) · LangChain Core |
| Metadata | SQLite (via `metadata_db.py`) |
| Vectors | LanceDB + PyArrow |
| Parsing | Tree-sitter (8 language grammars) |
| Embeddings | `nomic-embed-text` via `langchain-ollama` |
| LLM (primary) | Any Ollama model (default: `qwen2.5-coder:7b`) via `langchain-ollama` |
| LLM (fast tasks) | `llama3.2:1b` via Ollama (title generation) |
| LLM (cloud) | Groq via `langchain-groq` |

---

## Prerequisites

- **Node.js** ≥ 18
- **Python** ≥ 3.11
- **Ollama** — [ollama.com](https://ollama.com)
- A GitHub personal access token is optional but recommended for private repos and to avoid rate limits

---

## Quick Start

### 1. Install frontend dependencies

```bash
npm install
```

### 2. Set up the Python backend

```bash
# From the project root
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

pip install -r app/requirements.txt
```

### 3. Pull Ollama models

```bash
ollama serve                          # keep this running in a terminal

ollama pull qwen2.5-coder:7b          # main chat model
ollama pull nomic-embed-text          # embeddings
ollama pull llama3.2:1b               # fast model (title generation)
```

### 4. Configure (optional)

Create a `.env` at the project root, or configure everything through the **Settings** page in the UI. The defaults work out of the box with Ollama alone — cloud settings are only needed if you want Groq in the mix.

```env
# ── LLM ──────────────────────────────────────────────────────────────
LLM_PROVIDER=hybrid              # ollama | cloud | hybrid
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5-coder:7b
OLLAMA_FAST_MODEL=llama3.2:1b    # used for quick tasks like title generation
OLLAMA_EMBED_MODEL=nomic-embed-text

# ── Cloud fallback (optional, Groq) ────────────────────────────────────
CLOUD_API_PROVIDER=Groq
CLOUD_API_KEY=your-groq-key-here
CLOUD_API_BASE_URL=https://api.groq.com/openai/v1
CLOUD_MODEL=openai/gpt-oss-120b

# ── GitHub ────────────────────────────────────────────────────────────
GITHUB_TOKEN=ghp_...             # optional; avoids rate limits

# ── Limits ───────────────────────────────────────────────────────────
MAX_REPO_SIZE_MB=500
MAX_CONTEXT_TOKENS=32768
MAX_SEARCH_RESULTS=8
CHUNK_MAX_TOKENS=1000
AGENT_MAX_TOOL_HOPS=6            # max model<->tool round trips per turn
```

### 5. Run

Open two terminals from the project root:

**Terminal 1 — Backend**
```bash
cd app/backend
python main.py
# → http://localhost:8000
```

**Terminal 2 — Frontend**
```bash
npm run dev
# → http://localhost:5173
```

Open [http://localhost:5173](http://localhost:5173) and paste any public GitHub URL to get started.

---

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                            User Query                            │
└────────────────────────────┬──────────────────────────────────────┘
                              │
                     ┌────────▼────────┐
                     │  Intent Router  │  tiered heuristic — no LLM call
                     │ code / general  │  keyword set → regex → default
                     │  / follow_up    │
                     └────────┬────────┘
                              │ general → bare chat model, no tools
                              │ code / follow_up, continue below
                     ┌────────▼────────┐
                     │   Agent Setup   │  system prompt + repo map +
                     │ (build_agent_   │  windowed history, token-
                     │     run)        │  budgeted by intent
                     └────────┬────────┘
                              │
      ┌───────────────────────▼───────────────────────┐
      │                  agent <-> tools loop            │
      │                                                    │
      │   ┌─────────────┐   tool call(s)   ┌─────────────┐ │
      │   │  Agent Node │ ───────────────▶ │  Tool Node  │ │
      │   │ (streams:   │                  │search_code- │ │
      │   │  Groq or    │ ◀─────────────── │  base       │ │
      │   │  Ollama)    │   tool result(s) │read_file    │ │
      │   └─────────────┘                  │list_files   │ │
      │                                     └─────────────┘ │
      │   loops until the model stops requesting tools,      │
      │   capped by AGENT_MAX_TOOL_HOPS                       │
      └───────────────────────┬───────────────────────┘
                              │ no more tool calls
                     ┌────────▼────────┐
                     │   Final Answer  │  streamed token-by-token
                     │  (cited sources)│  → SSE → browser; sources
                     │                 │  emitted cumulatively per hop
                     └────────┬────────┘
                              │ on first message
                     ┌────────▼────────┐
                     │   Title Gen     │  llama3.2:1b (non-streaming)
                     │   (background)  │  runs after done event
                     └─────────────────┘
```

The agent (`services/agent_graph.py`) is a LangGraph `StateGraph`: an `agent` node that streams from the configured chat model, and a `tools` node (`ToolNode`) that executes whatever the model asked for. They loop until the model answers without requesting another tool call, or `AGENT_MAX_TOOL_HOPS` is hit. `general`-intent turns skip the graph entirely and stream straight from the bare chat model.

### Intent classification tiers

The router classifies every query in pure Python — no LLM, no network call:

1. **Length guard** (≤ 3 words, no code keywords) → `general` — skips the agent
2. **General patterns** (greetings, thanks, acknowledgements) → `general`
3. **Code keyword fast-path** (function, class, error, api, …) → `code`
4. **Follow-up patterns** (explain that, tell me more, …) → `follow_up`
5. **Default** → `code` (safe; always lets the agent search)

This avoids the round-trip cost of spinning up the agent for conversational messages.

---

## Project Structure

```
.
├── app/
│   ├── backend/
│   │   ├── config.py              # Pydantic settings — all env vars documented here
│   │   ├── main.py                # FastAPI app entry point
│   │   ├── models.py              # Pydantic request / response models
│   │   ├── logger.py              # Structured logger
│   │   ├── routes/
│   │   │   ├── clone.py           # POST /api/clone  (SSE)
│   │   │   ├── index.py           # POST /api/index  (SSE)
│   │   │   ├── chat.py            # POST /api/chat   (SSE) + conversation CRUD
│   │   │   ├── projects.py        # Project CRUD + index stats + file list
│   │   │   └── system.py          # /api/health, /api/config, /api/models, /api/preflight
│   │   ├── services/
│   │   │   ├── clone_service.py   # git clone + Windows-safe cleanup
│   │   │   ├── analyzer_service.py# Repo language/stats analysis
│   │   │   ├── treesitter_service.py  # AST parsing (8 grammars)
│   │   │   ├── chunker_service.py # Semantic chunking with overlap
│   │   │   ├── embedding_service.py   # LangChain Ollama embeddings, task-prefixed + batched
│   │   │   ├── indexing_service.py    # Full indexing pipeline
│   │   │   ├── repomap_service.py # Repo structure map for LLM context
│   │   │   ├── search_service.py  # Hybrid vector + FTS + RRF (used by search_codebase tool)
│   │   │   ├── agent_tools.py     # search_codebase / read_file / list_files tools
│   │   │   ├── agent_graph.py     # LangGraph StateGraph — the agent<->tools loop
│   │   │   ├── router_service.py  # Intent classification (heuristic)
│   │   │   └── llm_service.py     # LangChain chat models (Groq + Ollama) + title gen
│   │   ├── storage/
│   │   │   ├── metadata_db.py     # SQLite: projects, conversations, messages
│   │   │   └── vector_db.py       # LanceDB: chunk vectors + FTS index
│   │   ├── utils/
│   │   │   ├── exclusions.py      # File exclusion rules + .gitignore parsing
│   │   │   └── validators.py      # Input validation helpers
│   │   └── tests/
│   │       ├── benchmark_router.py    # 3-way router benchmark (regex vs heuristic vs LLM)
│   │       ├── test_agent_tools.py
│   │       ├── test_agent_graph.py
│   │       ├── test_search.py
│   │       ├── test_router.py
│   │       ├── test_api.py
│   │       └── ...
│   ├── requirements.txt           # Python runtime dependencies
│   ├── components/                # Shared React components (ChatMessage, ErrorCard, …)
│   ├── lib/
│   │   ├── api.ts                 # Typed API client + SSE parser
│   │   ├── types.ts               # Shared TypeScript types
│   │   └── utils.ts
│   └── routes/                    # Page components (file-based routing)
│       ├── home.tsx
│       ├── projects.tsx
│       ├── project-detail.tsx
│       ├── clone.tsx
│       ├── chat.tsx
│       ├── settings.tsx
│       └── guide.tsx
├── data/
│   ├── metadata.db                # SQLite database (auto-created)
│   └── lancedb/                   # Vector store (auto-created)
├── cloned_repos/                  # Cloned repositories (auto-created)
├── requirements-dev.txt           # Dev / test dependencies
└── package.json
```

---

## API Reference

### Projects

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/projects` | List all projects with index stats |
| `GET` | `/api/projects/:id` | Get project metadata |
| `DELETE` | `/api/projects/:id` | Delete a project and all its data |
| `DELETE` | `/api/projects` | Delete all projects |
| `GET` | `/api/projects/:id/index-stats` | Indexed file / chunk / function counts |
| `GET` | `/api/projects/:id/files` | List all indexed files |

### Clone & Index

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/preflight?url=...` | Fetch repo metadata before cloning (size, language, stars) |
| `POST` | `/api/clone` | Clone a GitHub repo — streams SSE progress events |
| `POST` | `/api/index` | Index a cloned project — streams SSE progress events |

### Chat

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/chat` | Send a message — streams `sources → token… → done` SSE events |
| `GET` | `/api/conversations/:project_id` | List conversations for a project |
| `GET` | `/api/conversations/:project_id/:id` | Get a conversation with all messages |
| `DELETE` | `/api/conversations/:project_id/:id` | Delete a conversation |

### System

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Service health — Ollama / cloud API status |
| `GET` | `/api/models` | List available chat models from all providers |
| `GET` | `/api/config` | Read current configuration |
| `PUT` | `/api/config` | Update configuration (persists to `.env`) |

#### Chat SSE event types

```
event: sources   data: { sources: [...], search_results_count, token_count }
event: token     data: { token: "..." }
event: done      data: { conversation_id, response_time_ms, conversation_title? }
event: error     data: { message: "..." }
```

`sources` is emitted cumulatively: once before the first token (empty if the agent hasn't searched yet), and again whenever a later tool hop turns up code the client hasn't seen. Each event carries the full deduplicated list, so the client just replaces what it's holding.

---

## Running Tests

```bash
cd app/backend

# Unit + integration tests
pip install -r ../../requirements-dev.txt
pytest tests/ -v

# Router benchmark (compares regex vs heuristic vs llama3.2:1b)
python tests/benchmark_router.py
# Results saved to tests/results/router_benchmark_latest.{txt,json}
```

---

## Configuration Reference

All settings live in `config.py` and can be overridden via `.env` at the project root.

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `hybrid` | `ollama` · `cloud` · `hybrid` (cloud-first, Ollama fallback) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API base URL |
| `OLLAMA_MODEL` | `qwen2.5-coder:7b` | Main chat model |
| `OLLAMA_FAST_MODEL` | `llama3.2:1b` | Lightweight model for quick tasks (title generation) |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `OLLAMA_TIMEOUT` | `120` | Ollama request timeout (seconds) |
| `CLOUD_API_PROVIDER` | — | Display label for the cloud provider (e.g. `Groq`) |
| `CLOUD_API_KEY` | — | Groq API key |
| `CLOUD_API_BASE_URL` | — | Groq's OpenAI-compatible base URL |
| `CLOUD_MODEL` | — | Groq model name |
| `GITHUB_TOKEN` | — | Personal access token (avoids rate limits) |
| `MAX_REPO_SIZE_MB` | `500` | Reject repos larger than this before cloning |
| `MAX_CONTEXT_TOKENS` | `32768` | Total token budget per LLM request |
| `MAX_SEARCH_RESULTS` | `8` | Max code chunks returned by hybrid search |
| `RETRIEVAL_CANDIDATES` | `30` | Candidates fetched before RRF re-ranking |
| `CHUNK_MAX_TOKENS` | `1000` | Max tokens per code chunk |
| `CHUNK_OVERLAP_LINES` | `3` | Line overlap between adjacent chunks |
| `EMBEDDING_BATCH_SIZE` | `64` | Chunks sent per embedding request |
| `MIN_RELEVANCE_SCORE` | `0.0` | Drop chunks below this RRF score (0 = keep all) |
| `AGENT_MAX_TOOL_HOPS` | `6` | Max model↔tool round trips before the agent is forced to answer |

---

## Notes

- **Windows support** — the project is developed and tested on Windows. `safe_rmtree()` handles read-only files in cloned repos. Backend output encoding is UTF-8.
- **First index** — large repositories (>100 MB) can take several minutes to index. Progress is streamed to the UI in real time.
- **Re-indexing** — deleting and re-cloning a project is the current way to pick up upstream changes. Incremental re-indexing (hash-based diffing) is scaffolded in the DB layer and planned for a future release.
- **`llama3.2:1b` is optional** — if it isn't pulled, title generation falls back to the first 60 characters of the user's message. Everything else keeps working.
- **Cloud is Groq-specific** — the cloud path is built on `langchain-groq`, not a generic OpenAI-compatible client. Groq's API happens to be OpenAI-compatible, so `CLOUD_API_BASE_URL` still takes a full endpoint URL, but only Groq is tested/supported.
