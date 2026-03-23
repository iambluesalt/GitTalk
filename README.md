# GitTalk — Chat with any codebase

AI-powered codebase analysis with GitHub integration. Local-only, privacy-first.

**Workflow:** GitHub URL → shallow clone → AST indexing → hybrid search → chat with code

---

## Stack

- **Frontend:** React 19 + React Router v7 + Vite + Tailwind CSS v4
- **Backend:** FastAPI + SQLite (metadata) + LanceDB (vectors)
- **LLM:** Ollama (local) with optional cloud API fallback
- **Embeddings:** nomic-embed-text via Ollama

---

## Quick Start

### 1. Install frontend dependencies

```bash
npm install
```

### 2. Set up the Python backend

```bash
cd app/backend
python -m venv ../../venv
../../venv/Scripts/activate   # Windows
# source ../../venv/bin/activate  # macOS/Linux
pip install -r ../../requirements.txt
```

### 3. Configure (optional)

Configure via the Settings UI, or manually edit `.env` at the project root:

```env
LLM_PROVIDER=hybrid          # ollama | cloud | hybrid
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5-coder:7b
OLLAMA_EMBED_MODEL=nomic-embed-text

# Cloud fallback (optional)
CLOUD_API_KEY=your-key
CLOUD_API_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
CLOUD_MODEL=gemini-2.0-flash
```

### 4. Start Ollama

```bash
ollama serve
ollama pull qwen2.5-coder:7b
ollama pull nomic-embed-text
```

### 5. Run

**Backend** (terminal 1):
```bash
cd app/backend
python main.py
```

**Frontend** (terminal 2):
```bash
npm run dev
```

Open [http://localhost:5173](http://localhost:5173).

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/clone` | Clone a GitHub repo (SSE) |
| `GET` | `/api/projects` | List all projects |
| `GET` | `/api/projects/:id` | Get project details |
| `DELETE` | `/api/projects/:id` | Delete a project |
| `POST` | `/api/index` | Index a project (SSE) |
| `POST` | `/api/chat` | Chat with a project (SSE) |
| `GET` | `/api/conversations/:project_id` | List conversations |
| `GET` | `/api/config` | Get current config |
| `PUT` | `/api/config` | Update config (persists to .env) |
| `GET` | `/api/health` | Service health check |

---

## Project Structure

```
app/
├── backend/
│   ├── config.py          # Pydantic settings
│   ├── main.py            # FastAPI app + health endpoint
│   ├── models.py          # Pydantic request/response models
│   ├── routes/            # API route handlers
│   ├── services/          # Business logic (clone, index, chat, etc.)
│   ├── storage/           # SQLite + LanceDB wrappers
│   └── utils/             # Exclusions, helpers
├── components/            # Shared React components
├── lib/                   # API client + types
└── routes/                # Page components
```
