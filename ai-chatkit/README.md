<h1 align="center"> AI-CHATKIT </h1>
<p align="center">
  <strong style="background-color: green;">English</strong>
  |
  <a href="./README_zh.md" target="_Self">中文</a>
</p>

AI-CHATKIT is a full-stack AI agent chat tool built with LangGraph, FastAPI, NextJS, ChromaDB, and other components.

This project serves as a template to help you quickly build AI agent chat applications using the LangGraph framework, with RAG support to enhance knowledge base Q&A capabilities.

<img src="./pictures/chat_img.png" width="700"/>

Multi-agent collaboration:

<img src="./pictures/chat_multi_agent_img.png" width="700"/>

## Features

1. AI agent chat application built on LangGraph, supporting custom behavior logic orchestration.
2. Custom knowledge base Q&A with ChromaDB for storage and semantic search.
3. Custom tool invocation for agents.
4. Python backend API with FastAPI, supporting full asynchronous calls.
5. Custom frontend with NextJS.
6. Streaming chat output with SSE support.
7. Multiple custom agents.
8. Multi-agent collaboration.
9. Chat history stored in local browser cache.

## Prerequisites

| Dependency | Version | Purpose | Installation |
|------------|---------|---------|-------------|
| Python | **≥ 3.13** | Backend runtime | [python.org](https://www.python.org/downloads/) |
| Ollama | Latest | Local embedding model | [ollama.com](https://ollama.com/download) |
| pnpm | Latest | Frontend package manager | `npm install -g pnpm` |
| uv | Latest | Python dependency manager | `pip install uv` |

## Structure

- `backend`: Backend service code
- `frontend`: Frontend service code

---

## Quick Start

Follow these steps in order. Estimated time: **10-15 minutes** (excluding model downloads).

### Step 1: Backend Environment

```bash
cd backend

# Copy and edit environment variables
# Linux / macOS:
cp .env.example .env
# Windows (PowerShell):
copy .env.example .env
# ➜ Edit .env: fill in your DEEPSEEK_API_KEY
```

**Key .env settings**:

```properties
# ── Required ──────────────────────────
DEEPSEEK_API_KEY=sk-your-key-here
DEFAULT_MODEL=deepseek-v4-flash

# ── Embedding (requires local Ollama) ─
EMBEDDING_MODEL=bge-m3

# ── Database ──────────────────────────
DATABASE_URL=sqlite+aiosqlite:///resource/database.db

# ── Reranker (optional, auto-degrades if unavailable) ─
RERANKER_BACKEND=cross-encoder          # cross-encoder | llm | none
RERANKER_MODEL_PATH=BAAI/bge-reranker-v2-m3
HF_ENDPOINT=https://hf-mirror.com       # Required for users in China
```

### Step 2: Install Dependencies & Start Backend

```bash
cd backend

# Install Python dependencies
uv sync --frozen

# Activate virtual environment
# Linux / macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# Start backend server (default: http://localhost:8000)
python app/run_server.py
```

### Step 3: Deploy Embedding Model

The backend requires a local Ollama instance running the BGE-M3 embedding model (~1.2GB):

```bash
# Install and start Ollama (https://ollama.com/download)
ollama pull bge-m3
```

Verify:
```bash
ollama list | grep bge-m3
# Output: bge-m3:latest    xxx GB    ...
```

### Step 4: Import Policy Data

**This step is required.** Without it, ChromaDB is empty and searches return nothing.

```bash
cd backend
# Ensure virtual environment is activated

# Import sample policy data (58 Chinese government policies)
python scripts/import_policies.py resource/sample_policies.json

# To clear and re-import:
python scripts/import_policies.py resource/sample_policies.json --clean
```

Expected output:
```
Import complete: 58 policies, xxx chunks
```

### Step 5: Configure Reranker (Optional)

The Reranker performs fine-grained re-ranking with MMR diversity control:

| Backend | Model | Accuracy | Extra Dependency |
|---------|------|----------|-----------------|
| `cross-encoder` | BGE-Reranker-v2-m3 (~2GB) | ⭐⭐⭐ | Model download required |
| `llm` | DeepSeek V4 Flash | ⭐⭐ | None (uses existing API) |
| `none` | None | ⭐ | None (vector similarity only) |

**Automatic degradation**: If the cross-encoder model fails to load, the system auto-degrades to LLM → None backend. Set `RERANKER_BACKEND=llm` in `.env` to skip model download entirely.

The first run with `cross-encoder` downloads ~2GB model files (1-3 minutes). Subsequent runs use cached files.

### Step 6: Configure Frontend

```bash
cd frontend

# The API base URL defaults to localhost:8000
# Edit .env.local if your backend runs on a different port:
#   NEXT_PUBLIC_API_BASE_URL=http://localhost:8000

# Install dependencies
pnpm install

# Start frontend
pnpm dev
```

Visit http://localhost:3000/ to chat.

---

## Verify the Retrieval Pipeline

Use the trace script to diagnose the full RAG pipeline (no LLM call):

```bash
cd backend

# Basic trace
python scripts/trace_query.py "renewable energy subsidies" --dry-run

# Verbose mode: view every chunk with MMR ordering
python scripts/trace_query.py "cross-border data flow" --dry-run --verbose

# Compare MMR on vs off
python scripts/trace_query.py "cross-border data flow" --dry-run --verbose --no-mmr
```

---

## Troubleshooting

| Problem | Likely Cause | Solution |
|---------|-------------|---------|
| `ModuleNotFoundError` | Virtual environment not activated | `source .venv/bin/activate` then `uv sync --frozen` |
| ChromaDB returns empty | No data imported | Run Step 4 to import policy data |
| Ollama connection error | Ollama not running | Run `ollama serve` |
| Embedding error | bge-m3 not pulled | `ollama pull bge-m3` |
| Reranker download timeout | HuggingFace blocked (China) | Set `HF_ENDPOINT=https://hf-mirror.com` or use `RERANKER_BACKEND=llm` |
| `pip install uv` fails | pip too old | `pip install --upgrade pip` |
| `python: command not found` | Need Python 3.13+ | Install [Python 3.13+](https://www.python.org/downloads/) and add to PATH |
| Frontend returns 500 | Backend not running | Check http://localhost:8000/docs is accessible |

---

## Built-in Agents

This project supports creating and orchestrating multiple agents with LangGraph. Agent logic is defined in `backend/app/ai/agent/`. Switch between agents in the frontend UI.

### 1. Policy Q&A Assistant (POLICY-ASSISTANT)

RAG-powered government policy Q&A with semantic search, full-text retrieval, and metadata filtering.

```
User query → LLM decides tool → ChromaDB search → Rerank+MMR → LLM response
```

See: `backend/app/ai/agent/policy_assistant.py`

### 2. OA Assistant (OA-ASSISTANT)

Demo office assistant with employee info lookup and handbook knowledge base retrieval.

See: `backend/app/ai/agent/oa_assistant.py`

### 3. Multi-Agent Collaboration (MULTI_AGENT)

Demonstrates multi-agent orchestration with three sub-agents:
- `math_agent`: Math calculations
- `code_agent`: Code generation
- `general_agent`: General Q&A

The three agents collaborate through a Supervisor.

See: `backend/app/ai/agent/multi_agent.py`
