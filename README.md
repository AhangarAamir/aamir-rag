# Legal RAG — Agno ingest + query agents

Two agents, one shared knowledge base. Query time uses the knowledge base **as a tool** (`think` → `search_knowledge` → `analyze`), not traditional RAG that dumps chunks into the prompt.

This is a v1 scaffold. Clause IDs, as-of-date tables, PI ledger, and fiscal calculators are later phases.

## Architecture

```text
data/incoming/*.pdf
        │
        ▼
 Ingest Agent
   tools: KnowledgeManagementTools
     ingest_path / ingest_url / list_content / ingest_status
        │
        ▼
 Shared Knowledge
   Chroma (vectors) + SQLite contents db
   Path-aware PDF reader + LegalAgenticChunking
   (inner agent returns clause metadata + running summary)
        │
        ▼
 Query Agent
   tools: KnowledgeTools
     think / search_knowledge / analyze
        │
        ▼
 Answer + citations from retrieved chunks
```

Ingest agent never answers legal questions. Query agent never writes to the knowledge base.

## Setup (uv only)

```bash
cd legal_rag_agno
uv sync
cp .env.example .env   # set OPENAI_API_KEY, or LLM_PROVIDER for a compatible server
```

Put PDFs in `data/incoming/`, using the same families as the old ingest notebook:

```text
data/incoming/
  JOA/New JOA/
  JOA/Old JOA/
  JAO/NEW JAO/
  JAO/OLD JAO/
  PSC/Domestic/
  PML/
  RSC, PEL, FDP, DGH, MCM, MOM, QPR, …
```

## Run

```bash
# ingest
uv run python main.py ingest ingest all PDFs in data/incoming
uv run python ingest_cli.py

# query — knowledge is a tool, not preloaded context
uv run python main.py query What is the participating interest of RIL?
uv run python query_cli.py

# AgentOS — both agents over HTTP, plus the query UI
uv run python main.py os
# or: uv run python agent_os.py
```

The query UI is a Vite React app. Build it once, then AgentOS serves the files at `/`:

```bash
cd ui
npm install
npm run build
cd ..
uv run python agent_os.py
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) by default (`HOST` / `PORT` in `.env`). `agent_os.py` and `uv run python main.py os` use the same bind address. The page calls `POST /agents/query-agent/runs` (SSE) and lists this browser's chats via `GET /sessions`. A new chat uses a new `session_id`; a different browser gets a different `user_id` in `localStorage`.

For live React work while AgentOS is running:

```bash
cd ui && npm run dev
```

Vite proxies `/agents` and `/sessions` to `http://127.0.0.1:8000`.

The hosted control plane at [os.agno.com](https://os.agno.com) can still connect to the same AgentOS instance. Pick **Ingest Agent** (`ingest-agent`) or **Query Agent** (`query-agent`).

```bash
# example: run query agent via the AgentOS API
curl -X POST http://127.0.0.1:8000/agents/query-agent/runs \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "message=What is the participating interest of RIL?" \
  -d "stream=false"
```

## Config (`.env`)

| Variable | Meaning |
|---|---|
| `LLM_PROVIDER` | `openai` (default), `openailike`, `ollama`, `litellm`, or `vllm` |
| `OPENAI_API_KEY` | required for `openai`. Optional for a compatible server (Ollama accepts any value) |
| `OPENAI_MODEL` | default `gpt-4o`. For Ollama, LiteLLM, or vLLM this is the served model id |
| `OPENAI_BASE_URL` | OpenAI-compatible base URL, required unless `LLM_PROVIDER=openai`. Example: `http://127.0.0.1:11434/v1` |
| `OPENAI_LIKE_BACKEND` | `ollama`, `litellm`, or `vllm`. Ollama sends `think: false`. A URL on port `11434` is treated as Ollama when this is empty |
| `OPENAI_LIKE_OPTIONS` | JSON `options` on compatible requests. Default `{"num_ctx": 32768, "num_batch": 2048}` |
| `EMBEDDING_PROVIDER` | `openai`, `openailike`, `ollama`, `litellm`, `vllm`, or `huggingface`. Empty follows `LLM_PROVIDER` |
| `EMBEDDING_MODEL` | model id for that provider. Hugging Face examples: `sentence-transformers/all-MiniLM-L6-v2`, `Alibaba-NLP/gte-base-en-v1.5`. The weights download once into `data/models` and later runs load that copy |
| `EMBEDDING_BASE_URL` | embedding server. Ollama defaults to `http://127.0.0.1:11434`, or reuses `OPENAI_BASE_URL` when chat is Ollama. Unused for `huggingface` |
| `EMBEDDING_API_KEY` | optional embedding key. Falls back to `OPENAI_API_KEY`. Unused for local Hugging Face models |
| `EMBEDDING_DIMENSIONS` | vector size. Required for Ollama and Hugging Face. all-MiniLM is 384, Alibaba gte-base is 768, gte-large is 1024 |
| `CHUNKING_STRATEGY` | `agentic` (default custom legal chunker), `document`, or `recursive` |
| `CHUNKING_MAX_SIZE` | max characters per chunker window (default `4000`) |
| `CHUNKER_ASYNC` | `true` (default): recursive split + parallel `agent.arun` labels |
| `CHUNKER_CONCURRENCY` | parallel metadata calls per document (default `8`) |
| `SEARCH_MAX_RESULTS` | hits returned by `search_knowledge` (default `12`) |
| `HOST` | bind address for AgentOS (default `127.0.0.1`; alias `AGENT_OS_HOST`) |
| `PORT` | bind port for AgentOS (default `8000`; alias `AGENT_OS_PORT`) |
| `CORS_ALLOWED_ORIGINS` | comma-separated browser origins (default `https://uracs.enp.ril.com`, `https://uracs.ucg.ril.com`) |

`agentic` uses the custom legal chunker. Default is async: local recursive split, one identity call, then parallel metadata labeling. Set `CHUNKER_ASYNC=false` for the old sequential window walk. Switch to `document` if ingest is still too expensive.

## What each agent does

**Ingest**

1. User points at a file, folder, or `data/incoming`.
2. Agent calls `ingest_path`.
3. PDF reader stamps folder/filename, then LegalAgenticChunking (or the strategy in `.env`).
4. Each chunk stores instrument, family, article, clause, and a locator prefix.
   Vectors land in Chroma. Content rows land in `data/contents.db`.

**Query**

1. User asks a legal question.
2. Agent calls `think` to plan searches.
3. Agent calls `search_knowledge` (can repeat).
4. Agent calls `analyze` to check if evidence is enough.
5. Agent answers only from those hits.

## Later (not in v1)

- Stable `clause_id` / heading index
- `effective_from` / `effective_to` filters
- Participating interest SQL ledger
- Fiscal slab lookup and deadline calculator
- Citation verifier and conflict detector

Those need structured stores and extra tools. This folder is the two-agent control plane they plug into.
