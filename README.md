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
cp .env.example .env   # set OPENAI_API_KEY
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

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) when you start with `agent_os.py` (port 8000). `uv run python main.py os` uses AgentOS’s default port 7777. The page calls `POST /agents/query-agent/runs` (SSE) and lists this browser's chats via `GET /sessions`. A new chat uses a new `session_id`; a different browser gets a different `user_id` in `localStorage`.

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
| `OPENAI_API_KEY` | required |
| `OPENAI_MODEL` | default `gpt-4o` |
| `EMBEDDING_MODEL` | default `text-embedding-3-large` |
| `CHUNKING_STRATEGY` | `agentic` (default custom legal chunker), `document`, or `recursive` |
| `CHUNKING_MAX_SIZE` | max characters per chunker window (default `4000`) |
| `CHUNKER_ASYNC` | `true` (default): recursive split + parallel `agent.arun` labels |
| `CHUNKER_CONCURRENCY` | parallel metadata calls per document (default `8`) |
| `SEARCH_MAX_RESULTS` | hits returned by `search_knowledge` (default `12`) |

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
