# Legal RAG — Agno ingest + query agents

Two agents, one shared knowledge base. Query time uses the knowledge base **as a tool** (`think` → `search_knowledge` → `analyze`), not traditional RAG that dumps chunks into the prompt.

After split, a metadata agent sees the **file + chunk** and writes a uniqueness card (document aliases, article/clause locator, search prefix). Query search applies **hard filters** for a named instrument so another PSC is not substituted.

## Architecture

```text
JAO / PSC / PML / ... nested PDFs
        │
        ▼
 Ingest Agent
   ingest_legal_tree (preferred)
     per-file folder tags
     split → metadata agent uniqueness card
     search prefix + Chroma metadata
   fallback: ingest_path / ingest_url
        │
        ▼
 Shared Knowledge
   Chroma (hybrid) + SQLite contents db + alias catalog
        │
        ▼
 Query Agent
   think / search_knowledge(query, document, article, clause) / analyze
        │
        ▼
 Answer + citations from scoped hits
```

Ingest agent never answers legal questions. Query agent never writes to the knowledge base.

## Setup (uv only)

```bash
cd legal_rag_agno
uv sync
cp .env.example .env   # set OPENAI_API_KEY
```

Put PDFs in `data/incoming/` (flat or nested `PSC/Domestic/...`, `JAO/NEW JAO/...`).

## Run

```bash
# ingest via the agent (it should call ingest_legal_tree)
uv run python main.py ingest ingest all PDFs in data/incoming
uv run python ingest_cli.py

# re-ingest the incoming tree with uniqueness metadata (required after this change)
uv run python main.py reingest

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
| `CHUNKING_STRATEGY` | `recursive` (default), `agentic`, or `document` |
| `METADATA_BATCH_SIZE` | chunks per uniqueness-agent call (default `4`) |
| `DOCUMENT_CARD_CHARS` | file prefix sent for the document card (default `12000`) |

`recursive` is the default splitter. The metadata agent tags identity after the cut. `agentic` is slower; use it when OCR layout is too messy for recursive cuts.

Old vectors do not pick up uniqueness fields. Run `uv run python main.py reingest` after this change, then restart AgentOS so the query agent loads the new search filters.

## What each agent does

**Ingest**

1. User points at a file, folder, or `data/incoming`.
2. Agent calls `ingest_legal_tree` (preferred). Each PDF is inserted with folder tags (`folder_family`, `folder_vintage`, `folder_region`).
3. The reader splits (recursive / agentic / document).
4. The metadata agent builds a document card from the file, then a uniqueness card per chunk (locator, aliases, distinguisher) and prepends a search prefix before embedding.
5. Aliases such as `KGD6 PSC` are stored for query-time filters.

**Query**

1. User asks a legal question.
2. Agent calls `think` to plan searches.
3. Agent calls `search_knowledge` with `query` plus optional `document`, `article`, `clause`.
4. Named-instrument searches are filtered; zero hits return `not_in_corpus` instead of another contract.
5. Agent calls `analyze`, then answers only from those hits.

## Later (not in this pass)

- Parent/child chunk graph
- `effective_from` / `effective_to` filters
- Participating interest SQL ledger
- Fiscal slab lookup and deadline calculator
- Citation verifier and conflict detector
