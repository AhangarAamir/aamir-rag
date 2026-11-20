# Legal RAG — Agno ingest + query agents

Two agents over one SQLite index (`data/index.db`). The query agent researches with tools
(clause lookup, definitions, hybrid search, page reads, live reads of the source file) and every
answer gets verified citations and a References section with Document Center links.

Models are OpenAI only: chat on the Responses API (`OPENAI_MODEL`) and embeddings (`EMBEDDING_MODEL`).

## Architecture

```text
data/incoming/**/*.pdf|.md|.txt   (CORPUS_DIR)
        │
        ▼
 Ingest pipeline  (main.py ingest, or the Ingest Agent's `ingest` tool)
   extract (pdftotext / pypdf) → legal structure (Article → clause → definition)
   → one OpenAI call per document (title, parties, block, date, status)
   → context headers → OpenAI embeddings → Document Center link
        │
        ▼
 data/index.db
   documents (with link) · pages · articles · chunks · chunks_fts (BM25)
   embeddings · meta · feedback
        │
        ▼
 Query Agent
   think · list_documents · search_documents · get_clause · find_definition · read_pages
   read_source_pages · read_markdown_section · grep_source · search_learnings
        │
        ▼
 Answer with [C…] tags → verify post-hook → References (documents.link)
```

`search_documents` fuses BM25, vectors, exact clause numbers, defined terms and phrases with
reciprocal rank fusion, and expands abbreviations (KG-D6 = KGD6 = KG-DWN-98/3, PSC, JOA, MoPNG …)
in code. The post-hook (`legal_rag/agents/verify.py`) flags citation tags that were not retrieved
in that turn and quotes that are not verbatim, then writes References from the index.

## Setup (uv only)

```bash
uv sync
cp .env.example .env   # set OPENAI_API_KEY, OPENAI_MODEL, EMBEDDING_MODEL
```

Put files in `data/incoming/` using the Document Center folder tree (JOA/New JOA, PSC/Domestic,
PML, …, or `sites/Regulatory/...`). The folder path becomes the document's link.

## Run

```bash
uv run python main.py ingest                    # index everything under CORPUS_DIR
uv run python main.py ingest "JOA/New JOA" --force
uv run python main.py ingest --prune            # also drop rows for deleted files
uv run python main.py reembed                   # after changing EMBEDDING_MODEL
uv run python main.py links                     # rebuild documents.link after changing base URLs
uv run python main.py status
uv run python main.py query What is the participating interest of RIL?
uv run python main.py os                        # AgentOS + UI
uv run pytest                                   # offline tests (no API calls)
```

Each indexed file writes `data/import_logs/<file name>.log`.

The query UI is a Vite React app. Build it once, then AgentOS serves the files at `/`, `/observability` and `/learning`:

```bash
cd ui && npm install && npm run build && cd ..
uv run python agent_os.py
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) (`HOST` / `PORT` in `.env`). The page calls
`POST /agents/query-agent/runs` (SSE) and lists this browser's chats via `GET /sessions`.
Corpus files are served at `/files/<path>` (used by References when both link base URLs are empty).

For live React work while AgentOS is running: `cd ui && npm run dev` (Vite proxies `/agents`,
`/sessions`, and `/observability/*` to `http://127.0.0.1:8000`).

```bash
curl -X POST http://127.0.0.1:8000/agents/query-agent/runs \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "message=What is the participating interest of RIL?" \
  -d "stream=false"
```

## Document links (`documents.link`)

Built at ingest by `legal_rag.domain.document_link`, with the rules the old query prompt gave the model:

| file under CORPUS_DIR | link |
|---|---|
| `JOA/NEW JOA/file.pdf` | `https://enp1.ril.com/sites/Regulatory/Document%20Center/JOA/NEW%20JOA/file.pdf` |
| `PEL/file.pdf` | `https://enp1.ril.com/sites/Regulatory/Document%20Center/PEL/file.pdf` |
| `sites/Regulatory/KGD6/.../Document Center/file.pdf` | `https://enp1.ril.com/sites/Regulatory/KGD6/.../Document%20Center/file.pdf` |
| `RSC/aamirWS/data/ril_data/RSC/file.pdf` | `.../Document%20Center/RSC/file.pdf` (workspace segment removed) |

## Config (`.env`)

| Variable | Meaning |
|---|---|
| `OPENAI_API_KEY` | required |
| `OPENAI_MODEL` | chat model, default `gpt-5.6-luna` |
| `OPENAI_REASONING_EFFORT` | optional: none … xhigh |
| `EMBEDDING_MODEL` | default `text-embedding-3-large` |
| `INGEST_MODEL` | model for the per-document profile call (default `OPENAI_MODEL`) |
| `INGEST_LLM_PROFILE` | `false` = filename / text heuristics only |
| `CORPUS_DIR` | default `data/incoming` |
| `DOC_CENTER_BASE_URL`, `SITES_BASE_URL` | link bases; both empty = local `/files/` links |
| `SEARCH_TOP_K`, `CHUNK_MAX_CHARS`, `NUM_HISTORY_RUNS`, `TOOL_CALL_LIMIT` | retrieval / agent tuning |
| `HOST`, `PORT`, `CORS_ALLOWED_ORIGINS` | server |
| `OBS_*` | observability, see below |

## Observability dashboard

Every ingest and query run is written to `data/observability.db` by a custom Agno post-hook adapter (run row + tool spans + estimated cost). Open the HRF-style dashboard at:

[http://127.0.0.1:8000/observability](http://127.0.0.1:8000/observability)

The query UI also has an **Observability** link in the sidebar. The page shows runs, success, avg/p95 latency, tokens, cost, query mix, routes, tokens vs latency, and day-wise pricing. User comments can be added from that page.

Cost is estimated from token counts × USD / 1M rates. OpenAI does not expose a pricing API (`GET /v1/models` has no prices). At startup the app fetches the [LiteLLM model-price catalog](https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json), caches it under `data/model_prices.json`, and looks up `OPENAI_MODEL`. If that fetch fails it uses the cache, then a small built-in table (including official `gpt-5.6-luna` rates: **$0.20 / $0.02 / $1.20** per 1M). Changing `OPENAI_MODEL` in `.env` picks up the new model's rates on the next start. If a run includes provider `metrics.cost`, that value wins over the table.

The dashboard reads `data/observability.db`. On load it also imports completed runs from `data/sessions.db` (the existing Agno session store), so older chats show up after deploy. Streaming post-hooks fire before Agno sets `COMPLETED`; those runs are stored as success, not as still running.
