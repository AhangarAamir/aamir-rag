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
   PDF reader uses AgenticChunking (LLM picks section cuts)
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

Put PDFs in `data/incoming/`.

## Run

```bash
# ingest
uv run python main.py ingest ingest all PDFs in data/incoming
uv run python ingest_cli.py

# query — knowledge is a tool, not preloaded context
uv run python main.py query What is the participating interest of RIL?
uv run python query_cli.py
```

## Config (`.env`)

| Variable | Meaning |
|---|---|
| `OPENAI_API_KEY` | required |
| `OPENAI_MODEL` | default `gpt-4o` |
| `EMBEDDING_MODEL` | default `text-embedding-3-large` |
| `CHUNKING_STRATEGY` | `agentic` (default), `document`, or `recursive` |

`agentic` matches the idea that the model decides section boundaries while saving. It is slower and can be inconsistent across re-ingests. Switch to `document` if ingest is too expensive.

## What each agent does

**Ingest**

1. User points at a file, folder, or `data/incoming`.
2. Agent calls `ingest_path`.
3. PDF reader splits with AgenticChunking (or the strategy in `.env`).
4. Chunks land in Chroma. Content rows land in `data/contents.db`.

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
