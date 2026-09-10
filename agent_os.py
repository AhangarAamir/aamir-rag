"""AgentOS FastAPI runtime: exposes ingest and query agents.

Run: `uv run python agent_os.py`  or  `uv run python main.py os`

Connect the UI at https://os.agno.com to http://localhost:7777
"""

from __future__ import annotations

from agno.db.sqlite import SqliteDb
from agno.os import AgentOS

from config import DATA_DIR, require_api_key
from ingest_agent import build_ingest_agent
from knowledge_store import get_knowledge
from query_agent import build_query_agent

require_api_key()

ingest_agent = build_ingest_agent()
query_agent = build_query_agent()
knowledge = get_knowledge()

agent_os = AgentOS(
    id="legal-rag-os",
    name="Legal RAG AgentOS",
    description="Ingest agent writes contracts; query agent answers from knowledge tools.",
    agents=[ingest_agent, query_agent],
    knowledge=[knowledge],
    db=SqliteDb(id="agentos-db", db_file=str(DATA_DIR / "agentos.db")),
)
app = agent_os.get_app()


if __name__ == "__main__":
    agent_os.serve(app="agent_os:app", reload=True, port=8000)
