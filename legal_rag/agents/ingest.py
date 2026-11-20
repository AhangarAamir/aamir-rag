"""Ingest agent: indexes corpus files into data/index.db.

Tools (legal_rag.agents.tools.LibraryTools): ingest, library_status,
list_documents. Does not answer legal questions. Each finished document
writes data/import_logs/<file name>.log.
"""

from __future__ import annotations

from agno.agent import Agent
from agno.db.sqlite import SqliteDb

from legal_rag.agents.tools import LibraryTools
from legal_rag.config import SESSIONS_DB_FILE
from legal_rag.models import build_chat_model
from legal_rag.observability.adapter import attach_observability
from legal_rag.prompts import INGEST_INSTRUCTIONS


def build_ingest_agent() -> Agent:
    return attach_observability(Agent(
        id="ingest-agent",
        name="Ingest Agent",
        description="Indexes PDF, Markdown and text files from the corpus folder.",
        model=build_chat_model(),
        db=SqliteDb(
            id="ingest-sessions-db",
            db_file=str(SESSIONS_DB_FILE),
            session_table="ingest_sessions",
        ),
        tools=[LibraryTools()],
        instructions=INGEST_INSTRUCTIONS,
        markdown=True,
    ))
