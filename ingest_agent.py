"""Ingest agent: loads files into the shared knowledge base.

Uses Agno KnowledgeManagementTools (ingest_path / list_content / ingest_status).
Does not answer legal questions.
"""

from __future__ import annotations

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIChat, OpenAIResponses
from agno.tools.knowledge import KnowledgeManagementTools

from config import OPENAI_MODEL, SESSIONS_DB_FILE
from knowledge_store import get_knowledge
from prompts import INGEST_INSTRUCTIONS


def build_ingest_agent() -> Agent:
    knowledge = get_knowledge()
    return Agent(
        id="ingest-agent",
        name="Ingest Agent",
        model=OpenAIResponses(id=OPENAI_MODEL),
        db=SqliteDb(
            id="ingest-sessions-db",
            db_file=str(SESSIONS_DB_FILE),
            session_table="ingest_sessions",
        ),
        tools=[
            KnowledgeManagementTools(
                knowledge=knowledge,
                ingest_path=True,
                ingest_url=True,
                ingest_text=False,
                remove_content=False,
                instructions=INGEST_INSTRUCTIONS,
                add_instructions=True,
            )
        ],
        search_knowledge=False,
        markdown=True,
        instructions=INGEST_INSTRUCTIONS,
    )
