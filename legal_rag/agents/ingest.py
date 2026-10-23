"""Ingest agent: loads files into the shared knowledge base.

Uses Agno KnowledgeManagementTools (ingest_path / list_content / ingest_status).
Does not answer legal questions. Each finished document writes
data/import_logs/<file name>.log.
"""

from __future__ import annotations

from datetime import datetime

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIResponses
from agno.tools.knowledge import KnowledgeManagementTools

from legal_rag.config import OPENAI_MODEL, SESSIONS_DB_FILE
from legal_rag.knowledge.import_log import record_import
from legal_rag.knowledge.store import get_knowledge
from legal_rag.observability.adapter import attach_observability
from legal_rag.prompts import INGEST_INSTRUCTIONS


class LoggingKnowledgeManagementTools(KnowledgeManagementTools):
    """Same tools. Each document log is written when that document finishes."""

    def ingest_path(self, run_context, path: str) -> str:  # type: ignore[override]
        started_at = datetime.now()
        result = super().ingest_path(run_context, path)
        return self._attach_import_log(path, started_at, result, run_context)

    async def aingest_path(self, run_context, path: str) -> str:  # type: ignore[override]
        started_at = datetime.now()
        result = await super().aingest_path(run_context, path)
        return self._attach_import_log(path, started_at, result, run_context)

    def _attach_import_log(self, path: str, started_at: datetime, result: str, run_context) -> str:
        try:
            user_id = self._owner_id(run_context)
        except Exception:
            user_id = None
        return record_import(
            self.knowledge,
            source_path=path,
            started_at=started_at,
            result=result,
            user_id=user_id,
        )


def build_ingest_agent() -> Agent:
    knowledge = get_knowledge()
    return attach_observability(Agent(
        id="ingest-agent",
        name="Ingest Agent",
        model=OpenAIResponses(id=OPENAI_MODEL),
        db=SqliteDb(
            id="ingest-sessions-db",
            db_file=str(SESSIONS_DB_FILE),
            session_table="ingest_sessions",
        ),
        tools=[
            LoggingKnowledgeManagementTools(
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
    ))
