"""Ingest agent: loads files into the shared knowledge base.

Uses ingest_legal_tree (per-file folder tags + uniqueness metadata) plus
Agno KnowledgeManagementTools for list/status/url fallback.
Does not answer legal questions.
"""

from __future__ import annotations

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIResponses
from agno.run import RunContext
from agno.tools import Toolkit
from agno.tools.knowledge import KnowledgeManagementTools

from config import INCOMING_DIR, OPENAI_MODEL, SESSIONS_DB_FILE
from ingest_pipeline import ingest_legal_tree
from knowledge_store import get_knowledge
from prompts import INGEST_INSTRUCTIONS


class LegalIngestTools(Toolkit):
    """Per-file ingest that keeps JAO/PSC/PML folder identity on each PDF."""

    def __init__(self, **kwargs):
        super().__init__(
            name="legal_ingest",
            tools=[self.ingest_legal_tree],
            **kwargs,
        )

    def ingest_legal_tree(self, run_context: RunContext, path: str = "") -> str:
        """Ingest a legal file or nested folder tree into the knowledge base.

        Prefer this over ingest_path. Nested folders such as JAO, NEW JAO,
        OLD JAO, PSC, Domestic, and PML become metadata on each file. After
        split, a metadata agent stamps uniqueness fields used at search time.

        Args:
            path: File or directory. Empty uses the default incoming folder.
        """
        target = (path or "").strip() or str(INCOMING_DIR)
        return ingest_legal_tree(target)


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
            LegalIngestTools(),
            KnowledgeManagementTools(
                knowledge=knowledge,
                ingest_path=True,
                ingest_url=True,
                ingest_text=False,
                remove_content=False,
                instructions=INGEST_INSTRUCTIONS,
                add_instructions=True,
            ),
        ],
        search_knowledge=False,
        markdown=True,
        instructions=INGEST_INSTRUCTIONS,
    )
