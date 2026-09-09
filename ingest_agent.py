"""Ingest agent: loads files into the shared knowledge base.

Uses Agno KnowledgeManagementTools (ingest_path / list_content / ingest_status).
Does not answer legal questions.
"""

from __future__ import annotations

from textwrap import dedent

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.knowledge import KnowledgeManagementTools

from config import INCOMING_DIR, OPENAI_MODEL
from knowledge_store import get_knowledge

INGEST_INSTRUCTIONS = dedent(
    f"""
    You are the contract ingestion operator. You do not answer legal questions.

    Default document folder: {INCOMING_DIR}

    Workflow:
    1. If the user gives a file or folder path, call ingest_path with that path.
    2. If they say "ingest incoming" or give no path, ingest_path("{INCOMING_DIR}").
    3. After ingest, call ingest_status or list_content and report what was loaded.
    4. Never invent clauses, participating interest, or dates during ingest.
    5. Do not search the knowledge base to write a legal answer.

    Prefer local PDFs over URLs unless the user explicitly gives a URL.
    """
).strip()


def build_ingest_agent() -> Agent:
    knowledge = get_knowledge()
    return Agent(
        name="Ingest Agent",
        model=OpenAIChat(id=OPENAI_MODEL),
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
