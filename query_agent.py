"""Query agent: answers from the knowledge base as a tool.

Uses Agno KnowledgeTools (think → search_knowledge → analyze).
Does not ingest files.
"""

from __future__ import annotations

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIChat
from agno.tools.knowledge import KnowledgeTools

from config import NUM_HISTORY_RUNS, OPENAI_MODEL, QUERY_SESSION_ID, SESSIONS_DB_FILE
from knowledge_store import get_knowledge
from prompts import QUERY_FEW_SHOT, QUERY_INSTRUCTIONS, QUERY_TOOL_INSTRUCTIONS


def build_query_agent() -> Agent:
    knowledge = get_knowledge()
    return Agent(
        id="query-agent",
        name="Query Agent",
        model=OpenAIChat(id=OPENAI_MODEL),
        db=SqliteDb(
            id="query-sessions-db",
            db_file=str(SESSIONS_DB_FILE),
            session_table="query_sessions",
        ),
        session_id=QUERY_SESSION_ID,
        enable_session_summaries=True,
        add_session_summary_to_context=True,
        add_history_to_context=True,
        num_history_runs=NUM_HISTORY_RUNS,
        tools=[
            KnowledgeTools(
                knowledge=knowledge,
                enable_think=True,
                enable_search=True,
                enable_analyze=True,
                instructions=f"{QUERY_TOOL_INSTRUCTIONS}\n\n{QUERY_FEW_SHOT}",
                add_instructions=True,
            )
        ],
        search_knowledge=False,
        markdown=True,
        instructions=QUERY_INSTRUCTIONS,
    )
