"""Query agent: answers from the knowledge base as a tool.

Uses Agno KnowledgeTools (think → search_knowledge → analyze).
Does not ingest files.
"""

from __future__ import annotations

from textwrap import dedent

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIChat
from agno.tools.knowledge import KnowledgeTools

from config import NUM_HISTORY_RUNS, OPENAI_MODEL, QUERY_SESSION_ID, SESSIONS_DB_FILE
from knowledge_store import get_knowledge

QUERY_INSTRUCTIONS = dedent(
    """
    You are a precise assistant for oil & gas contracts (PSC, JOA, assignments, notifications).

    Always use knowledge tools before answering:
    1. think — plan search terms (clause numbers, headings, party names, fiscal year).
    2. search_knowledge — retrieve evidence. Search again if the first pass is thin.
    3. analyze — check whether the hits actually support the answer.

    Rules:
    - Answer only from retrieved documents. If evidence is missing, say so.
    - Cite filename / heading / clause if present in the retrieved text.
    - If the user assumes a fact that retrieved documents contradict, correct the premise first.
    - If two sources conflict, show both values and dates. Do not pick one silently.
    - If the question is as-of a date, prefer the version that was in force then.
    - Use the session summary and recent conversation history when the user refers
      to earlier answers (for example "that rate", "same party", "and the other document").
    - Do not ingest files. Do not guess calculated dates or fiscal slabs.
    """
).strip()


def build_query_agent() -> Agent:
    knowledge = get_knowledge()
    return Agent(
        name="Query Agent",
        model=OpenAIChat(id=OPENAI_MODEL),
        db=SqliteDb(db_file=str(SESSIONS_DB_FILE), session_table="query_sessions"),
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
                add_instructions=True,
                add_few_shot=True,
            )
        ],
        search_knowledge=False,
        markdown=True,
        instructions=QUERY_INSTRUCTIONS,
    )
