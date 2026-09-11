"""Query agent: answers from the knowledge base as a tool.

Uses Agno KnowledgeTools (think → search_knowledge → analyze) as
agentic RAG: the model decides when to retrieve. Does not ingest files.
"""

from __future__ import annotations

import json
from typing import List

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.knowledge.document import Document
from agno.models.openai import OpenAIResponses
from agno.run import RunContext
from agno.tools.knowledge import KnowledgeTools
from agno.utils.log import log_debug, log_error

from config import NUM_HISTORY_RUNS, OPENAI_MODEL, SESSIONS_DB_FILE
from knowledge_store import get_knowledge
from prompts import QUERY_FEW_SHOT, QUERY_INSTRUCTIONS, QUERY_TOOL_INSTRUCTIONS


class SharedCorpusKnowledgeTools(KnowledgeTools):
    """Search the shared contract corpus, not a per-chat user slice."""

    def search_knowledge(self, run_context: RunContext, query: str) -> str:
        """Search ingested instruments. Call only when the question needs evidence."""
        try:
            log_debug(f"Searching knowledge base: {query}")
            relevant_docs: List[Document] = self.knowledge.search(query=query, user_id=None)
            if len(relevant_docs) == 0:
                return "No documents found"
            return json.dumps([doc.to_dict() for doc in relevant_docs])
        except Exception as e:
            log_error(f"Error searching knowledge base: {str(e)}")
            return f"Error searching knowledge base: {e}"


def build_query_agent() -> Agent:
    knowledge = get_knowledge()
    return Agent(
        id="query-agent",
        name="Query Agent",
        model=OpenAIResponses(id=OPENAI_MODEL),
        db=SqliteDb(
            id="query-sessions-db",
            db_file=str(SESSIONS_DB_FILE),
            session_table="query_sessions",
        ),
        enable_session_summaries=True,
        add_session_summary_to_context=True,
        add_history_to_context=True,
        num_history_runs=NUM_HISTORY_RUNS,
        tools=[
            SharedCorpusKnowledgeTools(
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
