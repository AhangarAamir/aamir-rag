"""Query agent: researches the indexed documents with tools and cites them.

Tools (legal_rag.agents.tools / source_tools):
  think, list_documents, search_documents, get_clause, find_definition,
  read_pages, read_source_pages, read_markdown_section, grep_source.
Every passage a tool returns is tagged [C<id>]. The finalize_answer post-hook
verifies the tags and quotes and appends References built from documents.link.
"""

from __future__ import annotations

from agno.agent import Agent
from agno.db.sqlite import SqliteDb

from legal_rag.agents.source_tools import SourceFileTools
from legal_rag.agents.tools import LegalResearchTools
from legal_rag.agents.verify import finalize_answer
from legal_rag.config import NUM_HISTORY_RUNS, SESSIONS_DB_FILE, TOOL_CALL_LIMIT
from legal_rag.learning.runtime import get_learning_machine
from legal_rag.models import build_chat_model
from legal_rag.observability.adapter import attach_observability
from legal_rag.prompts import QUERY_INSTRUCTIONS

QUERY_AGENT_ID = "query-agent"

LEARNING_RECALL = (
    "Before answering a factual question, call search_learnings and apply any "
    "approved lesson that fits. Those lessons are shared across sessions and "
    "users. Do not save a learning yourself; only a reviewer can approve one."
)


def build_query_agent() -> Agent:
    return attach_observability(Agent(
        id=QUERY_AGENT_ID,
        name="Query Agent",
        description="Answers contract and regulatory questions from the indexed documents, with verified citations.",
        model=build_chat_model(),
        db=SqliteDb(
            id="query-sessions-db",
            db_file=str(SESSIONS_DB_FILE),
            session_table="query_sessions",
        ),
        tools=[LegalResearchTools(), SourceFileTools()],
        instructions=f"{QUERY_INSTRUCTIONS}\n\n{LEARNING_RECALL}",
        learning=get_learning_machine(),
        add_history_to_context=True,
        num_history_runs=NUM_HISTORY_RUNS,
        max_tool_calls_from_history=0,
        add_datetime_to_context=True,
        tool_call_limit=TOOL_CALL_LIMIT,
        # finalize_answer runs before the observability hook, so traces store the verified answer.
        post_hooks=[finalize_answer],
        markdown=True,
    ))
