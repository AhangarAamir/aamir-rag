"""Query agent: answers from the knowledge base as a tool.

Uses Agno KnowledgeTools (think → search_knowledge → analyze) as
agentic RAG: the model decides when to retrieve. Does not ingest files.
"""

from __future__ import annotations

import json
from typing import List, Optional

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.knowledge.document import Document
from agno.models.openai import OpenAIResponses
from agno.run import RunContext
from agno.tools.knowledge import KnowledgeTools
from agno.utils.log import log_debug, log_error

from config import NUM_HISTORY_RUNS, OPENAI_MODEL, SEARCH_MAX_RESULTS, SESSIONS_DB_FILE
from knowledge_store import get_knowledge
from legal_chunking import build_chroma_filters, document_matches_instrument, rank_documents
from prompts import QUERY_FEW_SHOT, QUERY_INSTRUCTIONS, QUERY_TOOL_INSTRUCTIONS


class SharedCorpusKnowledgeTools(KnowledgeTools):
    """Search the shared contract corpus, not a per-chat user slice."""

    def search_knowledge(
        self,
        run_context: RunContext,
        query: str,
        instrument_name: str = "",
        doc_family: str = "",
        clause_id: str = "",
        article: str = "",
    ) -> str:
        """Search ingested instruments. Call only when the question needs evidence.

        Args:
            query: Search text. Include the instrument name and clause number.
            instrument_name: Named contract, e.g. KGD6 PSC. Hits from other
                contracts are ranked down.
            doc_family: Catalog family such as PSC, JOA, JAO, PML, RSC, PEL.
            clause_id: Numbered clause, e.g. 10.7.
            article: Article number, e.g. 10.
        """
        try:
            instrument_name = (instrument_name or "").strip()
            doc_family = (doc_family or "").strip()
            clause_id = (clause_id or "").strip()
            article = (article or "").strip()

            lead = [part for part in (clause_id, instrument_name) if part]
            search_query = " ".join(lead + [query]).strip() if lead else query
            filters = build_chroma_filters(
                doc_family=doc_family,
                clause_id=clause_id,
                article=article,
            )
            log_debug(
                f"Searching knowledge base: {search_query!r} filters={filters} "
                f"instrument={instrument_name!r}"
            )
            relevant_docs = self._search(search_query, filters)
            if not relevant_docs and filters:
                log_debug("Filtered search empty; retrying without metadata filters")
                relevant_docs = self._search(search_query, None)
            if not relevant_docs:
                return "No documents found"

            ranked = rank_documents(
                relevant_docs,
                instrument_name=instrument_name,
                clause_id=clause_id,
                doc_family=doc_family,
            )
            if instrument_name:
                scoped = [
                    doc for doc in ranked if self._keeps_instrument(doc, instrument_name)
                ]
                if scoped:
                    ranked = scoped
                else:
                    log_debug(
                        f"No hit matched instrument {instrument_name!r}; "
                        "returning ranked unfiltered hits"
                    )
            return json.dumps([doc.to_dict() for doc in ranked])
        except Exception as e:
            log_error(f"Error searching knowledge base: {str(e)}")
            return f"Error searching knowledge base: {e}"

    def _search(self, query: str, filters: Optional[dict]) -> List[Document]:
        return self.knowledge.search(
            query=query,
            max_results=SEARCH_MAX_RESULTS,
            filters=filters,
            user_id=None,
        )

    @staticmethod
    def _keeps_instrument(doc: Document, instrument_name: str) -> bool:
        return document_matches_instrument(doc, instrument_name)


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
