"""Query agent: answers from the knowledge base as a tool.

Uses Agno KnowledgeTools (think → search_knowledge → analyze) as
agentic RAG: the model decides when to retrieve. Does not ingest files.
"""

from __future__ import annotations

import json
from textwrap import dedent
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


def scratch_for_run(run_context: RunContext) -> dict:
    """Keep think/analyze notes for this run only, not the whole session."""
    session_state = run_context.session_state
    if session_state is None:
        session_state = {}
        run_context.session_state = session_state
    run_id = getattr(run_context, "run_id", None) or "current"
    if session_state.get("_scratch_run_id") != run_id:
        session_state["_scratch_run_id"] = run_id
        session_state["thoughts"] = []
        session_state["analysis"] = []
    return session_state


class SharedCorpusKnowledgeTools(KnowledgeTools):
    """Search the shared contract corpus, not a per-chat user slice."""

    def think(self, run_context: RunContext, thought: str) -> str:
        """Use this tool as a scratchpad to reason about the question, refine your approach, brainstorm search terms, or revise your plan.

        Call `Think` whenever you need to figure out what to do next, analyze the user's question, or plan your approach.
        You should use this tool as frequently as needed. Notes from earlier turns are not included.

        Args:
            thought: Your thought process and reasoning.

        Returns:
            str: The reasoning log for the current turn.
        """
        try:
            log_debug(f"Thought: {thought}")
            session_state = scratch_for_run(run_context)
            session_state.setdefault("thoughts", []).append(thought)
            thoughts = "\n".join(f"- {item}" for item in session_state["thoughts"])
            return dedent(
                f"""Thoughts:
                {thoughts}
                """
            ).strip()
        except Exception as e:
            log_error(f"Error recording thought: {str(e)}")
            return f"Error recording thought: {e}"

    def analyze(self, run_context: RunContext, analysis: str) -> str:
        """Use this tool to evaluate whether the returned documents are correct and sufficient.

        If not, go back to Think or Search with refined queries. Notes from earlier turns are not included.

        Args:
            analysis: Evaluation of the current turn's search hits.

        Returns:
            str: The analysis log for the current turn.
        """
        try:
            log_debug(f"Analysis: {analysis}")
            session_state = scratch_for_run(run_context)
            session_state.setdefault("analysis", []).append(analysis)
            joined = "\n".join(f"- {item}" for item in session_state["analysis"])
            return dedent(
                f"""Analysis:
                {joined}
                """
            ).strip()
        except Exception as e:
            log_error(f"Error recording analysis: {str(e)}")
            return f"Error recording analysis: {e}"

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
            query: Search text for the CURRENT user message. Do not paste an
                earlier turn's clause number into this string.
            instrument_name: Named contract from this message, e.g. KGD6 PSC.
                Hits from other contracts are ranked down.
            doc_family: Catalog family such as PSC, JOA, JAO, PML, RSC, PEL.
            clause_id: Clause named in this message, e.g. 21.5. Never reuse
                a previous turn's clause unless the user still means it.
            article: Article named in this message, e.g. 21.
        """
        try:
            instrument_name = (instrument_name or "").strip()
            doc_family = (doc_family or "").strip()
            clause_id = (clause_id or "").strip()
            article = (article or "").strip()
            search_query = (query or "").strip()
            if not search_query:
                search_query = " ".join(
                    part for part in (instrument_name, clause_id or article) if part
                )
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
