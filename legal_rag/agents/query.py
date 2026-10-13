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
from agno.run import RunContext
from agno.tools.knowledge import KnowledgeTools
from agno.utils.log import log_debug, log_error

from legal_rag.config import NUM_HISTORY_RUNS, SEARCH_MAX_RESULTS, SESSIONS_DB_FILE, build_chat_model
from legal_rag.knowledge.store import get_knowledge
from legal_rag.learning.runtime import get_learning_machine
from legal_rag.knowledge.chunking import (
    build_chroma_filters,
    document_matches_instrument,
    rank_documents,
    strip_workspace_path,
)
from legal_rag.observability.adapter import attach_observability
from legal_rag.prompts import QUERY_FEW_SHOT, QUERY_INSTRUCTIONS, QUERY_TOOL_INSTRUCTIONS
from legal_rag.agents.topic_shift import isolate_unrelated_query, restore_query_history

LEARNING_RECALL = (
    "Before answering a factual question, call search_learnings and apply any "
    "approved lesson that fits. Those lessons are shared across sessions and "
    "users. Do not save a learning yourself; only a reviewer can approve one."
)


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
        heading: str = "",
        parties: str = "",
        block: str = "",
        content_type: str = "",
        document_status: str = "",
        defined_terms: str = "",
        cross_references: str = "",
        ocr_uncertainty: str = "",
    ) -> str:
        """Search ingested instruments. Call only when the question needs evidence.

        Args:
            query: Search text for the CURRENT user message. Do not paste an
                earlier turn's clause number into this string.
            instrument_name: Named contract from this message, e.g. KGD6 PSC.
                Matches instrument_name or derived instrument_key.
            doc_family: Catalog family such as PSC, JOA, JAO, PML, RSC, PEL.
            clause_id: Clause named in this message, e.g. 21.5. Never reuse
                a previous turn's clause unless the user still means it.
            article: Article named in this message, e.g. 21.
            heading: Exact or stored heading, e.g. Assignment of Interest.
            parties: Party named in this message, e.g. RIL.
            block: Block or contract area, e.g. KG-D6 or NEC-OSN-97-2.
            content_type: clause, definition, table, toc, preamble,
                schedule, annex, or other.
            document_status: executed, draft, amendment, notification,
                minutes, playbook, or unknown.
            defined_terms: Defined term from this message, e.g. Petroleum.
            cross_references: Referenced article/schedule named here.
            ocr_uncertainty: "true" only when the user asks about OCR-uncertain
                text. Leave empty otherwise.
        """
        try:
            instrument_name = (instrument_name or "").strip()
            doc_family = (doc_family or "").strip()
            clause_id = (clause_id or "").strip()
            article = (article or "").strip()
            heading = (heading or "").strip()
            parties = (parties or "").strip()
            block = (block or "").strip()
            content_type = (content_type or "").strip()
            document_status = (document_status or "").strip()
            defined_terms = (defined_terms or "").strip()
            cross_references = (cross_references or "").strip()
            ocr_uncertainty = (ocr_uncertainty or "").strip()
            search_query = (query or "").strip()
            if not search_query:
                search_query = " ".join(
                    part
                    for part in (instrument_name, clause_id or article, heading, block)
                    if part
                )
            filters = build_chroma_filters(
                doc_family=doc_family,
                clause_id=clause_id,
                article=article,
                instrument_name=instrument_name,
                heading=heading,
                parties=parties,
                block=block,
                content_type=content_type,
                document_status=document_status,
                defined_terms=defined_terms,
                cross_references=cross_references,
                ocr_uncertainty=ocr_uncertainty,
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
                heading=heading,
                parties=parties,
                block=block,
                content_type=content_type,
                document_status=document_status,
                defined_terms=defined_terms,
                cross_references=cross_references,
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
            payload = [doc.to_dict() for doc in ranked]
            return strip_workspace_path(json.dumps(payload))
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
    return attach_observability(Agent(
        id="query-agent",
        name="Query Agent",
        model=build_chat_model(),
        db=SqliteDb(
            id="query-sessions-db",
            db_file=str(SESSIONS_DB_FILE),
            session_table="query_sessions",
        ),
        enable_session_summaries=True,
        add_session_summary_to_context=True,
        add_history_to_context=True,
        num_history_runs=NUM_HISTORY_RUNS,
        max_tool_calls_from_history=0,
        pre_hooks=[isolate_unrelated_query],
        post_hooks=[restore_query_history],
        learning=get_learning_machine(),
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
        instructions=f"{QUERY_INSTRUCTIONS}\n\n{LEARNING_RECALL}",
    ))
