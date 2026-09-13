"""Query agent: answers from the knowledge base as a tool.

Uses Agno KnowledgeTools (think → search_knowledge → analyze) as
agentic RAG. search_knowledge applies document/article/clause filters
so a named instrument is not silently replaced by another file.
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

from alias_catalog import SearchScope, chroma_filters, normalize_alias, resolve_search_scope
from config import NUM_HISTORY_RUNS, OPENAI_MODEL, SESSIONS_DB_FILE
from knowledge_store import get_knowledge
from prompts import QUERY_FEW_SHOT, QUERY_INSTRUCTIONS, QUERY_TOOL_INSTRUCTIONS


def _hit_matches_scope(doc: Document, scope: SearchScope) -> bool:
    if not scope.named:
        return True
    meta = doc.meta_data or {}
    blob = " ".join(
        [
            str(meta.get("instrument_key", "")),
            str(meta.get("filename", "")),
            str(meta.get("document_title", "")),
            str(meta.get("aliases", "")),
            str(meta.get("block_or_contract_area", "")),
            str(doc.name or ""),
            (doc.content or "")[:800],
        ]
    )
    normalized = normalize_alias(blob)
    if scope.instrument_key and scope.instrument_key.replace("_", " ") in normalized:
        return True
    if scope.filename and normalize_alias(scope.filename) in normalized:
        return True
    for alias in scope.aliases:
        if alias and alias in normalized:
            return True
    if scope.document and normalize_alias(scope.document) in normalized:
        return True
    return False


def _serialize(docs: List[Document]) -> str:
    return json.dumps([doc.to_dict() for doc in docs])


class SharedCorpusKnowledgeTools(KnowledgeTools):
    """Search the shared contract corpus, not a per-chat user slice."""

    def search_knowledge(
        self,
        run_context: RunContext,
        query: str,
        document: Optional[str] = None,
        article: Optional[str] = None,
        clause: Optional[str] = None,
    ) -> str:
        """Search ingested instruments. Call only when the question needs evidence.

        Args:
            query: Search text (clause wording, heading, defined term).
            document: Named instrument, filename, block, or alias. Empty if none.
            article: Article number such as "10". Empty if none.
            clause: Clause number such as "10.7". Empty if none.
        """
        try:
            scope = resolve_search_scope(query, document, article, clause)
            log_debug(
                f"Searching knowledge base: {query!r} document={scope.document!r} "
                f"article={scope.article!r} clause={scope.clause!r} "
                f"key={scope.instrument_key!r} resolved={scope.resolved}"
            )
            filters = chroma_filters(scope)
            hits = self._search(query, filters)
            if scope.named:
                hits = [doc for doc in hits if _hit_matches_scope(doc, scope)]
            if hits:
                payload = json.loads(_serialize(hits))
                if isinstance(payload, list):
                    return json.dumps({"status": "ok", "filters": filters, "hits": payload})
                return _serialize(hits)

            if scope.named and (scope.clause or scope.article):
                doc_only = chroma_filters(
                    SearchScope(
                        document=scope.document,
                        instrument_key=scope.instrument_key,
                        filename=scope.filename,
                        document_title=scope.document_title,
                        aliases=scope.aliases,
                        named=True,
                        resolved=scope.resolved,
                    )
                )
                relaxed = self._search(query, doc_only)
                relaxed = [doc for doc in relaxed if _hit_matches_scope(doc, scope)]
                if relaxed:
                    return json.dumps(
                        {
                            "status": "ok",
                            "note": "No chunk tagged with that exact clause; returning document-scoped hits.",
                            "filters": doc_only,
                            "hits": json.loads(_serialize(relaxed)),
                        }
                    )

            if scope.named:
                fallback = [doc for doc in self._search(query, None) if _hit_matches_scope(doc, scope)]
                if fallback:
                    return json.dumps(
                        {
                            "status": "ok",
                            "note": "Metadata filter missed; kept only hits matching the named instrument.",
                            "filters": filters,
                            "hits": json.loads(_serialize(fallback)),
                        }
                    )
                return json.dumps(
                    {
                        "status": "not_in_corpus",
                        "message": (
                            f"No chunks matched document={scope.document!r}. "
                            "Do not substitute another instrument."
                        ),
                        "filters": filters,
                        "hits": [],
                    }
                )
            return "No documents found"
        except Exception as e:
            log_error(f"Error searching knowledge base: {str(e)}")
            return f"Error searching knowledge base: {e}"

    def _search(self, query: str, filters: Optional[dict] = None) -> List[Document]:
        kwargs = {"query": query, "user_id": None}
        if filters:
            kwargs["filters"] = filters
        try:
            return self.knowledge.search(**kwargs)
        except Exception as exc:
            log_error(f"Filtered search failed ({exc}); retrying without filters")
            if not filters:
                raise
            return self.knowledge.search(query=query, user_id=None)


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
