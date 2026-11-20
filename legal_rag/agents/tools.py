"""Research tools for the query agent.

Every passage a tool returns is tagged [C<chunk id>]. The agent cites those
tags; a post-hook turns them into a References section with real links and
flags any tag or quote that did not come from this turn's tool results.
Output is compact plain text, which small local models follow more reliably
than nested JSON.
"""

from __future__ import annotations

import re
from typing import Any

from agno.run import RunContext
from agno.tools import Toolkit

from legal_rag.index import search as engine
from legal_rag.index import store

SEARCH_SNIPPET = 1800


def _where(hit: dict[str, Any]) -> str:
    parts = []
    if hit["article"]:
        label = f"Article {hit['article']}" if hit["article"].isdigit() else hit["article"]
        parts.append(label + (f" {hit['article_title']}" if hit["article_title"] else ""))
    if hit["kind"] == "definition" and hit["defined_term"]:
        parts.append(f'definition “{hit["defined_term"]}”' + (f" (clause {hit['clause_id']})" if hit["clause_id"] else ""))
    elif hit["clause_id"]:
        parts.append(f"clause {hit['clause_id']}" + (f" {hit['heading']}" if hit["heading"] else ""))
    elif hit["kind"] == "toc":
        parts.append("table of contents")
    elif hit["kind"] == "preamble":
        parts.append("preamble")
    return " › ".join(parts) or "text"


def format_hit(hit: dict[str, Any], limit: int | None = None) -> str:
    pages = f"p. {hit['page_start']}" if hit["page_start"] == hit["page_end"] else f"pp. {hit['page_start']}–{hit['page_end']}"
    meta = ", ".join(x for x in (hit["family"], hit["status"], hit["doc_date"]) if x)
    part = f" · part {hit['part']}/{hit['parts']}" if hit["parts"] > 1 else ""
    head = f"[C{hit['id']}] {hit['document']}{f' ({meta})' if meta else ''} — {_where(hit)} · {pages}{part}\nFile: {hit['path']}"
    text = hit["text"]
    if limit and len(text) > limit:
        follow = (
            f"get_clause('{hit['clause_id']}', '{hit['document']}')" if hit["clause_id"]
            else f"read_pages('{hit['document']}', {hit['page_start']})"
        )
        text = text[:limit].rstrip() + f" … [truncated — call {follow} for the full text]"
    refs = []
    own = hit["clause_id"] or ""
    for ref in re.findall(r"(?i)\b(?:article|clause|section)\s+(\d{1,3}(?:\.\d{1,3}){0,3})", hit["text"]):
        if ref != own and ref not in refs:
            refs.append(ref)
    for ref in re.findall(r"(?i)\b((?:appendix|annex|annexure|schedule)\s+[A-Z0-9]{1,3})\b", hit["text"]):
        if ref.title() not in refs:
            refs.append(ref.title())
    tail = f"\nRefers to: {', '.join(refs[:8])} (retrieve these if they matter to the answer)" if refs else ""
    return f"{head}\n{text}{tail}"


def _render(chunks: list[dict[str, Any]], note: str = "", header: str = "", limit: int | None = None) -> str:
    out = []
    if header:
        out.append(header)
    if note:
        out.append(f"Note: {note}")
    if not chunks:
        out.append("No passages found.")
    for hit in chunks:
        out.append(format_hit(hit, limit))
    return "\n\n".join(out)


class LegalResearchTools(Toolkit):
    def __init__(self, **kwargs: Any):
        super().__init__(
            name="legal_research",
            tools=[
                self.think,
                self.list_documents,
                self.search_documents,
                self.get_clause,
                self.find_definition,
                self.read_pages,
            ],
            **kwargs,
        )

    def think(self, run_context: RunContext, thought: str) -> str:
        """Private scratchpad. Plan searches, list what the question assumes, or check whether the passages you have are enough. Not shown to the user.

        Args:
            thought: Your reasoning.
        """
        state = run_context.session_state if run_context.session_state is not None else {}
        notes = state.setdefault("_thoughts", {})
        mine = notes.setdefault(run_context.run_id, [])
        mine.append(thought)
        # Keep only this run's notes.
        for key in [k for k in notes if k != run_context.run_id]:
            notes.pop(key, None)
        return "Noted. Thoughts so far:\n" + "\n".join(f"- {t}" for t in mine)

    def list_documents(self, filter: str = "") -> str:
        """List indexed documents with their names, family, date and status. Use it to check that a document the user names is indexed and to get its exact name.

        Args:
            filter: Optional words to match against name, file, family or block (e.g. "KG-D6", "JOA"). Empty lists everything.
        """
        docs = store.list_documents()
        if filter.strip():
            scope = engine.resolve_documents(filter)
            if scope.doc_ids:
                docs = scope.documents
            else:
                needle = filter.lower()
                docs = [d for d in docs if needle in f"{d['title']} {d['filename']} {d['family']} {d['block']}".lower()]
        if not docs:
            return f'No indexed documents match "{filter}".' if filter else "The library is empty."
        lines = [f"{len(docs)} document(s):"]
        for d in docs[:60]:
            name = d["short_name"] or d["title"]
            extra = ", ".join(x for x in (d["family"], d["status"], d["doc_date"], f"{d['pages']} pages") if x)
            lines.append(f"- {name} — {d['folder'] + '/' if d['folder'] else ''}{d['filename']} ({extra})")
        if len(docs) > 60:
            lines.append(f"… {len(docs) - 60} more. Narrow the filter.")
        return "\n".join(lines)

    def search_documents(self, query: str, document: str = "", family: str = "", max_results: int = 8) -> str:
        """Hybrid keyword + semantic search over every indexed passage. Abbreviations (PSC, JOA, KG-D6, PI, MoPNG …) are expanded automatically.

        Args:
            query: What to find, in the user's words plus legal synonyms. Include clause numbers and headings when known.
            document: Name of the document to search inside, e.g. "KG-D6 PSC" or part of a file name. Leave empty to search all documents.
            family: Optional document family code: PSC, JOA, JAO, PML, RSC, PEL, FDP, DGH, MCM, MCR, MOM, AMENDMENT, ASSIGNMENT, NOTIFICATION.
            max_results: Passages to return (1-12).
        """
        result = engine.search(query, document=document, family=family, top_k=max(1, min(int(max_results or 8), 12)))
        scope = ", ".join(s["document"] for s in result.get("scope", []))
        header = f'Search: "{result["query"]}"'
        if result.get("expanded"):
            header += f" (also searched: {', '.join(result['expanded'][:6])})"
        if scope:
            header += f" · within: {scope}"
        return _render(result["hits"], result.get("note", ""), header, SEARCH_SNIPPET)

    def get_clause(self, clause: str, document: str = "") -> str:
        """Exact full text of a numbered clause with its sub-clauses (e.g. "10.7"), a whole article ("10" or "Article 10"), or a part ("Appendix C", "Schedule 4"). Use whenever the user or a passage names a clause.

        Args:
            clause: Clause number, article number or part label.
            document: Document name, e.g. "KG-D6 PSC". Strongly recommended: many contracts reuse the same clause numbers.
        """
        result = engine.get_clause(clause, document)
        return _render(result["chunks"], result.get("note", ""), f"Clause lookup: {clause}" + (f" in {document}" if document else ""))

    def find_definition(self, term: str, document: str = "") -> str:
        """Definition of a defined term ("Discovery", "Contract Area", "Petroleum"). Tolerates small spelling mistakes.

        Args:
            term: The defined term.
            document: Document name to look in. Empty searches all documents.
        """
        result = engine.find_definition(term, document)
        return _render(result["chunks"], result.get("note", ""), f"Definition lookup: {term}" + (f" in {document}" if document else ""))

    def read_pages(self, document: str, start_page: int, end_page: int = 0) -> str:
        """Read a document in order from start_page (up to 5 pages). Use to continue a truncated passage, read tables, schedules or letters.

        Args:
            document: Document name or file name.
            start_page: First page (1-based).
            end_page: Last page. Empty or 0 reads one page.
        """
        result = engine.read_pages(document, start_page, end_page)
        header = f"Reading {result.get('document', document)} from page {start_page}"
        if result.get("pages"):
            header += f" (document has {result['pages']} pages)"
        return _render(result["chunks"], result.get("note", ""), header)


# ----------------------------------------------------------------- ingest tools

class LibraryTools(Toolkit):
    def __init__(self, **kwargs: Any):
        super().__init__(name="library", tools=[self.ingest, self.library_status, self.list_documents], **kwargs)

    def ingest(self, path: str = "", force: bool = False) -> str:
        """Index PDF / text files under the corpus folder. Unchanged files are skipped.

        Args:
            path: File or folder relative to the corpus folder. Empty ingests the whole corpus.
            force: Re-index even unchanged files.
        """
        from legal_rag.ingest.pipeline import ingest_path

        try:
            report = ingest_path(path or None, force=force)
        except (ValueError, FileNotFoundError) as exc:
            return f"Error: {exc}"
        lines = [
            f"Added {len(report.added)}, updated {len(report.updated)}, unchanged {len(report.skipped)}, failed {len(report.failed)}.",
            f"Pages {report.pages}, chunks {report.chunks}.",
        ]
        lines += [f"- added: {f}" for f in report.added]
        lines += [f"- updated: {f}" for f in report.updated]
        lines += [f"- FAILED: {f['file']}: {f['error']}" for f in report.failed]
        lines += [f"- warning: {w}" for w in report.warnings]
        return "\n".join(lines)

    def library_status(self) -> str:
        """Counts of documents, pages, chunks and embeddings in the index."""
        s = store.stats()
        return (
            f"{s['documents']} documents, {s['pages']} pages, {s['chunks']} chunks, "
            f"{s['embedded']} embedded with {s['embedding_model'] or 'no model'}; "
            f"{s['empty_pages']} pages without a text layer."
        )

    def list_documents(self, filter: str = "") -> str:
        """List indexed documents.

        Args:
            filter: Optional words to match.
        """
        return LegalResearchTools.list_documents(self, filter)  # type: ignore[arg-type]
