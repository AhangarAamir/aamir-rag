"""Post-hook: turn [C…] tags into a verified References section.

Checks, using only what tools returned in this run:
  - every cited tag was actually retrieved (a tag the model made up is flagged)
  - every quotation of 25+ characters appears verbatim in a cited/retrieved passage
  - a researched answer that cites nothing is flagged
Links come from the index, never from the model.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from agno.run.agent import RunOutput

from legal_rag.index.search import get_chunks

# Tags in another letter ("[B35]", "[S2]") are model inventions.
STRAY_TAG_RE = re.compile(r"\[\s*(?:[A-BD-Z]|C(?=\D))[A-Z]?\s*-?\d+\s*\]")
TAG_RE = re.compile(r"\[\s*C\d+(?:\s*[,;]\s*C?\d+)*\s*\]")
_REF_HEADING = re.compile(r"(?im)^\s*(?:#{1,6}\s*|\*\*)?\s*(references|sources|citations)\s*:?\s*(?:\*\*)?\s*$")
_QUOTE_RE = re.compile(r"[\"“]([^\"”\n]{25,600})[\"”]")
VERIFY_MARK = "<!-- legal-rag:verified -->"


def cited_ids(text: str) -> list[int]:
    ids: list[int] = []
    for tag in TAG_RE.findall(text or ""):
        for num in re.findall(r"\d+", tag):
            value = int(num)
            if value not in ids:
                ids.append(value)
    return ids


def _retrieved_ids(run_output: RunOutput) -> set[int]:
    ids: set[int] = set()
    for tool in run_output.tools or []:
        result = getattr(tool, "result", None)
        if isinstance(result, str):
            ids.update(int(n) for n in re.findall(r"\[C(\d+)\]", result))
    return ids


def _norm(text: str) -> str:
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    text = re.sub(r"\[\s*C\d+[^\]]*\]", " ", text)
    return re.sub(r"[^a-z0-9%]+", " ", text.lower()).strip()


def quote_supported(quote: str, pool: str) -> bool:
    """Each piece between "…" / "..." / [inserted words] must appear verbatim.

    Legal quoting elides with ellipses and adds clarifications in brackets, so
    the quote is checked piece by piece; pieces under 12 characters are skipped.
    """
    cleaned = quote.replace("**", "").replace("__", "")
    pieces = re.split(r"\.{3,}|…|\[[^\]]*\]", cleaned)
    checked = 0
    for piece in pieces:
        p = _norm(piece)
        if len(p) < 12:
            continue
        checked += 1
        if p not in pool:
            return False
    return True


def _strip_reference_section(text: str) -> str:
    lines = text.rstrip().split("\n")
    for index in range(len(lines) - 1, -1, -1):
        if _REF_HEADING.match(lines[index]):
            tail = "\n".join(lines[index + 1:])
            # Only drop it when it is a trailing list of sources.
            if len(tail) < 4000:
                return "\n".join(lines[:index]).rstrip()
            break
    return text


def _pages(hit: dict[str, Any]) -> str:
    return f"p. {hit['page_start']}" if hit["page_start"] == hit["page_end"] else f"pp. {hit['page_start']}–{hit['page_end']}"


def _where(hit: dict[str, Any]) -> str:
    if hit["kind"] == "definition" and hit["defined_term"]:
        return f"definition “{hit['defined_term']}”"
    if hit["clause_id"]:
        return f"clause {hit['clause_id']}"
    if hit["article"]:
        return f"Article {hit['article']}" if hit["article"].isdigit() else hit["article"]
    return hit["kind"]


def build_references(hits: list[dict[str, Any]]) -> str:
    by_doc: dict[int, list[dict[str, Any]]] = {}
    for hit in hits:
        by_doc.setdefault(hit["doc_id"], []).append(hit)
    lines = ["## References", ""]
    for doc_hits in by_doc.values():
        first = doc_hits[0]
        spots = "; ".join(f"{_where(h)}, {_pages(h)} [C{h['id']}]" for h in doc_hits)
        lines.append(f"- [{first['cite_path']}]({first['url']}) — {first['document']}: {spots}")
    return "\n".join(lines)


def verify_answer(text: str, run_output: Optional[RunOutput]) -> tuple[str, dict[str, Any]]:
    text = re.sub(r"(?s)<think>.*?</think>\s*", "", text or "").strip()
    if VERIFY_MARK in text:
        return text, {}
    body = _strip_reference_section(text)
    stray = STRAY_TAG_RE.findall(body)
    if stray:
        body = STRAY_TAG_RE.sub("", body)
    retrieved = _retrieved_ids(run_output) if run_output is not None else set()
    used_tools = bool(run_output and run_output.tools)
    ids = cited_ids(body)
    report: dict[str, Any] = {
        "cited": ids, "retrieved": sorted(retrieved), "unknown": [], "unverified_quotes": [], "stray_tags": stray,
    }
    if not ids and not used_tools:
        return body, report

    unknown = [i for i in ids if i not in retrieved]
    known = [i for i in ids if i in retrieved]
    hits = get_chunks(known)
    report["unknown"] = unknown

    # Quotes may come from indexed passages or from live reads of the source file.
    pool = " ".join(_norm(h["text"]) for h in get_chunks(sorted(retrieved)))
    pool += " " + " ".join(
        _norm(t.result) for t in ((run_output.tools or []) if run_output else [])
        if isinstance(getattr(t, "result", None), str)
    )
    for quote in _QUOTE_RE.findall(body):
        if not quote_supported(quote, pool):
            report["unverified_quotes"].append(quote.strip())

    notes = []
    declined = re.search(r"(?i)insufficient evidence|not indexed|no indexed document|could not find|no matching", body)
    if used_tools and not ids and not declined:
        notes.append("This answer cites no passage. Treat it as unverified.")
    if stray:
        notes.append(f"Removed {len(stray)} malformed citation tag(s): {', '.join(t.strip() for t in stray[:5])}.")
    if unknown:
        notes.append(
            "Citation tag(s) " + ", ".join(f"C{i}" for i in unknown)
            + " were not returned by any search in this answer and may be invented."
        )
    if report["unverified_quotes"]:
        sample = "; ".join(f"“{q[:90]}{'…' if len(q) > 90 else ''}”" for q in report["unverified_quotes"][:3])
        notes.append(f"Quoted text not found verbatim in the retrieved passages: {sample}")

    parts = [body]
    if hits:
        parts.append(build_references(hits))
    if notes:
        parts.append("> **Verification:** " + " ".join(notes))
    parts.append(VERIFY_MARK)
    return "\n\n".join(parts), report


def finalize_answer(run_output: RunOutput) -> None:
    """Agno post-hook."""
    content = run_output.content
    if not isinstance(content, str) or not content.strip():
        return
    new_text, report = verify_answer(content, run_output)
    run_output.content = new_text
    metadata = dict(run_output.metadata or {})
    metadata["verification"] = report
    run_output.metadata = metadata
