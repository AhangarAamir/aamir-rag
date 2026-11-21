"""Post-hook: turn [C…] tags into a verified References section.

Checks, using only what tools returned in this run:
  - every cited tag was actually retrieved (a tag the model made up is flagged)
  - every quotation of 25+ characters appears verbatim in a cited/retrieved passage
  - a researched answer that cites nothing is flagged
Links come from the index, never from the model. Each reference opens the
document at the page of the cited text and searches for that text: the quote
the answer took from the passage, or else the passage's first line on each
page it spans. One document can therefore have several references.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from agno.run.agent import RunOutput

from legal_rag.domain import page_link, search_snippet
from legal_rag.index import store
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


def _where(hit: dict[str, Any]) -> str:
    if hit["kind"] == "definition" and hit["defined_term"]:
        return f"definition “{hit['defined_term']}”"
    if hit["clause_id"]:
        return f"clause {hit['clause_id']}"
    if hit["article"]:
        return f"Article {hit['article']}" if hit["article"].isdigit() else hit["article"]
    return hit["kind"]


# --------------------------------------------------------------- references

_MIN_LINE = 20


def _page_texts(hit: dict[str, Any]) -> dict[int, str]:
    rows = store.query(
        "SELECT page_no, text FROM pages WHERE doc_id=? AND page_no BETWEEN ? AND ? ORDER BY page_no",
        (hit["doc_id"], hit["page_start"], hit["page_end"]),
    )
    return {int(r["page_no"]): r["text"] for r in rows}


def quotes_by_tag(body: str) -> dict[int, list[str]]:
    """Quotes in the answer, keyed by the [C…] tags that follow them in the same line."""
    found: dict[int, list[str]] = {}
    for match in _QUOTE_RE.finditer(body):
        tail = body[match.end():match.end() + 400].split("\n", 1)[0]
        tag = TAG_RE.search(tail)
        if not tag:
            continue
        # Only the tag that closes this quote's sentence, not one after a later quote.
        if _QUOTE_RE.search(tail[:tag.start()]):
            continue
        for num in re.findall(r"\d+", tag.group(0)):
            found.setdefault(int(num), []).append(match.group(1))
    return found


def _quote_target(quote: str, hit: dict[str, Any], pages: dict[int, str]) -> Optional[tuple[int, str]]:
    """(page, search text) for a quote that really comes from this passage."""
    pieces = [p for p in re.split(r"\.{3,}|…|\[[^\]]*\]", quote.replace("**", "")) if len(_norm(p)) >= 12]
    if not pieces or _norm(pieces[0]) not in _norm(hit["text"]):
        return None
    piece = pieces[0]
    for page_no, text in pages.items():
        if _norm(piece) in _norm(text):
            return page_no, piece
    # Quote runs across a page break: open the page where its first words are.
    words = search_snippet(piece, 4)
    for page_no, text in pages.items():
        if words and _norm(words) in _norm(text):
            return page_no, words
    return hit["page_start"], piece


def _first_line_on_page(hit_norm: str, page_text: str) -> str:
    """First line of the page that belongs to this passage."""
    for line in page_text.split("\n"):
        line = line.strip()
        norm = _norm(line)
        if len(norm) >= _MIN_LINE and norm in hit_norm:
            return line
    return ""


def reference_targets(hit: dict[str, Any], quotes: list[str]) -> list[dict[str, Any]]:
    """Where a cited passage should open: one target per page (or per quote)."""
    pages = _page_texts(hit)
    targets: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()

    def add(page_no: int, text: str, continued: bool) -> None:
        snippet = search_snippet(text)
        key = (page_no, _norm(snippet))
        if key in seen:
            return
        seen.add(key)
        targets.append({"page": page_no, "snippet": snippet, "continued": continued})

    for quote in quotes:
        target = _quote_target(quote, hit, pages)
        if target:
            add(target[0], target[1], target[0] != hit["page_start"])
    if targets:
        return sorted(targets, key=lambda t: t["page"])

    hit_norm = _norm(hit["text"])
    for page_no in range(hit["page_start"], hit["page_end"] + 1):
        line = _first_line_on_page(hit_norm, pages.get(page_no, ""))
        if line or page_no == hit["page_start"]:
            add(page_no, line, page_no != hit["page_start"])
    return targets


def build_references(hits: list[dict[str, Any]], quotes: Optional[dict[int, list[str]]] = None) -> str:
    """One line per cited passage and page, grouped by document in citation order."""
    quotes = quotes or {}
    by_doc: dict[int, list[dict[str, Any]]] = {}
    for hit in hits:
        by_doc.setdefault(hit["doc_id"], []).append(hit)
    lines = ["## References", ""]
    for doc_hits in by_doc.values():
        rows = []
        for hit in doc_hits:
            for target in reference_targets(hit, quotes.get(hit["id"], [])):
                rows.append((target["page"], hit, target))
        rows.sort(key=lambda r: (r[0], r[1]["id"]))
        for page_no, hit, target in rows:
            url = page_link(hit["link"], page_no, target["snippet"])
            where = _where(hit) + (" (continued)" if target["continued"] else "")
            label = f"{hit['cite_path']}, p. {page_no}"
            text = f" — “{target['snippet']}…”" if target["snippet"] else ""
            lines.append(f"- [{label}]({url}) — {hit['document']}: {where}{text} [C{hit['id']}]")
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
        parts.append(build_references(hits, quotes_by_tag(body)))
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
