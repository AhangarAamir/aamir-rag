"""Hybrid retrieval over the clause index.

Ranked lists fused with reciprocal rank fusion (RRF):
  bm25        FTS5 over chunk text, context header and identifier keys
  bm25+gloss  same, with glossary expansions (PSC <-> Production Sharing Contract)
  vector      cosine similarity on the configured embedding model
  exact       chunks whose clause number / defined term is named in the query

Document names are resolved against each document's aliases. When a named
document is not in the index the caller is told so; the search does not
quietly fall back to a different contract.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

from legal_rag.config import SEARCH_TOP_K
from legal_rag.domain import (
    citation_path,
    citation_url,
    clean_rel_path,
    expansions,
    family_by_code,
    family_codes,
    fix_typos,
    identifier_keys,
    local_url,
    squash,
)
from legal_rag.index import store
from legal_rag.index.vectors import VECTORS

RRF_K = 60
_STOP = frozenset(
    "a an and are as at be by does do did for from has have how in into is it its me of on or per "
    "please show tell than that the their there these this those to under upon was were what when "
    "where which who whom why will with within would can could should shall may about any all also "
    "give list explain describe say says said state states mention mentioned according".split()
)
_GENERIC_DOC_WORDS = frozenset(
    "contract agreement document documents doc pdf file instrument the of as per in for and dated "
    "article clause section definition definitions meaning".split()
)
_CLAUSE_IN_TEXT = re.compile(r"(?<![\d.])(\d{1,3}(?:\.\d{1,3}){1,3})(?![\d.])")
_ARTICLE_IN_TEXT = re.compile(r"(?i)\b(?:article|art\.?|section)\s+(\d{1,3}|[IVXLC]{1,6})\b(?!\.\d)")


@dataclass
class Scope:
    doc_ids: list[int] = field(default_factory=list)
    documents: list[dict[str, Any]] = field(default_factory=list)
    note: str = ""
    unmatched: bool = False


# ------------------------------------------------------------------ helpers

def _words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if w not in _STOP]


def _fts_query(terms: Iterable[str]) -> str:
    seen, parts = set(), []
    for term in terms:
        term = re.sub(r"[^a-z0-9]", "", term.lower())
        if term and term not in seen:
            seen.add(term)
            parts.append(f'"{term}"')
    return " OR ".join(parts)


def _clause_numbers(text: str) -> list[str]:
    out = []
    for match in _CLAUSE_IN_TEXT.finditer(text or ""):
        value = match.group(1)
        # Skip things that look like money / percentages / dates.
        tail = (text[match.end(): match.end() + 2] or "").strip()
        if tail.startswith("%"):
            continue
        if value not in out:
            out.append(value)
    return out


def _scope_sql(doc_ids: Sequence[int], alias: str = "c") -> tuple[str, list[int]]:
    if not doc_ids:
        return "", []
    marks = ",".join("?" * len(doc_ids))
    return f" AND {alias}.doc_id IN ({marks})", list(doc_ids)


# ------------------------------------------------------------------ documents

def resolve_documents(text: str, family: str = "") -> Scope:
    """Map a free-text document reference ("KGD6 PSC", a file name) to documents."""
    text = (text or "").strip()
    docs = store.list_documents()
    fam = family_by_code(family) if family else None
    if fam:
        docs = [d for d in docs if d["family"] == fam.code]
    if not text:
        if fam:
            return Scope([d["id"] for d in docs], docs, f"All {fam.code} documents.")
        return Scope()

    codes = {c.lower() for c in family_codes()}
    raw_words = re.findall(r"[a-z0-9]+", text.lower())
    family_words = {w for w in raw_words if w in codes}
    distinctive = [w for w in raw_words if w not in codes and w not in _GENERIC_DOC_WORDS and w not in _STOP]
    distinctive += [k for k in identifier_keys(text) if k not in distinctive]
    whole = squash(text)

    scored: list[tuple[float, dict[str, Any]]] = []
    for doc in docs:
        keys = set((doc.get("alias_keys") or "").split())
        names = [doc["title"], doc["short_name"], doc["filename"], *doc["aliases"]]
        squashed_names = [squash(n) for n in names if n]
        if whole and any(whole == n or (len(whole) >= 6 and whole in n) for n in squashed_names):
            scored.append((1.2, doc))
            continue
        if not distinctive:
            if family_words and doc["family"].lower() in family_words:
                scored.append((0.5, doc))
            continue
        hits = 0.0
        for word in distinctive:
            if word in keys:
                hits += 1
            elif len(word) >= 4 and difflib.get_close_matches(word, keys, n=1, cutoff=0.85):
                hits += 0.8
        score = hits / len(distinctive)
        if family_words and doc["family"].lower() not in family_words and not (family_words & keys):
            score *= 0.6
        if score > 0:
            scored.append((score, doc))

    if not scored:
        names = ", ".join(sorted({d["short_name"] or d["title"] for d in store.list_documents()})[:25])
        return Scope(
            note=f'No indexed document matches "{text}". Indexed documents include: {names or "none"}.',
            unmatched=True,
        )
    best = max(s for s, _ in scored)
    if best < 0.5:
        close = ", ".join(d["short_name"] or d["title"] for s, d in sorted(scored, key=lambda x: -x[0])[:5])
        return Scope(
            note=f'No indexed document clearly matches "{text}". Closest: {close}. '
                 "Do not answer from a different document without saying so.",
            unmatched=True,
        )
    keep = [d for s, d in scored if s >= best * 0.8]
    keep.sort(key=lambda d: d["rel_path"])
    return Scope([d["id"] for d in keep], keep)


# ------------------------------------------------------------------ hits

_HIT_SQL = """SELECT c.*, d.rel_path, d.filename, d.link, d.title, d.short_name, d.family, d.status, d.doc_date
              FROM chunks c JOIN documents d ON d.id = c.doc_id"""


def _hit(row: Any, score: float = 0.0, matched: Sequence[str] = ()) -> dict[str, Any]:
    rel = row["rel_path"]
    return {
        "id": int(row["id"]),
        "doc_id": int(row["doc_id"]),
        "document": row["short_name"] or row["title"] or row["filename"],
        "title": row["title"],
        "filename": row["filename"],
        "path": clean_rel_path(rel),
        "cite_path": citation_path(rel),
        "family": row["family"],
        "status": row["status"],
        "doc_date": row["doc_date"],
        "kind": row["kind"],
        "article": row["article"],
        "article_title": row["article_title"],
        "clause_id": row["clause_id"],
        "heading": row["heading"],
        "defined_term": row["defined_term"],
        "page_start": int(row["page_start"]),
        "page_end": int(row["page_end"]),
        "part": int(row["part"]),
        "parts": int(row["parts"]),
        "context": row["context"],
        "text": row["text"],
        "link": row["link"] or citation_url(rel),
        "url": row["link"] or citation_url(rel, int(row["page_start"])),
        "local_url": local_url(rel, int(row["page_start"])),
        "score": round(score, 5),
        "matched": list(matched),
    }


def get_chunks(ids: Sequence[int]) -> list[dict[str, Any]]:
    if not ids:
        return []
    marks = ",".join("?" * len(ids))
    rows = store.query(f"{_HIT_SQL} WHERE c.id IN ({marks})", list(ids))
    by_id = {int(r["id"]): _hit(r) for r in rows}
    return [by_id[i] for i in ids if i in by_id]


def _bm25(fts: str, doc_ids: Sequence[int], limit: int = 60) -> list[int]:
    if not fts:
        return []
    scope, params = _scope_sql(doc_ids)
    try:
        rows = store.query(
            f"""SELECT c.id FROM chunks_fts f JOIN chunks c ON c.id = f.rowid
                WHERE chunks_fts MATCH ? {scope}
                ORDER BY bm25(chunks_fts, 1.0, 2.5, 3.0) LIMIT ?""",
            [fts, *params, limit],
        )
    except Exception:
        return []
    return [int(r["id"]) for r in rows]


def _exact_clause(clauses: Sequence[str], doc_ids: Sequence[int]) -> list[int]:
    if not clauses:
        return []
    scope, params = _scope_sql(doc_ids)
    marks = ",".join("?" * len(clauses))
    rows = store.query(
        f"SELECT c.id FROM chunks c WHERE c.clause_id IN ({marks}) {scope} ORDER BY c.part, c.doc_id LIMIT 40",
        [*clauses, *params],
    )
    return [int(r["id"]) for r in rows]


_TERMS_CACHE: dict[str, Any] = {"version": -1, "terms": []}


def _defined_terms() -> list[str]:
    version = store.index_version()
    if _TERMS_CACHE["version"] != version:
        rows = store.query("SELECT DISTINCT defined_term FROM chunks WHERE defined_term != ''")
        _TERMS_CACHE["terms"] = sorted({r["defined_term"] for r in rows}, key=len, reverse=True)
        _TERMS_CACHE["version"] = version
    return _TERMS_CACHE["terms"]


def _terms_in_query(query: str) -> list[str]:
    """Defined terms named in the query, longest first ("Discovery Area" beats "Discovery")."""
    found: list[str] = []
    spans: list[tuple[int, int]] = []
    for term in _defined_terms():
        if len(term) < 4:
            continue
        pattern = r"(?<![A-Za-z0-9])" + re.escape(term).replace(r"\ ", r"[\s-]+") + r"(?![A-Za-z0-9])"
        for m in re.finditer(pattern, query, re.IGNORECASE):
            if any(m.start() < e and s < m.end() for s, e in spans):
                continue
            spans.append((m.start(), m.end()))
            if term not in found:
                found.append(term)
    quoted = re.findall(r"[\"“']([^\"”']{2,60})[\"”']", query)
    m = re.search(r"(?i)(?:definition of|meaning of|define[sd]?|what is meant by)\s+(?:the\s+)?(?:term\s+)?([A-Za-z][A-Za-z \-]{1,60})", query)
    if m:
        quoted.append(re.split(r"(?i)\b(?:in|under|as per|per|according)\b", m.group(1))[0])
    for q in quoted:
        q = q.strip(" ?.,")
        if q and q.lower() not in {f.lower() for f in found}:
            found.append(q)
    return found


def _exact_terms(terms: Sequence[str], doc_ids: Sequence[int]) -> list[int]:
    if not terms:
        return []
    scope, params = _scope_sql(doc_ids)
    ids: list[int] = []
    for term in terms:
        rows = store.query(
            f"SELECT c.id FROM chunks c WHERE c.defined_term = ? COLLATE NOCASE {scope} LIMIT 10",
            [term, *params],
        )
        ids.extend(int(r["id"]) for r in rows if int(r["id"]) not in ids)
    return ids


def _phrase_query(query: str) -> str:
    """Adjacent content-word pairs as FTS phrases: "participating interest"."""
    tokens = [w for w in re.findall(r"[a-z0-9]+", query.lower())]
    phrases = []
    for a, b in zip(tokens, tokens[1:]):
        if a in _STOP or b in _STOP or a.isdigit() or b.isdigit():
            continue
        phrase = f'"{a} {b}"'
        if phrase not in phrases:
            phrases.append(phrase)
    return " OR ".join(phrases)


def _coverage(text: str, words: Sequence[str]) -> float:
    if not words:
        return 0.0
    hay = set(re.findall(r"[a-z0-9]+", text.lower()))
    stems = {w[:5] for w in hay}
    hit = sum(1 for w in words if w in hay or w[:5] in stems)
    return hit / len(words)


def search(
    query: str,
    *,
    document: str = "",
    family: str = "",
    top_k: int = SEARCH_TOP_K,
    use_vectors: bool = True,
) -> dict[str, Any]:
    query = fix_typos((query or "").strip())
    scope = resolve_documents(document, family) if (document or family) else Scope()
    if (document or family) and scope.unmatched:
        return {"hits": [], "note": scope.note, "scope": [], "query": query}

    doc_ids = scope.doc_ids
    clauses = _clause_numbers(query)
    articles = [m.group(1) for m in _ARTICLE_IN_TEXT.finditer(query)]
    words = _words(query)
    keys = list(identifier_keys(query)) + ["cl" + c.replace(".", "p") for c in clauses]
    keys += ["art" + a.lower() for a in articles]
    expanded = expansions(query)

    lists: list[tuple[str, float, list[int]]] = []
    lists.append(("bm25", 1.0, _bm25(_fts_query([*words, *keys]), doc_ids)))
    if expanded:
        extra = [w for e in expanded for w in _words(e)]
        lists.append(("bm25+glossary", 0.7, _bm25(_fts_query([*words, *keys, *extra]), doc_ids)))
    if use_vectors:
        vec_query = query if not expanded else f"{query} ({'; '.join(expanded[:6])})"
        lists.append(("vector", 1.0, [cid for cid, _ in VECTORS.search(vec_query, 60, doc_ids)]))
    phrase_text = query + (" " + " ".join(expanded) if expanded else "")
    phrases = _phrase_query(phrase_text)
    if phrases:
        lists.append(("phrase", 0.8, _bm25(phrases, doc_ids)))
    exact = _exact_clause(clauses, doc_ids)
    if exact:
        lists.append(("clause", 1.6, exact))
    named_terms = _terms_in_query(query)
    term_ids = _exact_terms(named_terms, doc_ids)
    if term_ids:
        definitional = bool(re.search(r"(?i)defin|mean|what is|what are|\bterm\b", query))
        lists.append(("definition", 1.6 if definitional else 0.6, term_ids))

    scores: dict[int, float] = {}
    matched: dict[int, list[str]] = {}
    for name, weight, ids in lists:
        for rank, cid in enumerate(ids):
            scores[cid] = scores.get(cid, 0.0) + weight / (RRF_K + rank + 1)
            matched.setdefault(cid, []).append(name)
    if not scores:
        note = "No matching text."
        if scope.documents:
            note += " Searched only: " + ", ".join(d["short_name"] or d["title"] for d in scope.documents)
        return {"hits": [], "note": note, "scope": _scope_view(scope), "query": query}

    candidates = sorted(scores, key=lambda c: -scores[c])[: max(top_k * 4, 30)]
    hits = get_chunks(candidates)
    wants_toc = bool(re.search(r"(?i)\b(contents|index|list of articles)\b", query))
    for hit in hits:
        s = scores[hit["id"]]
        if hit["kind"] == "toc" and not wants_toc:
            s *= 0.3
        if hit["kind"] == "definition" and re.search(r"(?i)defin|mean", query):
            s *= 1.15
        # Passages that contain more of the question's words rank higher.
        s *= 0.85 + 0.3 * _coverage(hit["context"] + " " + hit["text"], words)
        hit["score"] = round(s * 1000, 3)
        hit["matched"] = matched.get(hit["id"], [])
    hits.sort(key=lambda h: -h["score"])
    return {"hits": hits[:top_k], "note": scope.note, "scope": _scope_view(scope), "query": query, "expanded": expanded}


def _scope_view(scope: Scope) -> list[dict[str, Any]]:
    return [
        {"id": d["id"], "document": d["short_name"] or d["title"], "filename": d["filename"], "family": d["family"]}
        for d in scope.documents
    ]


# ------------------------------------------------------------------ exact lookups

def get_clause(clause: str, document: str = "", max_chars: int = 14000) -> dict[str, Any]:
    clause = (clause or "").strip()
    scope = resolve_documents(document) if document else Scope()
    if document and scope.unmatched:
        return {"chunks": [], "note": scope.note}
    doc_ids = scope.doc_ids

    match = re.fullmatch(r"(?i)(?:article|art\.?|clause|section)?\s*(\d{1,3}[A-Z]?(?:\.\d{1,3}){0,3})\s*\.?", clause)
    if match:
        # "article 31a" -> stored as "31A"
        match_value = match.group(1).upper()
        clause = clause[: match.start(1)] + match_value + clause[match.end(1):]
        match = re.fullmatch(r"(?i)(?:article|art\.?|clause|section)?\s*(\d{1,3}[A-Z]?(?:\.\d{1,3}){0,3})\s*\.?", clause)
    part = re.fullmatch(r"(?i)(schedule|annexure|annex|appendix|exhibit)\s+([A-Z0-9]{1,3}(?:-[A-Z0-9]{1,3})?)", clause)
    scope_sql, params = _scope_sql(doc_ids)
    if match and "." in match.group(1):
        number = match.group(1)
        rows = store.query(
            f"{_HIT_SQL} WHERE (c.clause_id = ? OR c.clause_id LIKE ?) {scope_sql} ORDER BY c.doc_id, c.ord",
            [number, number + ".%", *params],
        )
        label = f"clause {number}"
    elif match:
        number = match.group(1)
        rows = store.query(
            f"{_HIT_SQL} WHERE c.article = ? AND c.kind != 'toc' {scope_sql} ORDER BY c.doc_id, c.ord",
            [number, *params],
        )
        label = f"Article {number}"
    elif part:
        name = f"{part.group(1).title()} {part.group(2).upper()}"
        rows = store.query(
            f"{_HIT_SQL} WHERE c.article = ? COLLATE NOCASE {scope_sql} ORDER BY c.doc_id, c.ord",
            [name, *params],
        )
        label = name
    else:
        return {"chunks": [], "note": f'"{clause}" is not a clause number (e.g. 10.7), article (10) or part (Appendix C).'}

    if not rows:
        where = " in " + ", ".join(d["short_name"] or d["title"] for d in scope.documents) if scope.documents else ""
        return {"chunks": [], "note": f"No {label} found{where}. Try search_documents with the heading words."}

    by_doc: dict[int, list] = {}
    for row in rows:
        by_doc.setdefault(int(row["doc_id"]), []).append(row)
    note = ""
    if len(by_doc) > 1 and not document:
        names = []
        for rows_ in by_doc.values():
            names.append(rows_[0]["short_name"] or rows_[0]["title"] or rows_[0]["filename"])
        note = (
            f"{label} exists in {len(by_doc)} documents: {'; '.join(names[:12])}. "
            "Showing the first ones. Pass document= to pick the one the user means."
        )
    out, used = [], 0
    for doc_rows in list(by_doc.values())[: (3 if not document else 6)]:
        for row in doc_rows:
            if used + len(row["text"]) > max_chars and out:
                note = (note + " " if note else "") + f"Truncated at {max_chars} characters; ask for a specific clause to read the rest."
                break
            out.append(_hit(row))
            used += len(row["text"])
    return {"chunks": out, "note": note}


def find_definition(term: str, document: str = "") -> dict[str, Any]:
    term = (term or "").strip().strip("\"“”'")
    scope = resolve_documents(document) if document else Scope()
    if document and scope.unmatched:
        return {"chunks": [], "note": scope.note}
    scope_sql, params = _scope_sql(scope.doc_ids)
    rows = store.query(
        f"{_HIT_SQL} WHERE c.defined_term = ? COLLATE NOCASE {scope_sql} ORDER BY c.doc_id, c.ord",
        [term, *params],
    )
    note = ""
    if not rows:
        all_terms = store.query(
            f"SELECT DISTINCT c.defined_term FROM chunks c WHERE c.defined_term != '' {scope_sql}", params
        )
        names = [r["defined_term"] for r in all_terms]
        lowered = {n.lower(): n for n in names}
        close = difflib.get_close_matches(term.lower(), list(lowered), n=3, cutoff=0.75)
        contains = [n for n in names if term.lower() in n.lower()][:3]
        picks = [lowered[c] for c in close] + [c for c in contains if c not in close]
        if picks:
            marks = ",".join("?" * len(picks))
            rows = store.query(
                f"{_HIT_SQL} WHERE c.defined_term IN ({marks}) {scope_sql} ORDER BY c.doc_id, c.ord",
                [*picks, *params],
            )
            note = f'No exact definition of "{term}". Closest defined terms: {", ".join(picks)}.'
    if not rows:
        found = search(f'"{term}" means', document=document, top_k=5)
        return {
            "chunks": found["hits"],
            "note": f'"{term}" is not a defined term in the indexed definitions. Showing text that uses it.',
        }
    return {"chunks": [_hit(r) for r in rows[:8]], "note": note}


def read_pages(document: str, start_page: int, end_page: int = 0, max_chars: int = 12000) -> dict[str, Any]:
    scope = resolve_documents(document)
    if scope.unmatched or not scope.doc_ids:
        return {"chunks": [], "note": scope.note or "Name the document to read."}
    doc = scope.documents[0]
    start_page = max(1, int(start_page or 1))
    end_page = max(start_page, int(end_page or start_page))
    end_page = min(end_page, start_page + 4)
    rows = store.query(
        f"{_HIT_SQL} WHERE c.doc_id = ? AND c.page_end >= ? AND c.page_start <= ? ORDER BY c.ord",
        [doc["id"], start_page, end_page],
    )
    out, used, note = [], 0, ""
    for row in rows:
        if used + len(row["text"]) > max_chars and out:
            note = f"Stopped at {max_chars} characters. Read from page {row['page_start']} to continue."
            break
        out.append(_hit(row))
        used += len(row["text"])
    if len(scope.documents) > 1:
        note = (note + " " if note else "") + f"Several documents matched; read {doc['short_name'] or doc['title']}."
    return {"chunks": out, "note": note, "document": doc["short_name"] or doc["title"], "pages": doc["pages"]}
