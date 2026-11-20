"""Split legal text into retrieval units that follow the document's own structure.

Unit boundaries, in order of strength:
  ARTICLE / SECTION headings  -> top-level part (article)
  SCHEDULE / ANNEX / APPENDIX -> top-level part (resets clause numbering)
  numbered clauses 10.7, 10.7.1 -> one unit each, with all (a)/(i) items
  definition entries         -> one unit each ("Discovery" means ...)
  table of contents pages    -> marked toc, ranked low at search time

Numbered lines are accepted as clauses only when they agree with the current
article and move forward. That rejects lines such as "10.2 above" or
"12.5 per cent" that a line wrap happened to put at the start of a line.
No model call is involved, so the result is the same for every LLM.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass, field
from typing import Optional

_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}

_TOP_RE = re.compile(
    r"^[ \t]*(?P<kw>ARTICLE|Article|SECTION|Section|CLAUSE|Clause)[ \t]+"
    r"(?P<num>\d{1,3}|[IVXLC]{1,7}|[0-9IlO]{1,3})\b\.?[ \t]*(?P<sep>[-–—:.]?)[ \t]*(?P<rest>.*)$"
)
_PART_RE = re.compile(
    r"^[ \t]*(?P<kw>SCHEDULE|Schedule|ANNEXURE|Annexure|ANNEX|Annex|APPENDIX|Appendix|EXHIBIT|Exhibit)"
    r"[ \t]+(?P<num>[A-Z]{1,3}|\d{1,3}|[IVX]{1,5})(?:-(?P<sub>[A-Z0-9]{1,3}))?\b\.?[ \t]*[-–—:.]?[ \t]*(?P<rest>.*)$"
)
_NUM_TOP_RE = re.compile(r"^[ \t]*(?P<num>\d{1,2})\.?[ \t]+(?P<rest>[A-Z][A-Z0-9 ,&/()'’\-]{3,80})[ \t]*$")
_CLAUSE_RE = re.compile(r"^[ \t]*(?P<num>\d{1,3}(?:\.\d{1,3}){1,3})\.?(?:[ \t]+(?P<rest>.*))?$")
_DEF_RE = re.compile(
    r"^[ \t]*(?:(?P<num>\d{1,3}(?:\.\d{1,3}){1,3})\.?[ \t]+)?(?:\([a-z0-9]{1,4}\)[ \t]+)?"
    r"[\"“‘'](?P<term>[^\"”’'\n]{2,90})[\"”’'][ \t]*(?:\(.*?\)[ \t]*)?,?[ \t]*"
    r"(?:shall[ \t]+)?(?:means?|includes?|has[ \t]+the[ \t]+meaning|have[ \t]+the[ \t]+meaning|refers?[ \t]+to|shall[ \t]+have)\b"
)
_CONTINUATION_WORDS = re.compile(
    r"^(?:and|or|of|to|in|on|at|by|for|with|above|below|hereof|herein|hereto|thereof|"
    r"shall|may|is|are|was|were|be|has|have|per|percent|which|that|the|a|an|as|from)\b|^%",
)
# A previous line ending like this means the next line continues a reference
# ("... under Article\n10.6 and ...") rather than starting a clause.
_DANGLING = re.compile(
    r"(?i)\b(?:article|articles|clause|clauses|section|sections|paragraph|paragraphs|sub-clause|"
    r"schedule|annex|appendix|and|or|of|under|in|to|with|per|by|at|than|from|see)\s*$"
)
_SUBITEM_RE = re.compile(r"^[ \t]*\((?:[a-z]{1,2}|[ivxl]{1,5}|\d{1,2})\)[ \t]")
_TOC_LINE_RE = re.compile(r"(?:\.{3,}|…{2,}|\s{2,}|\t)\s*\d{1,4}\s*$|^\s*\S.{2,90}?\s\d{1,4}\s*$")


@dataclass
class Unit:
    kind: str  # clause | definition | article | part | preamble | toc | text
    text: str
    char_start: int
    char_end: int
    article: str = ""
    article_title: str = ""
    clause_id: str = ""
    heading: str = ""
    defined_term: str = ""
    page_start: int = 1
    page_end: int = 1
    part: int = 1
    parts: int = 1
    # Heading path for Markdown: "PART III › 14. Equality before the law › 14.2 …"
    breadcrumb: str = ""


@dataclass
class Outline:
    units: list[Unit] = field(default_factory=list)
    articles: list[dict] = field(default_factory=list)


# ------------------------------------------------------------------ helpers

def _to_int(token: str) -> Optional[int]:
    if token.isdigit():
        return int(token)
    if any(ch.isdigit() for ch in token) or "O" in token or "l" in token:
        fixed = token.replace("I", "1").replace("l", "1").replace("O", "0")
        return int(fixed) if fixed.isdigit() else None
    if all(ch in _ROMAN for ch in token):
        total, prev = 0, 0
        for ch in reversed(token):
            value = _ROMAN[ch]
            total = total - value if value < prev else total + value
            prev = max(prev, value)
        return total
    return None


def _tuple(clause: str) -> tuple[int, ...]:
    return tuple(int(p) for p in clause.split("."))


def _heading_like(text: str) -> bool:
    text = text.strip()
    if not text or len(text) > 110:
        return False
    if _CONTINUATION_WORDS.match(text):
        return False
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    upper_ratio = sum(c.isupper() for c in letters) / len(letters)
    words = [w for w in re.split(r"\s+", text) if w]
    capitalized = sum(1 for w in words if w[:1].isupper() or not w[:1].isalpha()) / max(1, len(words))
    return upper_ratio > 0.6 or (capitalized >= 0.6 and not text.endswith((",", ";")))


def _clause_heading(rest: str) -> str:
    rest = (rest or "").strip()
    if not rest:
        return ""
    match = re.match(r"^([A-Z][^.:\n]{1,90}?)[.:][ \t]+(?=[A-Z(“\"'])", rest)
    if match:
        candidate = match.group(1).strip()
        # "Right of retention. The Contractor ..." — short, no verb, no comma.
        short_phrase = (
            len(candidate) <= 70
            and len(candidate.split()) <= 9
            and "," not in candidate
            and not re.search(r"\b(shall|may|will|must|is|are|was|has|have|be)\b", candidate)
        )
        if short_phrase or _heading_like(candidate):
            return candidate
    if len(rest) <= 90 and _heading_like(rest) and not rest.endswith("."):
        return rest
    return ""


def _prev_line(lines: list[str], index: int) -> Optional[str]:
    """Previous non-empty line, or None when a blank line / start precedes."""
    j = index - 1
    if j < 0 or not lines[j].strip():
        return None
    return lines[j].strip()


def _starts_block(lines: list[str], index: int) -> bool:
    prev = _prev_line(lines, index)
    if prev is None:
        return True
    if _DANGLING.search(prev):
        return False
    return prev.endswith((".", ":", ";", ")", '"', "”")) or _heading_like(prev)


def _is_toc_page(text: str) -> bool:
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if len(lines) < 6:
        return False
    head = " ".join(lines[:8]).lower()
    hits = sum(1 for ln in lines if _TOC_LINE_RE.search(ln) and len(ln) < 120)
    ratio = hits / len(lines)
    if re.search(r"\b(table of )?contents\b|\bindex\b", head) and ratio > 0.25:
        return True
    dotted = sum(1 for ln in lines if re.search(r"\.{4,}|…{2,}", ln))
    return dotted / len(lines) > 0.3


# ------------------------------------------------------------------ parsing

def parse(pages: list[str], max_chars: int = 2400) -> Outline:
    page_starts: list[int] = []
    buf: list[str] = []
    offset = 0
    for text in pages:
        page_starts.append(offset)
        buf.append(text)
        offset += len(text) + 1
    full = "\n".join(buf)
    toc_pages = _toc_pages(pages)

    def page_of(pos: int) -> int:
        return max(1, bisect.bisect_right(page_starts, pos))

    # Which keyword marks top-level parts in this document?
    counts = {"article": 0, "section": 0, "clause": 0}
    for match in re.finditer(r"(?m)^[ \t]*(ARTICLE|Article|SECTION|Section|CLAUSE|Clause)[ \t]+[0-9IVXLClO]{1,7}\b", full):
        counts[match.group(1).lower()] += 1
    top_kw = max(counts, key=lambda k: counts[k]) if any(counts.values()) else ""
    numeric_tops = not top_kw

    boundaries: list[dict] = []
    article_no: Optional[int] = None
    article_label = ""
    article_title = ""
    in_part = False
    in_definitions = False
    last_clause: tuple[int, ...] = ()
    skip_line_at = -1

    lines = full.split("\n")
    pos = 0
    line_offsets = []
    for line in lines:
        line_offsets.append(pos)
        pos += len(line) + 1

    i = 0
    while i < len(lines):
        line = lines[i]
        start = line_offsets[i]
        page = page_of(start)
        if page in toc_pages or start == skip_line_at or not line.strip():
            i += 1
            continue

        top = _TOP_RE.match(line)
        if top and top.group("kw").lower() == top_kw:
            number = _to_int(top.group("num"))
            rest = top.group("rest").strip()
            upper = top.group("kw").isupper()
            prev = _prev_line(lines, i)
            if upper:
                heading_ok = not rest or bool(top.group("sep")) or _heading_like(rest)
                heading_ok = heading_ok and not (prev and _DANGLING.search(prev))
            else:
                # "Article 21." at a line start is usually a wrapped reference.
                heading_ok = _starts_block(lines, i) and (
                    (not rest and not line.rstrip().endswith("."))
                    or (bool(top.group("sep")) and bool(rest))
                    or _heading_like(rest)
                )
            if number is not None and heading_ok and not (rest and _CONTINUATION_WORDS.match(rest)):
                title = rest
                if not title:
                    title, consumed = _next_title(lines, i)
                    if consumed:
                        skip_line_at = line_offsets[consumed]
                article_no = number
                article_label = str(number)
                article_title = _strip_title(title)
                in_part = False
                in_definitions = bool(re.search(r"(?i)definition|interpretation", article_title))
                last_clause = ()
                boundaries.append(
                    dict(kind="article", start=start, article=article_label, article_title=article_title)
                )
                i += 1
                continue

        part = _PART_RE.match(line)
        if part and (part.group("kw").isupper() or _heading_like(part.group("rest") or "") or not part.group("rest")):
            rest = (part.group("rest") or "").strip()
            if not (rest and _CONTINUATION_WORDS.match(rest)):
                label = f"{part.group('kw').title()} {part.group('num')}" + (
                    f"-{part.group('sub')}" if part.group("sub") else ""
                )
                title = rest
                if not title:
                    title, consumed = _next_title(lines, i)
                    if consumed:
                        skip_line_at = line_offsets[consumed]
                article_no = None
                article_label = label
                article_title = _strip_title(title)
                in_part = True
                in_definitions = bool(re.search(r"(?i)definition", article_title))
                last_clause = ()
                boundaries.append(dict(kind="part", start=start, article=label, article_title=article_title))
                i += 1
                continue

        if numeric_tops:
            numtop = _NUM_TOP_RE.match(line)
            if numtop:
                number = int(numtop.group("num"))
                if article_no is None or number > article_no:
                    article_no = number
                    article_label = str(number)
                    article_title = _strip_title(numtop.group("rest"))
                    in_part = False
                    in_definitions = bool(re.search(r"(?i)definition|interpretation", article_title))
                    last_clause = ()
                    boundaries.append(
                        dict(kind="article", start=start, article=article_label, article_title=article_title)
                    )
                    i += 1
                    continue

        definition = _DEF_RE.match(line)
        clause = _CLAUSE_RE.match(line)
        clause_ok = False
        if clause:
            number = clause.group("num")
            rest = (clause.group("rest") or "").strip()
            numbers = _tuple(number)
            agrees = in_part or article_no is None or numbers[0] == article_no
            forward = not last_clause or numbers > last_clause
            text_ok = not rest or not (_CONTINUATION_WORDS.match(rest) or rest[:1].isdigit())
            prev = _prev_line(lines, i)
            not_dangling = not (prev and _DANGLING.search(prev))
            clause_ok = agrees and forward and text_ok and not_dangling and len(numbers) >= 2
            if clause_ok:
                last_clause = numbers
                term = definition.group("term").strip() if definition and definition.group("num") == number else ""
                label = article_label or str(numbers[0])
                boundaries.append(
                    dict(
                        kind="definition" if term else "clause",
                        start=start,
                        article=label,
                        article_title=article_title,
                        clause_id=number,
                        heading=term or _clause_heading(rest),
                        defined_term=term,
                    )
                )
                i += 1
                continue

        if definition and not definition.group("num") and in_definitions:
            term = definition.group("term").strip()
            boundaries.append(
                dict(
                    kind="definition",
                    start=start,
                    article=article_label,
                    article_title=article_title,
                    clause_id=".".join(map(str, last_clause)) if last_clause else "",
                    heading=term,
                    defined_term=term,
                )
            )
        i += 1

    units = _build_units(full, boundaries, toc_pages, page_starts, page_of, max_chars)
    articles = _articles(boundaries, full, page_of)
    return Outline(units=units, articles=articles)


def _strip_title(title: str) -> str:
    title = re.sub(r"\s+", " ", (title or "")).strip(" .:-–—")
    return title[:120]


def _next_title(lines: list[str], index: int) -> tuple[str, int]:
    for j in range(index + 1, min(index + 4, len(lines))):
        candidate = lines[j].strip()
        if not candidate:
            continue
        if _CLAUSE_RE.match(candidate) or _TOP_RE.match(candidate) or len(candidate) > 90:
            return "", 0
        if _heading_like(candidate) and not candidate.endswith((".", ",", ";")):
            return candidate, j
        return "", 0
    return "", 0


def _toc_pages(pages: list[str]) -> set[int]:
    toc: set[int] = set()
    for number, text in enumerate(pages, start=1):
        if number > max(12, len(pages) // 3):
            break
        if _is_toc_page(text):
            toc.add(number)
    return toc


def _articles(boundaries: list[dict], full: str, page_of) -> list[dict]:
    tops = [b for b in boundaries if b["kind"] in {"article", "part"}]
    out = []
    for idx, b in enumerate(tops):
        end = tops[idx + 1]["start"] if idx + 1 < len(tops) else len(full)
        out.append(
            dict(
                article=b["article"],
                title=b["article_title"],
                char_start=b["start"],
                char_end=end,
                page_start=page_of(b["start"]),
                page_end=page_of(max(b["start"], end - 1)),
            )
        )
    return out


def _build_units(full, boundaries, toc_pages, page_starts, page_of, max_chars) -> list[Unit]:
    units: list[Unit] = []
    # Table of contents pages become their own low-priority units.
    for page in sorted(toc_pages):
        start = page_starts[page - 1]
        end = page_starts[page] - 1 if page < len(page_starts) else len(full)
        text = full[start:end].strip()
        if text:
            units.append(Unit("toc", text, start, end, heading="Table of contents", page_start=page, page_end=page))

    if not boundaries:
        body_start = 0
        units.extend(_pack_text(full, body_start, len(full), "text", toc_pages, page_starts, page_of, max_chars))
        return units

    first = boundaries[0]["start"]
    if first > 0:
        units.extend(_pack_text(full, 0, first, "preamble", toc_pages, page_starts, page_of, max_chars))

    for idx, b in enumerate(boundaries):
        end = boundaries[idx + 1]["start"] if idx + 1 < len(boundaries) else len(full)
        raw = full[b["start"]:end]
        text = raw.strip()
        if not text:
            continue
        if b["kind"] in {"article", "part"}:
            # Heading lines alone carry no content; their title lives in every
            # child chunk's context header.
            body_lines = [ln for ln in text.split("\n")[1:] if ln.strip()]
            body = "\n".join(body_lines)
            if len(body) < 40 and idx + 1 < len(boundaries):
                continue
        base = Unit(
            kind=b["kind"],
            text=text,
            char_start=b["start"],
            char_end=end,
            article=b.get("article", ""),
            article_title=b.get("article_title", ""),
            clause_id=b.get("clause_id", ""),
            heading=b.get("heading", ""),
            defined_term=b.get("defined_term", ""),
            page_start=page_of(b["start"]),
            page_end=page_of(max(b["start"], end - 1)),
        )
        units.extend(_split_long(base, full, page_of, max_chars))
    return units


def _paragraph_spans(text: str) -> list[tuple[int, int]]:
    """Paragraph / sub-item blocks as (start, end) offsets inside text."""
    starts = [0]
    for match in re.finditer(r"\n", text):
        nxt = match.end()
        line_end = text.find("\n", nxt)
        line = text[nxt: line_end if line_end >= 0 else len(text)]
        prev_blank = match.start() > 0 and text[match.start() - 1] == "\n"
        if prev_blank or _SUBITEM_RE.match(line):
            starts.append(nxt)
    starts = sorted(set(starts))
    spans = [(s, starts[i + 1] if i + 1 < len(starts) else len(text)) for i, s in enumerate(starts)]
    return [(s, e) for s, e in spans if text[s:e].strip()]


def _sentence_spans(text: str, start: int, end: int, max_chars: int) -> list[tuple[int, int]]:
    spans, cursor = [], start
    while end - cursor > max_chars:
        window = text[cursor: cursor + max_chars]
        cut = max(window.rfind(". "), window.rfind("; "), window.rfind("\n"))
        if cut < max_chars // 3:
            cut = window.rfind(" ")
        if cut <= 0:
            cut = max_chars
        spans.append((cursor, cursor + cut + 1))
        cursor += cut + 1
    spans.append((cursor, end))
    return spans


def _split_long(unit: Unit, full: str, page_of, max_chars: int) -> list[Unit]:
    if len(unit.text) <= max_chars:
        return [unit]
    raw = full[unit.char_start: unit.char_end]
    blocks: list[tuple[int, int]] = []
    for s, e in _paragraph_spans(raw):
        if e - s > max_chars:
            blocks.extend(_sentence_spans(raw, s, e, max_chars))
        else:
            blocks.append((s, e))
    groups: list[tuple[int, int]] = []
    cur_s, cur_e = None, None
    for s, e in blocks:
        if cur_s is None:
            cur_s, cur_e = s, e
        elif e - cur_s <= max_chars:
            cur_e = e
        else:
            groups.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    if cur_s is not None:
        groups.append((cur_s, cur_e))
    out = []
    for n, (s, e) in enumerate(groups, start=1):
        abs_s, abs_e = unit.char_start + s, unit.char_start + e
        out.append(
            Unit(
                kind=unit.kind,
                text=full[abs_s:abs_e].strip(),
                char_start=abs_s,
                char_end=abs_e,
                article=unit.article,
                article_title=unit.article_title,
                clause_id=unit.clause_id,
                heading=unit.heading,
                defined_term=unit.defined_term,
                page_start=page_of(abs_s),
                page_end=page_of(max(abs_s, abs_e - 1)),
                part=n,
                parts=len(groups),
                breadcrumb=unit.breadcrumb,
            )
        )
    return out


# ------------------------------------------------------------------ markdown

_MD_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_MD_NUMBER = re.compile(r"^(\d{1,3}[A-Z]?(?:\.\d{1,3})*)\.?[ \t]+(.+)$")
_MD_ARTICLE = re.compile(r"(?i)\barticle\s+(\d{1,3}[A-Z]?)\b")
_FENCE = re.compile(r"^\s*(```|~~~)")


def is_markdown(pages: list[str]) -> bool:
    text = "\n".join(pages[:3])
    return len(re.findall(r"(?m)^#{1,6}[ \t]+\S", text)) >= 3


def parse_markdown(pages: list[str], max_chars: int = 2400) -> Outline:
    """Markdown structure from ATX headings.

    "## 14. Equality before the law" opens Article 14; "### 14.2 Equal protection"
    is clause 14.2 inside it; "#### 14.2.1 …" is clause 14.2.1. Unnumbered headings
    keep their title. Every unit carries the heading path as a breadcrumb.
    """
    page_starts, buf, offset = [], [], 0
    for text in pages:
        page_starts.append(offset)
        buf.append(text)
        offset += len(text) + 1
    full = "\n".join(buf)

    def page_of(pos: int) -> int:
        return max(1, bisect.bisect_right(page_starts, pos))

    headings: list[dict] = []
    pos, fenced = 0, False
    for line in full.split("\n"):
        if _FENCE.match(line):
            fenced = not fenced
        elif not fenced:
            m = _MD_HEADING.match(line)
            if m:
                headings.append({"level": len(m.group(1)), "title": m.group(2).strip(" *_"), "start": pos})
        pos += len(line) + 1

    units: list[Unit] = []
    if not headings:
        return Outline(units=_pack_text(full, 0, len(full), "text", set(), page_starts, page_of, max_chars))
    if headings[0]["start"] > 0 and full[: headings[0]["start"]].strip():
        units.extend(_pack_text(full, 0, headings[0]["start"], "preamble", set(), page_starts, page_of, max_chars))

    stack: list[dict] = []
    articles: dict[str, dict] = {}
    for idx, h in enumerate(headings):
        end = headings[idx + 1]["start"] if idx + 1 < len(headings) else len(full)
        while stack and stack[-1]["level"] >= h["level"]:
            stack.pop()
        num = _MD_NUMBER.match(h["title"])
        h["number"] = num.group(1) if num else ""
        h["name"] = num.group(2).strip() if num else h["title"]
        stack.append(h)

        article, article_title = "", ""
        for node in reversed(stack):
            if node["number"] and "." not in node["number"]:
                article, article_title = node["number"], node["name"]
                break
            named = _MD_ARTICLE.search(node["title"])
            if named and not node["number"]:
                article, article_title = named.group(1), node["title"]
                break
        clause = h["number"] if "." in h["number"] else ""
        breadcrumb = " › ".join(n["title"] for n in stack)

        if article:
            a = articles.setdefault(article, {"article": article, "title": article_title, "char_start": h["start"], "char_end": end})
            a["char_end"] = end

        text = full[h["start"]:end].strip()
        body = "\n".join(text.split("\n")[1:]).strip()
        if len(body) < 40:
            continue  # heading only; its title lives in the children's breadcrumb
        base = Unit(
            kind="clause" if (clause or (article and h["number"] == article)) else "text",
            text=text,
            char_start=h["start"],
            char_end=end,
            article=article,
            article_title=article_title,
            clause_id=clause,
            heading=h["name"],
            page_start=page_of(h["start"]),
            page_end=page_of(max(h["start"], end - 1)),
            breadcrumb=breadcrumb,
        )
        units.extend(_split_long(base, full, page_of, max_chars))

    outline_articles = [
        {**a, "page_start": page_of(a["char_start"]), "page_end": page_of(max(a["char_start"], a["char_end"] - 1))}
        for a in articles.values()
    ]
    return Outline(units=units, articles=outline_articles)


def _pack_text(full, start, end, kind, toc_pages, page_starts, page_of, max_chars) -> list[Unit]:
    """Paragraph-packed units for text without legal structure (letters, minutes)."""
    segment = full[start:end]
    out: list[Unit] = []
    spans = []
    for s, e in _paragraph_spans(segment):
        if page_of(start + s) in toc_pages:
            continue
        if e - s > max_chars:
            spans.extend(_sentence_spans(segment, s, e, max_chars))
        else:
            spans.append((s, e))
    cur: Optional[list[int]] = None
    groups = []
    for s, e in spans:
        if cur is None:
            cur = [s, e]
        elif e - cur[0] <= max_chars:
            cur[1] = e
        else:
            groups.append(tuple(cur))
            # Small overlap keeps a sentence that straddles the cut searchable.
            cur = [max(cur[1] - 200, cur[0]) if kind == "text" else s, e]
            cur[0] = s if e - cur[0] > max_chars else cur[0]
    if cur is not None:
        groups.append(tuple(cur))
    for n, (s, e) in enumerate(groups, start=1):
        text = segment[s:e].strip()
        if len(text) < 15:
            continue
        out.append(
            Unit(
                kind=kind,
                text=text,
                char_start=start + s,
                char_end=start + e,
                page_start=page_of(start + s),
                page_end=page_of(max(start + s, start + e - 1)),
                part=n,
                parts=len(groups),
            )
        )
    return out
