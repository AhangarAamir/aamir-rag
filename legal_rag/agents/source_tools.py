"""Live reads of the original files under the corpus folder.

The index answers most questions. These tools go back to the source file at
question time, for:
  - exact wording checks (the index stores cleaned text; this is the file as is)
  - tables and schedules, read with `pdftotext -layout` so columns stay aligned
  - Markdown sections by heading number or title
  - files added to the corpus but not indexed yet

PDFs are read with poppler's `pdftotext` / `pdfinfo` from this environment, run
as fixed argument lists (no shell, no user-built commands) with a timeout, and
only on files inside CORPUS_DIR. When poppler is missing, pypdf is used in-process.
Results from indexed files list the [C…] tags of the indexed passages on the same
pages, so the answer can still cite verifiable passages.
"""

from __future__ import annotations

import difflib
import re
import shutil
import subprocess
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

from agno.tools import Toolkit

from legal_rag.config import CORPUS_DIR
from legal_rag.index import store
from legal_rag.index.search import resolve_documents
from legal_rag.ingest.extract import SUPPORTED_SUFFIXES, TEXT_SUFFIXES, _text_file

MAX_CHARS = 12_000
MAX_PAGES = 6
MAX_GREP_FILES = 400
_CACHE: "OrderedDict[tuple, list[str]]" = OrderedDict()
_CACHE_LOCK = threading.Lock()


# ------------------------------------------------------------------ files

def _corpus() -> Path:
    return CORPUS_DIR.resolve()


def _inside(path: Path) -> bool:
    root = _corpus()
    try:
        path.resolve().relative_to(root)
        return True
    except ValueError:
        return False


def _all_files(folder: str = "") -> list[Path]:
    base = (_corpus() / folder.strip("/").replace("\\", "/")).resolve() if folder else _corpus()
    if not _inside(base) or not base.exists():
        return []
    files = [p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
             and not any(part.startswith(".") for part in p.relative_to(_corpus()).parts)]
    return sorted(files)[:MAX_GREP_FILES]


def find_files(document: str, folder: str = "") -> tuple[list[Path], str]:
    """Indexed documents by name/alias first, then file-name / path match on disk."""
    document = (document or "").strip()
    if not document:
        return _all_files(folder), ""
    scope = resolve_documents(document)
    if scope.documents:
        paths = [(_corpus() / d["rel_path"]) for d in scope.documents]
        return [p for p in paths if p.is_file() and _inside(p)], ""
    needle = re.sub(r"[^a-z0-9]+", "", document.lower())
    hits = [p for p in _all_files(folder)
            if needle and needle in re.sub(r"[^a-z0-9]+", "", p.relative_to(_corpus()).as_posix().lower())]
    if hits:
        return hits, ""
    return [], f'No file in the corpus matches "{document}".'


def rel(path: Path) -> str:
    return path.resolve().relative_to(_corpus()).as_posix()


# ------------------------------------------------------------------ text

def _run(args: list[str], timeout: int = 60) -> Optional[str]:
    exe = shutil.which(args[0])
    if not exe:
        return None
    try:
        result = subprocess.run([exe, *args[1:]], capture_output=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.decode("utf-8", errors="replace")


def pdf_page_count(path: Path) -> int:
    info = _run(["pdfinfo", str(path)], timeout=20)
    if info:
        m = re.search(r"(?m)^Pages:\s+(\d+)", info)
        if m:
            return int(m.group(1))
    from pypdf import PdfReader

    return len(PdfReader(str(path)).pages)


def pdf_pages(path: Path, first: int, last: int, layout: bool = False) -> list[str]:
    """Pages first..last (1-based) as the PDF has them. `layout` keeps table columns."""
    args = ["pdftotext", "-enc", "UTF-8", "-f", str(first), "-l", str(last)]
    if layout:
        args.append("-layout")
    out = _run([*args, str(path), "-"])
    if out is not None:
        pages = out.split("\f")
        if pages and not pages[-1].strip():
            pages.pop()
        return pages
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return [
        (reader.pages[i].extract_text(extraction_mode="layout" if layout else "plain") or "")
        for i in range(first - 1, min(last, len(reader.pages)))
    ]


def file_pages(path: Path) -> list[str]:
    """Whole file as pages, cached by size + mtime. Text files use the ingest's pseudo-pages."""
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    with _CACHE_LOCK:
        if key in _CACHE:
            _CACHE.move_to_end(key)
            return _CACHE[key]
    if path.suffix.lower() in TEXT_SUFFIXES:
        pages = _text_file(path)
    else:
        pages = pdf_pages(path, 1, pdf_page_count(path))
    with _CACHE_LOCK:
        _CACHE[key] = pages
        while len(_CACHE) > 64:
            _CACHE.popitem(last=False)
    return pages


def overlapping_tags(path: Path, first: int, last: int) -> str:
    """Tags of indexed passages on these pages, so live text stays citable."""
    doc = store.find_document_by_path(rel(path))
    if not doc:
        return "This file is not indexed yet: cite it by file name and page in prose (run ingest to make it citable)."
    rows = store.query(
        "SELECT id, clause_id, heading, article, page_start, page_end FROM chunks"
        " WHERE doc_id=? AND page_end>=? AND page_start<=? AND kind!='toc' ORDER BY ord LIMIT 40",
        (doc["id"], first, last),
    )
    if not rows:
        return ""
    tags = []
    for r in rows:
        label = r["clause_id"] or (f"Article {r['article']}" if r["article"] else (r["heading"] or "text"))
        tags.append(f"[C{r['id']}] {label}")
    return "Cite with the indexed passages on these pages: " + "; ".join(tags)


# ------------------------------------------------------------------ markdown sections

_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")


def markdown_sections(text: str) -> list[dict[str, Any]]:
    lines = text.split("\n")
    heads = []
    fenced = False
    for i, line in enumerate(lines):
        if re.match(r"^\s*(```|~~~)", line):
            fenced = not fenced
            continue
        m = None if fenced else _HEADING.match(line)
        if m:
            heads.append({"line": i, "level": len(m.group(1)), "title": m.group(2).strip(" *_")})
    for idx, h in enumerate(heads):
        end = len(lines)
        for nxt in heads[idx + 1:]:
            if nxt["level"] <= h["level"]:
                end = nxt["line"]
                break
        h["end"] = end
        h["text"] = "\n".join(lines[h["line"]:end]).strip()
    return heads


def pick_section(sections: list[dict[str, Any]], heading: str) -> tuple[Optional[dict[str, Any]], list[str]]:
    want = heading.strip().lower()
    want = re.sub(r"^(article|section|clause)\s+", "", want)
    number = re.match(r"^(\d{1,3}[a-z]?(?:\.\d{1,3})*)\b", want)
    if number:
        for s in sections:
            m = re.match(r"^(\d{1,3}[A-Za-z]?(?:\.\d{1,3})*)\.?\s", s["title"] + " ")
            if m and m.group(1).lower() == number.group(1):
                return s, []
    for s in sections:
        if want and want in s["title"].lower():
            return s, []
    titles = [s["title"] for s in sections]
    close = difflib.get_close_matches(heading, titles, n=6, cutoff=0.5)
    return None, close


# ------------------------------------------------------------------ toolkit

class SourceFileTools(Toolkit):
    def __init__(self, **kwargs: Any):
        super().__init__(
            name="source_files",
            tools=[self.read_source_pages, self.read_markdown_section, self.grep_source],
            **kwargs,
        )

    def read_source_pages(self, document: str, start_page: int, end_page: int = 0, keep_layout: bool = False) -> str:
        """Read pages straight from the original file (PDF via pdftotext, or Markdown/text), not from the index.
        Use to check exact wording, to read tables/schedules (keep_layout=True keeps columns aligned), or to read a file that is not indexed yet.

        Args:
            document: Document name ("KG-D6 PSC") or part of its file name / path.
            start_page: First page, 1-based.
            end_page: Last page (at most 6 pages per call). 0 reads one page.
            keep_layout: True to preserve column layout (tables, fiscal schedules).
        """
        files, note = find_files(document)
        if not files:
            return note or f'No file found for "{document}".'
        first = max(1, int(start_page or 1))
        last = max(first, int(end_page or first))
        last = min(last, first + MAX_PAGES - 1)

        def count(p: Path) -> int:
            return pdf_page_count(p) if p.suffix.lower() == ".pdf" else len(file_pages(p))

        # Several files can match one name (a contract and its amendments): use one that has the page.
        try:
            sized = [(p, count(p)) for p in files[:12]]
        except Exception as exc:
            return f"Could not read {rel(files[0])}: {exc}"
        candidates = [(p, n) for p, n in sized if n >= first]
        if not candidates:
            sizes = ", ".join(f"{rel(p)} ({n} pages)" for p, n in sized)
            return f"Page {first} is out of range for every matching file: {sizes}."
        path, total = max(candidates, key=lambda x: x[1]) if len(candidates) > 1 else candidates[0]
        files = [path] + [p for p, _ in sized if p != path]
        last = min(last, total)
        try:
            if path.suffix.lower() == ".pdf":
                pages = pdf_pages(path, first, last, layout=keep_layout)
            else:
                pages = file_pages(path)[first - 1:last]
        except Exception as exc:
            return f"Could not read {rel(path)}: {exc}"
        out = [f"File: {rel(path)} (pages {first}–{first + len(pages) - 1} of {total})"]
        if len(files) > 1:
            out.append("Other matching files: " + ", ".join(rel(p) for p in files[1:6]))
        used = 0
        for n, text in enumerate(pages, start=first):
            text = text.rstrip()
            if used + len(text) > MAX_CHARS:
                out.append(f"[stopped at {MAX_CHARS} characters — read from page {n} to continue]")
                break
            out.append(f"--- page {n} ---\n{text if text.strip() else '[no text layer on this page: scanned image, needs OCR]'}")
            used += len(text)
        tags = overlapping_tags(path, first, first + len(pages) - 1)
        if tags:
            out.append(tags)
        return "\n".join(out)

    def read_markdown_section(self, document: str, heading: str) -> str:
        """Read one section of a Markdown/text file straight from disk, by heading number or title, including its sub-sections.
        Example: heading="14.2" or "Article 21" or "Equality before the law".

        Args:
            document: Document name or part of its file name.
            heading: Section number or (part of) the heading title.
        """
        files, note = find_files(document)
        files = [p for p in files if p.suffix.lower() in TEXT_SUFFIXES]
        if not files:
            return note or f'No Markdown/text file found for "{document}". For PDFs use get_clause or read_source_pages.'
        # Try every matching file until one has the heading.
        path, sections, section, close = files[0], [], None, []
        for candidate in files:
            found = markdown_sections("\n".join(_text_file(candidate)))
            hit, near = pick_section(found, heading) if found else (None, [])
            if hit is not None:
                path, sections, section = candidate, found, hit
                break
            if candidate == files[0]:
                sections, close = found, near
        if not sections:
            return f"{rel(path)} has no Markdown headings. Use read_source_pages."
        if section is None:
            outline = ", ".join(s["title"] for s in sections if s["level"] <= 2)[:1500]
            hint = f" Closest headings: {'; '.join(close)}." if close else ""
            return f'No heading "{heading}" in {rel(path)}.{hint} Top-level headings: {outline}'
        body = section["text"]
        if len(body) > MAX_CHARS:
            subs = [s["title"] for s in sections if section["line"] < s["line"] < section["end"]]
            body = body[:MAX_CHARS] + f"\n[truncated — read a sub-section instead: {'; '.join(subs[:20])}]"
        doc = store.find_document_by_path(rel(path))
        tags = ""
        if doc:
            number = re.match(r"^(\d{1,3}[A-Za-z]?(?:\.\d{1,3})*)\.?\s", section["title"] + " ")
            rows = []
            if number:
                n = number.group(1).upper() if re.fullmatch(r"\d+[a-z]", number.group(1)) else number.group(1)
                rows = store.query(
                    "SELECT id, clause_id, article FROM chunks WHERE doc_id=? AND (clause_id=? OR clause_id LIKE ?"
                    " OR (article=? AND clause_id='')) ORDER BY ord LIMIT 30",
                    (doc["id"], n, n + ".%", n),
                )
            if rows:
                tags = "Cite with: " + "; ".join(f"[C{r['id']}] {r['clause_id'] or 'Article ' + r['article']}" for r in rows)
        return "\n".join(x for x in (f"File: {rel(path)} · section “{section['title']}”", body, tags) if x)

    def grep_source(self, pattern: str, document: str = "", folder: str = "", max_hits: int = 10) -> str:
        """Search the raw text of the original files (like grep), page by page, with surrounding lines.
        Finds exact phrases, numbers and names, including in files not indexed yet. Several words must all appear within a few lines.

        Args:
            pattern: Exact phrase or a few words, e.g. "deed of adherence" or "12.5 per cent".
            document: Optional document name / file name to search in.
            folder: Optional folder under the corpus, e.g. "PSC/Domestic".
            max_hits: Maximum matches to return (1-25).
        """
        pattern = (pattern or "").strip().strip('"“”')
        if len(pattern) < 2:
            return "Give a phrase or words to look for."
        limit = max(1, min(int(max_hits or 10), 25))
        phrase = re.sub(r"\s+", " ", pattern.lower())
        words = [w for w in re.findall(r"[a-z0-9%.]+", phrase) if len(w) > 1] or [phrase]

        def match(text: str) -> Optional[tuple[int, str]]:
            lines = text.split("\n")
            lowered = [re.sub(r"\s+", " ", ln.lower()) for ln in lines]
            for i in range(len(lines)):
                window = " ".join(lowered[i:i + 3])
                if phrase in window or (len(words) > 1 and all(w in window for w in words)):
                    if phrase not in lowered[i] and not any(w in lowered[i] for w in words):
                        continue
                    return i + 1, "\n".join(lines[max(0, i - 2): i + 4]).strip()
            return None

        # Scope: a named document, a folder, or everything.
        doc_ids: list[int] = []
        disk_files: list[Path] = []
        if document.strip():
            scope = resolve_documents(document)
            doc_ids = scope.doc_ids
            if not doc_ids:
                disk_files, note = find_files(document, folder)
                if not disk_files:
                    return note or f'No file matches "{document}".'
        folder_prefix = folder.strip("/").replace("\\", "/")

        # 1) Indexed files: page text is already stored, so this covers the whole corpus quickly.
        where, params = ["1=1"], []
        if doc_ids:
            where.append(f"d.id IN ({','.join('?' * len(doc_ids))})")
            params += doc_ids
        if folder_prefix:
            where.append("d.rel_path LIKE ?")
            params.append(folder_prefix + "/%")
        for w in words[:6]:
            where.append("p.text LIKE ?")
            params.append(f"%{w}%")
        rows = store.query(
            f"SELECT d.rel_path, p.page_no, p.text FROM pages p JOIN documents d ON d.id = p.doc_id"
            f" WHERE {' AND '.join(where)} ORDER BY d.rel_path, p.page_no LIMIT 5000",
            params,
        )
        hits: list[str] = []
        files_hit: set[str] = set()
        for row in rows:
            found = match(row["text"])
            if found:
                files_hit.add(row["rel_path"])
                hits.append(f"{row['rel_path']} · p. {row['page_no']}, line {found[0]}:\n{found[1]}")
                if len(hits) >= limit:
                    break
        indexed_docs = len(doc_ids) if doc_ids else int(store.query(
            "SELECT COUNT(*) n FROM documents d WHERE " + ("d.rel_path LIKE ?" if folder_prefix else "1=1"),
            [folder_prefix + "/%"] if folder_prefix else [],
        )[0]["n"])

        # 2) Files on disk that are not indexed yet: extract them (capped, slower).
        unindexed_read = 0
        if len(hits) < limit and not doc_ids:
            candidates = disk_files or _all_files(folder_prefix)
            known = {r["rel_path"] for r in store.query("SELECT rel_path FROM documents")}
            for path in [p for p in candidates if rel(p) not in known][:50]:
                try:
                    pages = file_pages(path)
                except Exception:
                    continue
                unindexed_read += 1
                for page_no, text in enumerate(pages, start=1):
                    found = match(text)
                    if found:
                        hits.append(f"{rel(path)} · p. {page_no}, line {found[0]} (not indexed yet):\n{found[1]}")
                        break
                if len(hits) >= limit:
                    break

        searched = f"{indexed_docs} indexed file(s)" + (f" + {unindexed_read} not-yet-indexed file(s)" if unindexed_read else "")
        if not hits:
            return f'"{pattern}" not found in {searched}.'
        stopped = f" (stopped at the {limit}-match limit; narrow with document= or folder=)" if len(hits) >= limit else ""
        return f'{len(hits)} match(es) for "{pattern}" across {searched}{stopped}:\n\n' + "\n\n".join(hits) + (
            "\n\nTo cite, open the passage with get_clause / read_source_pages to get its [C…] tag."
        )
