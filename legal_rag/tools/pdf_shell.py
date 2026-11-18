"""Shell commands that read PDFs under data/ril_data.

The query agent keeps its knowledge tools. Factual reading goes through
this shell command instead of search_knowledge. The process is the
current Python interpreter, and the text comes from pypdf in this
environment. Arguments are a list, not a shell string, so the same
command runs on Linux and Windows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Iterable, Optional

from agno.run import RunContext
from agno.tools.shell import ShellTools

from legal_rag.agents.messages import latest_user_text
from legal_rag.config import RIL_DATA_DIR, ROOT_DIR, SEARCH_MAX_RESULTS
from legal_rag.knowledge.catalog import identify_collection
from legal_rag.knowledge.paths import sites_relative_folder, strip_workspace_path

PDF_MODULE = "legal_rag.tools.pdf_shell"
CORPUS_LABEL = "data/ril_data"
MAX_SNIPPET = 480
MAX_READ_PAGES = 6
MAX_FILES = 2000
MAX_FILE_BYTES = 80 * 1024 * 1024
_STOP_WORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "what",
        "does",
        "per",
        "are",
        "was",
        "were",
        "into",
        "about",
    }
)


def _terms(query: str) -> list[str]:
    kept: list[str] = []
    for part in re.findall(r"[A-Za-z0-9][A-Za-z0-9./_-]*", query or ""):
        token = part.casefold()
        if token in _STOP_WORDS:
            continue
        if len(token) < 2 and not any(character.isdigit() for character in token):
            continue
        if token not in kept:
            kept.append(token)
    return kept


def resolve_under(root: Path, relative: str) -> Optional[Path]:
    """Resolve a relative path and refuse anything outside root.

    Accepts Windows separators. Rejects drive letters and absolute paths.
    """
    base = root.resolve()
    raw = (relative or "").strip().replace("\\", "/")
    if not raw or raw == ".":
        return base
    if raw.startswith("/") or re.match(r"^[A-Za-z]:", raw):
        return None
    candidate = (base / raw).resolve()
    try:
        candidate.relative_to(base)
    except ValueError:
        return None
    return candidate


def citation_for(path: Path, corpus_root: Path) -> dict[str, str]:
    """folder_path and filename in the same shape as knowledge citations."""
    relative = path.resolve().relative_to(corpus_root.resolve())
    filename = relative.name
    anchored = sites_relative_folder(Path(relative.as_posix()))
    if anchored is not None:
        folder = anchored
    else:
        parent = relative.parent.as_posix()
        folder = "" if parent == "." else parent
    folder = strip_workspace_path(folder).strip("/")
    hit = identify_collection(folder, filename)
    return {
        "filename": filename,
        "folder_path": folder,
        "doc_family": hit.code if hit is not None else "",
    }


def _page_matches(body: str, terms: list[str]) -> bool:
    """True when this page holds the query, not merely the file name."""
    present = [term for term in terms if term in body]
    if not present:
        return False
    if len(terms) <= 3:
        return len(present) == len(terms)
    anchors = [term for term in terms if any(character.isdigit() for character in term)]
    if not anchors:
        anchors = [max(terms, key=len)]
    return any(term in present for term in anchors) and len(present) >= 2


def _excerpt(text: str, terms: Iterable[str]) -> str:
    """The passage around the match. The rest of the page is discarded."""
    compact = text.replace("\r", "")
    lowered = compact.casefold()
    indexes = [lowered.find(term) for term in terms]
    indexes = [index for index in indexes if index >= 0]
    if not indexes:
        return ""
    index = min(indexes)
    line_start = compact.rfind("\n", 0, index)
    start = line_start + 1 if line_start >= 0 and index - line_start <= 180 else max(0, index - 80)
    paragraph_end = compact.find("\n\n", index)
    end = paragraph_end if paragraph_end != -1 and paragraph_end <= index + MAX_SNIPPET else index + MAX_SNIPPET
    end = min(len(compact), end)
    excerpt = re.sub(r"[ \t]+", " ", compact[start:end]).strip()
    return excerpt[:MAX_SNIPPET]


def _console(message: str) -> None:
    """Timing line for the process console. stdout stays JSON for the CLI."""
    print(message, file=sys.stderr)


def _page_text(page) -> str:
    """Extract upright text. Rotated text is read only when that is empty."""
    try:
        text = page.extract_text(orientations=(0,)) or ""
    except Exception:
        return ""
    if text.strip():
        return text
    try:
        rotated = page.extract_text(orientations=(90, 270)) or ""
    except Exception:
        return text
    return rotated or text


_CACHE_LOCK = threading.Lock()
_PAGE_MEMORY: dict[str, dict] = {}


def _cache_file(corpus: Path, pdf: Path) -> Path:
    digest = hashlib.sha1(str(pdf.resolve()).encode()).hexdigest()
    return corpus / ".pdf_text_cache" / digest[:2] / f"{digest}.json"


def _file_stamp(pdf: Path) -> tuple[int, int]:
    stat = pdf.stat()
    return stat.st_size, stat.st_mtime_ns


def _remember(pdf: Path, pages: list[str], complete: bool) -> None:
    size, mtime_ns = _file_stamp(pdf)
    _PAGE_MEMORY[str(pdf.resolve())] = {
        "size": size,
        "mtime_ns": mtime_ns,
        "complete": complete,
        "pages": pages,
    }


def _cached_pages(corpus: Path, pdf: Path) -> tuple[list[str], bool]:
    """Pages already extracted for this file. complete means every page is stored."""
    size, mtime_ns = _file_stamp(pdf)
    key = str(pdf.resolve())
    with _CACHE_LOCK:
        record = _PAGE_MEMORY.get(key)
        if record and record.get("size") == size and record.get("mtime_ns") == mtime_ns:
            return list(record.get("pages") or []), bool(record.get("complete"))
    path = _cache_file(corpus, pdf)
    if not path.is_file():
        return [], False
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return [], False
    if record.get("size") != size or record.get("mtime_ns") != mtime_ns:
        return [], False
    pages = [text if isinstance(text, str) else "" for text in record.get("pages") or []]
    complete = bool(record.get("complete"))
    with _CACHE_LOCK:
        _PAGE_MEMORY[key] = {
            "size": size,
            "mtime_ns": mtime_ns,
            "complete": complete,
            "pages": list(pages),
        }
    return pages, complete


def _store_pages(corpus: Path, pdf: Path, pages: list[str], complete: bool) -> None:
    _remember(pdf, pages, complete)
    path = _cache_file(corpus, pdf)
    payload = {
        "size": _file_stamp(pdf)[0],
        "mtime_ns": _file_stamp(pdf)[1],
        "complete": complete,
        "pages": pages,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)
    except OSError:
        return


def _open_reader(path: Path):
    if path.stat().st_size > MAX_FILE_BYTES:
        raise OSError("file is larger than the read limit")
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    if getattr(reader, "is_encrypted", False):
        raise OSError("PDF is encrypted")
    return reader


def _excerpt_if_match(text: str, terms: list[str]) -> str:
    if not text or not _page_matches(text.casefold(), terms):
        return ""
    return _excerpt(text, terms)


def _first_passage(corpus: Path, pdf: Path, terms: list[str]) -> tuple[Optional[tuple[int, str]], bool]:
    """Return the first matching passage and whether the PDF had to be opened.

    Pages already extracted are reused. Later pages stay unread.
    """
    pages, complete = _cached_pages(corpus, pdf)
    for index, text in enumerate(pages, start=1):
        excerpt = _excerpt_if_match(text, terms)
        if excerpt:
            return (index, excerpt), False
    if complete:
        return None, False
    reader = _open_reader(pdf)
    total = len(reader.pages)
    for index in range(len(pages) + 1, total + 1):
        text = _page_text(reader.pages[index - 1])
        pages.append(text)
        excerpt = _excerpt_if_match(text, terms)
        if excerpt:
            _store_pages(corpus, pdf, pages, complete=index == total)
            return (index, excerpt), True
    _store_pages(corpus, pdf, pages, complete=True)
    return None, True


def _text_on_page(corpus: Path, pdf: Path, number: int) -> tuple[str, bool]:
    """One stored page. Opens the PDF only when that page is not stored yet."""
    pages, complete = _cached_pages(corpus, pdf)
    if 1 <= number <= len(pages):
        return pages[number - 1], False
    if complete or number < 1:
        return "", False
    reader = _open_reader(pdf)
    total = len(reader.pages)
    while len(pages) < number and len(pages) < total:
        pages.append(_page_text(reader.pages[len(pages)]))
    _store_pages(corpus, pdf, pages, complete=len(pages) >= total)
    if number <= len(pages):
        return pages[number - 1], True
    return "", True


def _iter_pdfs(base: Path, filename: str) -> list[Path]:
    needle = (filename or "").casefold()
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
        dirnames[:] = [
            name for name in dirnames if name not in {".git", "__pycache__", ".pdf_text_cache"}
        ]
        for name in filenames:
            if not name.lower().endswith(".pdf"):
                continue
            if needle and needle not in name.casefold():
                continue
            found.append(Path(dirpath) / name)
            if len(found) >= MAX_FILES:
                return sorted(found)
    return sorted(found)


def _rank_pdfs(files: list[Path], terms: list[str]) -> list[Path]:
    """Files whose names contain the query are opened before the rest."""

    def rank(path: Path) -> tuple[int, str]:
        name = path.name.casefold()
        overlap = sum(1 for term in terms if len(term) >= 3 and term in name)
        return (-overlap, name)

    return sorted(files, key=rank)


def _parse_pages(spec: str, page_count: int) -> list[int]:
    """1-based page numbers. Empty means the first pages."""
    raw = (spec or "").strip()
    if not raw:
        last = min(page_count, 3)
        return list(range(1, last + 1))
    numbers: list[int] = []
    for piece in raw.split(","):
        part = piece.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                start, end = end, start
            numbers.extend(range(start, end + 1))
        else:
            numbers.append(int(part))
    chosen = [number for number in numbers if 1 <= number <= page_count]
    return chosen[:MAX_READ_PAGES]


def search_corpus(
    query: str,
    *,
    root: Path,
    folder: str = "",
    filename: str = "",
    max_hits: int = SEARCH_MAX_RESULTS,
) -> dict:
    corpus = root.resolve()
    terms = _terms(query)
    payload: dict = {
        "root": CORPUS_LABEL,
        "query": query,
        "folder": (folder or "").replace("\\", "/"),
        "filename": filename,
        "hits": [],
        "truncated": False,
        "files_scanned": 0,
        "errors": [],
    }
    if not terms:
        payload["note"] = "Provide a query. Use read_context to open one file."
        return payload
    if not corpus.is_dir():
        payload["note"] = f"No PDF folder at {CORPUS_LABEL}."
        return payload
    base = resolve_under(corpus, folder)
    if base is None:
        payload["note"] = "folder must stay inside data/ril_data."
        return payload
    if not base.exists():
        payload["note"] = "That folder is not under data/ril_data."
        return payload

    limit = max(1, min(int(max_hits), 30))
    files = _rank_pdfs(_iter_pdfs(base, filename), terms)
    opened = 0
    started = time.perf_counter()
    for path in files:
        if len(payload["hits"]) >= limit:
            payload["truncated"] = True
            break
        payload["files_scanned"] += 1
        try:
            found, did_open = _first_passage(corpus, path, terms)
        except ImportError:
            payload["note"] = "pypdf is not installed in this Python environment."
            return payload
        except Exception as exc:
            if len(payload["errors"]) < 5:
                payload["errors"].append(f"{path.name}: {exc}")
            continue
        if did_open:
            opened += 1
        if found is None:
            continue
        page, excerpt = found
        payload["hits"].append({**citation_for(path, corpus), "page": page, "snippet": excerpt})
    if not payload["hits"] and not payload["errors"]:
        payload["note"] = "No matching PDF text in data/ril_data."
    _console(
        "PDF search: "
        f"{len(payload['hits'])} hit(s), {payload['files_scanned']} file(s), "
        f"{opened} opened, {payload['files_scanned'] - opened} from stored text, "
        f"{time.perf_counter() - started:.1f}s"
    )
    return payload


def read_corpus(path: str, *, root: Path, pages: str = "", query: str = "") -> dict:
    """Return the matching passage of one PDF and stop.

    Pages before the match are checked only to find it. Later pages are
    not extracted, and the rest of the matching page is not returned.
    """
    corpus = root.resolve()
    target = resolve_under(corpus, path)
    payload: dict = {"root": CORPUS_LABEL, "path": (path or "").replace("\\", "/"), "pages": []}
    terms = _terms(query)
    if target is None or not target.is_file() or target.suffix.lower() != ".pdf":
        payload["note"] = "path must be a PDF inside data/ril_data."
        return payload
    if not terms:
        payload["note"] = "Pass query. Only the matching passage is read."
        return payload
    started = time.perf_counter()
    located = citation_for(target, corpus)
    payload.update(located)
    try:
        if (pages or "").strip():
            selected = _parse_pages(pages, 100000)
            for number in selected:
                text, _opened = _text_on_page(corpus, target, number)
                excerpt = _excerpt_if_match(text, terms)
                if not excerpt:
                    continue
                payload["pages"].append({"page": number, "text": excerpt})
                payload["note"] = "Matching passage only. The rest of the file was not returned."
                _console(f"PDF read: {target.name} page {number} in {time.perf_counter() - started:.1f}s")
                return payload
        else:
            found, _opened = _first_passage(corpus, target, terms)
            if found is not None:
                number, excerpt = found
                payload["pages"].append({"page": number, "text": excerpt})
                payload["note"] = "Matching passage only. The rest of the file was not returned."
                _console(f"PDF read: {target.name} page {number} in {time.perf_counter() - started:.1f}s")
                return payload
    except Exception as exc:
        payload["note"] = f"{target.name}: {exc}"
        return payload
    payload["note"] = "No matching passage in that PDF."
    _console(f"PDF read: {target.name} no match in {time.perf_counter() - started:.1f}s")
    return payload


def _dump(payload: dict) -> str:
    return strip_workspace_path(json.dumps(payload, ensure_ascii=False))


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Read PDFs under data/ril_data.")
    parser.add_argument("--root", default=str(RIL_DATA_DIR))
    subparsers = parser.add_subparsers(dest="command", required=True)

    search = subparsers.add_parser("search")
    search.add_argument("--query", required=True)
    search.add_argument("--folder", default="")
    search.add_argument("--filename", default="")
    search.add_argument("--max-hits", type=int, default=SEARCH_MAX_RESULTS)

    read = subparsers.add_parser("read")
    read.add_argument("--path", required=True)
    read.add_argument("--pages", default="")
    read.add_argument("--query", default="")

    args = parser.parse_args(argv)
    root = Path(args.root)
    if args.command == "search":
        print(
            _dump(
                search_corpus(
                    args.query,
                    root=root,
                    folder=args.folder,
                    filename=args.filename,
                    max_hits=args.max_hits,
                )
            )
        )
        return 0
    print(_dump(read_corpus(args.path, root=root, pages=args.pages, query=args.query)))
    return 0


class PdfCorpusShell(ShellTools):
    """Shell tool that only reads PDFs under data/ril_data.

    run_shell_command stays on the class and is not offered to the model.
    search_context and read_context read PDFs in this process and keep the
    extracted text, so a later search or read of the same page does not
    open the file again. The same module is the shell command on Linux
    and Windows: python -m legal_rag.tools.pdf_shell.
    """

    def __init__(self, corpus_dir: Optional[Path] = None, timeout: int = 180, **kwargs):
        self.corpus_dir = Path(corpus_dir) if corpus_dir is not None else RIL_DATA_DIR
        super().__init__(
            base_dir=ROOT_DIR,
            enable_run_shell_command=False,
            timeout=timeout,
            **kwargs,
        )
        self.register(self.search_context)
        self.register(self.read_context)

    def _command(self, *parts: str) -> list[str]:
        return [sys.executable, "-m", PDF_MODULE, "--root", str(self.corpus_dir), *parts]

    def _run(self, args: list[str]) -> str:
        if args[:3] != [sys.executable, "-m", PDF_MODULE]:
            return "Error: only the data/ril_data PDF shell command is allowed."
        env = os.environ.copy()
        root = str(ROOT_DIR)
        previous = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = root if not previous else root + os.pathsep + previous
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        try:
            result = subprocess.run(
                args,
                capture_output=True,
                cwd=str(ROOT_DIR),
                env=env,
                timeout=self.timeout or 180,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
        except subprocess.TimeoutExpired:
            return "Error: PDF shell command timed out."
        except OSError as exc:
            return f"Error: {exc}"
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "PDF shell command failed").strip()
            return f"Error: {detail}"
        return (result.stdout or "").strip()

    def search_context(
        self,
        run_context: RunContext,
        query: str = "",
        folder: str = "",
        filename: str = "",
    ) -> str:
        """Context search of PDF text in data/ril_data.

        Returns only the matching passage of a file, then stops. The
        rest of that file is not returned. Pass folder and filename
        when they are known so the search does not open every PDF.
        Extracted pages are kept for the next search and for read_context.

        Args:
            query: Words, clause, heading, or party from the current
                user message. Empty uses that user message.
            folder: Folder under data/ril_data, with forward slashes.
                Examples: PSC, JOA/New JOA, sites/Regulatory/KGD6.
                Empty searches the whole ril_data tree and is slow.
            filename: Case-insensitive part of the PDF file name.
                Empty searches every PDF in folder.

        Returns:
            JSON hits with filename, folder_path, page, and a short
            snippet. Cite folder_path and filename with the existing
            link rules.
        """
        question = (query or "").strip() or latest_user_text(run_context)
        try:
            return _dump(
                search_corpus(
                    question,
                    root=self.corpus_dir,
                    folder=folder,
                    filename=filename,
                )
            )
        except Exception as exc:
            return f"Error: {exc}"

    def read_context(
        self,
        run_context: RunContext,
        path: str = "",
        query: str = "",
        pages: str = "",
    ) -> str:
        """Read the PDF context: the matching passage of one file under data/ril_data.

        Stops at the first matching passage. Does not return the rest
        of the file. Pass pages as the page number from the search hit.
        That page is reused and the PDF is not opened again.

        Args:
            path: PDF path relative to data/ril_data, forward slashes.
                Example: JOA/New JOA/file.pdf
            query: Text to locate. Empty uses the current user message.
            pages: 1-based page from the search hit, such as "4" or "4-6".
                Empty continues from stored text until the first match.

        Returns:
            JSON with filename, folder_path, and the matching passage.
        """
        question = (query or "").strip() or latest_user_text(run_context)
        if not (path or "").strip():
            return self.search_context(run_context, query=question)
        try:
            return _dump(
                read_corpus(path, root=self.corpus_dir, pages=pages, query=question)
            )
        except Exception as exc:
            return f"Error: {exc}"


if __name__ == "__main__":
    raise SystemExit(main())
