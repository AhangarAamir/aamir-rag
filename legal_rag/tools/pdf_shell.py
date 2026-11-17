"""Shell commands that read PDFs under data/ril_data.

The query agent keeps its knowledge tools. Factual reading goes through
this shell command instead of search_knowledge. The process is the
current Python interpreter, and the text comes from pypdf in this
environment. Arguments are a list, not a shell string, so the same
command runs on Linux and Windows.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional

from agno.tools.shell import ShellTools

from legal_rag.config import RIL_DATA_DIR, ROOT_DIR, SEARCH_MAX_RESULTS
from legal_rag.knowledge.catalog import identify_collection
from legal_rag.knowledge.chunking import sites_relative_folder, strip_workspace_path

PDF_MODULE = "legal_rag.tools.pdf_shell"
CORPUS_LABEL = "data/ril_data"
MAX_SNIPPET = 700
MAX_READ_CHARS = 4000
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


def _snippet(text: str, terms: Iterable[str]) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    lowered = compact.casefold()
    index = -1
    for term in terms:
        index = lowered.find(term)
        if index >= 0:
            break
    if index < 0:
        index = 0
    start = max(0, index - 80)
    end = min(len(compact), start + MAX_SNIPPET)
    return compact[start:end]


def _page_text(page) -> str:
    try:
        return page.extract_text() or ""
    except Exception:
        return ""


def _iter_pdfs(base: Path, filename: str) -> list[Path]:
    needle = (filename or "").casefold()
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
        dirnames[:] = [name for name in dirnames if name not in {".git", "__pycache__"}]
        for name in filenames:
            if not name.lower().endswith(".pdf"):
                continue
            if needle and needle not in name.casefold():
                continue
            found.append(Path(dirpath) / name)
            if len(found) >= MAX_FILES:
                return sorted(found)
    return sorted(found)


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
        payload["note"] = "Provide a query. Use read_pdf to open one file."
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

    try:
        from pypdf import PdfReader
    except ImportError:
        payload["note"] = "pypdf is not installed in this Python environment."
        return payload

    limit = max(1, min(int(max_hits), 30))
    files = _iter_pdfs(base, filename)
    for path in files:
        if len(payload["hits"]) >= limit:
            payload["truncated"] = True
            break
        payload["files_scanned"] += 1
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                raise OSError("file is larger than the read limit")
            reader = PdfReader(str(path))
            if getattr(reader, "is_encrypted", False):
                raise OSError("PDF is encrypted")
        except Exception as exc:
            if len(payload["errors"]) < 5:
                payload["errors"].append(f"{path.name}: {exc}")
            continue
        located = citation_for(path, corpus)
        location = f"{located['folder_path']}\n{located['filename']}".casefold()
        for index, page in enumerate(reader.pages, start=1):
            text = _page_text(page)
            body = text.casefold()
            in_body = all(term in body for term in terms)
            in_name = index == 1 and all(term in f"{location}\n{body}" for term in terms)
            if not in_body and not in_name:
                continue
            payload["hits"].append(
                {
                    **located,
                    "page": index,
                    "snippet": _snippet(text or located["filename"], terms),
                }
            )
            if len(payload["hits"]) >= limit:
                payload["truncated"] = True
                break
    if not payload["hits"] and not payload["errors"]:
        payload["note"] = "No matching PDF text in data/ril_data."
    return payload


def read_corpus(path: str, *, root: Path, pages: str = "") -> dict:
    corpus = root.resolve()
    target = resolve_under(corpus, path)
    payload: dict = {"root": CORPUS_LABEL, "path": (path or "").replace("\\", "/"), "pages": []}
    if target is None or not target.is_file() or target.suffix.lower() != ".pdf":
        payload["note"] = "path must be a PDF inside data/ril_data."
        return payload
    try:
        from pypdf import PdfReader
    except ImportError:
        payload["note"] = "pypdf is not installed in this Python environment."
        return payload
    try:
        reader = PdfReader(str(target))
        if getattr(reader, "is_encrypted", False):
            raise OSError("PDF is encrypted")
        selected = _parse_pages(pages, len(reader.pages))
    except Exception as exc:
        payload["note"] = f"{target.name}: {exc}"
        return payload
    located = citation_for(target, corpus)
    payload.update(located)
    truncated = False
    for number in selected:
        text = re.sub(r"\s+", " ", _page_text(reader.pages[number - 1])).strip()
        if len(text) > MAX_READ_CHARS:
            text = text[:MAX_READ_CHARS]
            truncated = True
        payload["pages"].append({"page": number, "text": text})
    payload["truncated"] = truncated
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
    print(_dump(read_corpus(args.path, root=root, pages=args.pages)))
    return 0


class PdfCorpusShell(ShellTools):
    """Shell tool that only reads PDFs under data/ril_data.

    run_shell_command stays on the class and is not offered to the model.
    search_pdfs and read_pdf build one argv and execute it with
    subprocess, which is the same on Linux and Windows.
    """

    def __init__(self, corpus_dir: Optional[Path] = None, timeout: int = 180, **kwargs):
        self.corpus_dir = Path(corpus_dir) if corpus_dir is not None else RIL_DATA_DIR
        super().__init__(
            base_dir=ROOT_DIR,
            enable_run_shell_command=False,
            timeout=timeout,
            **kwargs,
        )
        self.register(self.search_pdfs)
        self.register(self.read_pdf)

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

    def search_pdfs(self, query: str, folder: str = "", filename: str = "") -> str:
        """Search PDF text in data/ril_data with a shell command.

        The command runs the current Python interpreter and reads PDFs
        with pypdf from this environment. Pass the same arguments on
        Linux and Windows. Do not call grep, find, pdftotext, or PowerShell.

        Args:
            query: Words, clause, heading, or party from the current
                user message.
            folder: Folder under data/ril_data, with forward slashes.
                Examples: PSC, JOA/New JOA, sites/Regulatory/KGD6.
                Empty searches the whole ril_data tree.
            filename: Case-insensitive part of the PDF file name.
                Empty searches every PDF in folder.

        Returns:
            JSON hits with filename, folder_path, page, snippet, and
            doc_family. Cite folder_path and filename with the existing
            link rules.
        """
        args = self._command("search", f"--query={query or ''}")
        if folder:
            args.append(f"--folder={folder}")
        if filename:
            args.append(f"--filename={filename}")
        return self._run(args)

    def read_pdf(self, path: str, pages: str = "") -> str:
        """Read one PDF in data/ril_data with a shell command.

        Uses the current Python interpreter and pypdf. The command is
        the same on Linux and Windows.

        Args:
            path: PDF path relative to data/ril_data, forward slashes.
                Example: JOA/New JOA/file.pdf
            pages: 1-based pages, such as "4" or "4-6". Empty reads the
                first pages.

        Returns:
            JSON with filename, folder_path, and page text.
        """
        args = self._command("read", f"--path={path or ''}")
        if pages:
            args.append(f"--pages={pages}")
        return self._run(args)


if __name__ == "__main__":
    raise SystemExit(main())
