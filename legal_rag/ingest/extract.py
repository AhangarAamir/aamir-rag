"""Page text extraction and cleanup.

pdftotext (poppler) keeps reading order and word spacing better than pypdf
on contract scans, so it is used when installed; pypdf fills any page that
comes back empty. Repeated headers/footers and page-number lines are removed
so they do not split clauses or pollute search.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from agno.utils.log import log_warning

TEXT_SUFFIXES = {".txt", ".md", ".markdown"}
SUPPORTED_SUFFIXES = {".pdf", *TEXT_SUFFIXES}


@dataclass
class ExtractedDocument:
    pages: list[str]
    extractor: str
    empty_pages: list[int] = field(default_factory=list)


def _pdftotext(path: Path) -> list[str] | None:
    exe = shutil.which("pdftotext")
    if not exe:
        return None
    try:
        result = subprocess.run(
            [exe, "-enc", "UTF-8", str(path), "-"],
            capture_output=True,
            timeout=300,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        log_warning(f"pdftotext failed on {path.name}: {exc}")
        return None
    if result.returncode != 0:
        return None
    text = result.stdout.decode("utf-8", errors="replace")
    pages = text.split("\f")
    if pages and not pages[-1].strip():
        pages.pop()
    return pages


def _pypdf_pages(path: Path) -> list[str]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as exc:  # pragma: no cover - depends on file
            raise OSError("PDF is encrypted") from exc
    pages: list[str] = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if not text.strip():
            try:
                text = page.extract_text(orientations=(90, 270)) or ""
            except Exception:
                text = ""
        pages.append(text)
    return pages


def _text_file(path: Path) -> list[str]:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    # Treat form feeds as page breaks; otherwise ~3000-char pseudo pages.
    if "\f" in text:
        return text.split("\f")
    pages, current = [], []
    size = 0
    for line in text.splitlines():
        current.append(line)
        size += len(line) + 1
        if size >= 3000 and not line.strip():
            pages.append("\n".join(current))
            current, size = [], 0
    if current:
        pages.append("\n".join(current))
    return pages or [""]


def extract(path: Path) -> ExtractedDocument:
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        pages = _text_file(path)
        extractor = "text"
    else:
        primary = _pdftotext(path)
        if primary is not None:
            extractor = "pdftotext"
            pages = primary
            if any(not p.strip() for p in pages):
                try:
                    backup = _pypdf_pages(path)
                except Exception:
                    backup = []
                for i, text in enumerate(pages):
                    if not text.strip() and i < len(backup) and backup[i].strip():
                        pages[i] = backup[i]
                        extractor = "pdftotext+pypdf"
        else:
            pages = _pypdf_pages(path)
            extractor = "pypdf"
    pages = clean_pages(pages)
    empty = [i + 1 for i, text in enumerate(pages) if len(text.strip()) < 20]
    return ExtractedDocument(pages=pages, extractor=extractor, empty_pages=empty)


# ------------------------------------------------------------------ cleanup

_LIGATURES = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"}
_PAGE_NUMBER = re.compile(
    r"(?i)^\s*(?:page\s*)?[-–—(\[]?\s*\d{1,4}\s*[-–—)\]]?\s*(?:(?:of|/)\s*\d{1,4})?\s*$"
)


def _normalize_chars(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    for src, dst in _LIGATURES.items():
        text = text.replace(src, dst)
    text = text.replace(" ", " ").replace("­", "").replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"[ \t]+", " ", text)


def _edge_key(line: str) -> str:
    return re.sub(r"\d+", "#", line.strip().lower())


# Article / clause headings often open a page. They must never be taken for
# a running head just because "ARTICLE #" repeats once digits are masked.
_STRUCTURAL = re.compile(
    r"(?i)^\s*(?:article|section|clause|schedule|annex|annexure|appendix|exhibit|part)\b|^\s*\d{1,3}(?:\.\d{1,3})+\s"
)


def clean_pages(pages: list[str]) -> list[str]:
    pages = [_normalize_chars(p) for p in pages]
    lines_per_page = [[ln.rstrip() for ln in p.split("\n")] for p in pages]

    # Lines repeated at the top or bottom of many pages are running heads.
    edge_counts: Counter[str] = Counter()
    for lines in lines_per_page:
        content = [ln for ln in lines if ln.strip()]
        edges = {_edge_key(ln) for ln in content[:3] + content[-3:]}
        edge_counts.update(edges)
    threshold = max(3, int(len(pages) * 0.4))
    repeated = {key for key, count in edge_counts.items() if count >= threshold and len(key) < 160}

    cleaned: list[str] = []
    for lines in lines_per_page:
        content_idx = [i for i, ln in enumerate(lines) if ln.strip()]
        edge_idx = set(content_idx[:3] + content_idx[-3:])
        kept: list[str] = []
        for i, line in enumerate(lines):
            if i in edge_idx and not _STRUCTURAL.match(line) and (
                _edge_key(line) in repeated or _PAGE_NUMBER.match(line)
            ):
                continue
            kept.append(line)
        text = "\n".join(kept)
        # Re-join words hyphenated across a line break: "relin-\nquishment".
        text = re.sub(r"([a-z])-\n\s*([a-z])", r"\1\2", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        cleaned.append(text.strip("\n"))
    return cleaned
