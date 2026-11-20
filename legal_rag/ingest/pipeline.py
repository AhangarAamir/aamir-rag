"""Ingest files into the index.

Per document:
  1. extract page text (pdftotext / pypdf), drop running heads
  2. parse legal structure into clause / definition units (no model)
  3. profile the document: path heuristics, plus one optional LLM call on the
     opening pages for title, parties, block, date and status
  4. prefix each chunk with a context header (instrument, article, clause)
     so both keyword and vector search see where the text sits
  5. embed (context + text) with the OpenAI embedding model
  6. store the document's Document Center link in documents.link

The model is called once per document (not once per chunk).
"""

from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from agno.utils.log import log_info, log_warning

from legal_rag.config import CHUNK_MAX_CHARS, CORPUS_DIR, INGEST_LLM_PROFILE, INGEST_WORKERS
from legal_rag.domain import document_link, family_by_code, family_for, glossary_variants, identifier_keys, squash, unique
from legal_rag.index import store
from legal_rag.knowledge.import_log import write_document_log
from legal_rag.index.vectors import embed_texts, to_blob
from legal_rag.ingest.extract import SUPPORTED_SUFFIXES, extract
from legal_rag.ingest.structure import Unit, is_markdown, parse, parse_markdown
from legal_rag.models import EmbeddingSettings, complete, embedding_settings

Progress = Callable[[dict[str, Any]], None]


@dataclass
class IngestReport:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[dict[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    chunks: int = 0
    pages: int = 0

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


# ------------------------------------------------------------------ discovery

def resolve_corpus_path(path: str | Path | None) -> Path:
    corpus = CORPUS_DIR.resolve()
    if path is None or str(path).strip() in {"", "."}:
        return corpus
    raw = Path(str(path).strip()).expanduser()
    target = (raw if raw.is_absolute() else corpus / raw).resolve()
    if target != corpus and corpus not in target.parents:
        raise ValueError(
            f"{target} is outside the corpus folder {corpus}. "
            "Copy the files there or set CORPUS_DIR in .env."
        )
    if not target.exists():
        raise FileNotFoundError(f"{target} does not exist")
    return target


def discover(path: str | Path | None = None) -> list[Path]:
    target = resolve_corpus_path(path)
    if target.is_file():
        return [target] if target.suffix.lower() in SUPPORTED_SUFFIXES else []
    files = [
        p for p in target.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES and not any(part.startswith(".") for part in p.relative_to(target).parts)
    ]
    return sorted(files)


def rel_path_of(path: Path) -> str:
    return path.resolve().relative_to(CORPUS_DIR.resolve()).as_posix()


def _sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ------------------------------------------------------------------ profile

_TITLE_WORDS = re.compile(
    r"(?i)\b(contract|agreement|lease|licen[cs]e|minutes|notification|letter|amendment|addendum|"
    r"resolution|plan|report|deed|memorandum|approval|order|budget|statement)\b"
)
_BLOCK_RE = re.compile(r"\b[A-Z]{2,3}-[A-Z]{2,5}-\d{2,4}/\d{1,2}\b|\bKG-?D6\b|\bNEC-?25\b|\bKG-UDWHP-\d{4}/\d\b")
_DATE_RE = re.compile(
    r"\b(\d{1,2}(?:st|nd|rd|th)?\s+(?:day\s+of\s+)?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*,?\s+\d{4}"
    r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}"
    r"|\d{1,2}[./-]\d{1,2}[./-]\d{4})\b"
)


def _heuristic_profile(path: Path, rel: str, pages: list[str]) -> dict[str, Any]:
    folder = str(Path(rel).parent.as_posix()) if "/" in rel else ""
    family = family_for(folder, path.name)
    head = "\n".join(pages[:2])
    title = ""
    lines = [ln.strip() for ln in head.split("\n") if ln.strip()][:50]
    for idx, line in enumerate(lines):
        letters = [c for c in line if c.isalpha()]
        if len(letters) < 8 or len(line) > 140:
            continue
        upper = sum(c.isupper() for c in letters) / len(letters)
        if upper > 0.7 and _TITLE_WORDS.search(line):
            title = line
            if idx + 1 < len(lines) and len(lines[idx + 1]) < 100:
                nxt = lines[idx + 1]
                nl = [c for c in nxt if c.isalpha()]
                if nl and sum(c.isupper() for c in nl) / len(nl) > 0.7 and not _TITLE_WORDS.search(nxt):
                    title = f"{title} {nxt}"
            break
    stem = re.sub(r"[_]+", " ", path.stem).strip()
    blocks = unique(m.group(0) for m in _BLOCK_RE.finditer(head + " " + rel))
    date = _DATE_RE.search(head)
    # Path and title words decide; a body that merely mentions "Amendment No. 1" does not.
    status = ""
    words = (("draft", "draft"), ("amendment", "amendment"), ("addendum", "amendment"),
             ("minutes", "minutes"), ("notification", "notification"), ("resolution", "resolution"))
    for source in (rel.lower(), title.lower(), head[:300].lower()):
        for word, label in words:
            if word in source:
                status = label
                break
        if status:
            break
    return {
        "family": family.code if family else "",
        "family_desc": family.description if family else "",
        "title": re.sub(r"\s+", " ", title.title() if title.isupper() else title) or stem,
        "short_name": "",
        "aliases": unique([stem, *blocks]),
        "parties": [],
        "block": blocks[0] if blocks else "",
        "doc_date": date.group(0) if date else "",
        "status": status,
        "summary": "",
        "folder": folder,
    }


_PROFILE_PROMPT = """Read the opening pages of a legal / regulatory document and return JSON only:
{"title": "...", "short_name": "...", "aliases": ["..."], "parties": ["..."],
 "block": "...", "document_date": "...", "status": "...", "summary": "..."}
- title: the document's own title as printed.
- short_name: how lawyers would name it, e.g. "KG-D6 PSC", "NEC-OSN-97/2 JOA", "MCM minutes 12 March 2019".
- aliases: other names, codes, spellings and hyphen/space variants (KGD6, KG-D6, KG-DWN-98/3).
- parties: legal names of the parties / signatories.
- block: contract area or block code if stated.
- document_date: execution or issue date as printed.
- status: one of executed, draft, amendment, notification, minutes, resolution, letter, report, other.
- summary: two sentences on what the document is and does.
Use "" or [] when the text does not say. Never guess.

FILE: {file}
FOLDER: {folder}
TEXT:
---
{text}
---"""


def _llm_profile(path: Path, folder: str, pages: list[str]) -> Optional[dict[str, Any]]:
    text = "\n".join(pages[:4])[:7000]
    if len(text.strip()) < 50:
        return None
    prompt = _PROFILE_PROMPT.replace("{file}", path.name).replace("{folder}", folder).replace("{text}", text)
    # Reasoning models spend output tokens on reasoning first; give them room.
    try:
        raw, _ = complete(prompt, json_mode=True, max_tokens=4000)
    except Exception as exc:
        log_warning(f"Document profile call failed for {path.name}: {exc}")
        return None
    raw = re.sub(r"(?s)<think>.*?</think>", "", raw)
    match = re.search(r"(?s)\{.*\}", raw)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def build_profile(path: Path, rel: str, pages: list[str], use_llm: bool) -> tuple[dict[str, Any], str]:
    profile = _heuristic_profile(path, rel, pages)
    source = "heuristic"
    if use_llm:
        llm = _llm_profile(path, profile["folder"], pages)
        if llm:
            source = "llm"
            as_text = lambda v: v.strip() if isinstance(v, str) else ""  # noqa: E731
            as_list = lambda v: [str(x).strip() for x in v if str(x).strip()] if isinstance(v, list) else []  # noqa: E731
            profile["title"] = as_text(llm.get("title")) or profile["title"]
            profile["short_name"] = as_text(llm.get("short_name"))
            profile["parties"] = as_list(llm.get("parties"))[:12]
            profile["block"] = as_text(llm.get("block")) or profile["block"]
            # Small models misread dates and status. Keep a date found verbatim
            # by regex; accept an LLM date only if its digits are in the text.
            head = " ".join(pages[:4])
            llm_date = as_text(llm.get("document_date"))
            digits = re.findall(r"\d+", llm_date)
            if not profile["doc_date"] and llm_date and digits and all(d in head for d in digits):
                profile["doc_date"] = llm_date
            # Accept a status only when the document itself uses that word.
            status = as_text(llm.get("status")).lower()
            stems = {"amendment": "amend", "notification": "notif", "minutes": "minute",
                     "resolution": "resolution", "draft": "draft", "letter": "", "report": "report"}
            if not profile["status"] and status in stems:
                stem = stems[status]
                if not stem or stem in (head + " " + rel).lower():
                    profile["status"] = status
            elif not profile["status"] and status == "executed":
                profile["status"] = "executed"
            profile["summary"] = as_text(llm.get("summary"))[:600]
            profile["aliases"] = unique([*profile["aliases"], *as_list(llm.get("aliases"))])[:20]
    family = family_by_code(profile["family"])
    names = [profile["short_name"], profile["title"], profile["block"], *profile["aliases"]]
    if family and profile["block"]:
        names.append(f"{profile['block']} {family.code}")
    profile["aliases"] = unique(n for n in names if n)[:24]
    return profile, source


def alias_keys(profile: dict[str, Any], filename: str) -> str:
    keys: set[str] = set()
    names = [*profile["aliases"], profile["title"], Path(filename).stem, profile["family"]]
    # KG-D6 == KGD6 == KG-DWN-98/3, NEC25 == NEC-OSN-97/2, PSC == Production Sharing Contract ...
    names += glossary_variants(" | ".join(n for n in names if n))
    for name in names:
        keys |= identifier_keys(name)
        sq = squash(name)
        if 2 <= len(sq) <= 40:
            keys.add(sq)
        for word in re.findall(r"[A-Za-z0-9]+", name or ""):
            if len(word) >= 2:
                keys.add(word.lower())
    return " ".join(sorted(keys))


# ------------------------------------------------------------------ chunks

def context_header(profile: dict[str, Any], filename: str, unit: Unit) -> str:
    name = profile["short_name"] or profile["title"] or filename
    family = f" ({profile['family']})" if profile["family"] else ""
    lines = [f"{name}{family} | {filename}"]
    where = []
    if unit.breadcrumb:
        lines.append(f"Section: {unit.breadcrumb}")
    if unit.article:
        label = f"Article {unit.article}" if re.fullmatch(r"\d{1,3}[A-Z]?", unit.article) else unit.article
        where.append(f"{label}" + (f" – {unit.article_title}" if unit.article_title else ""))
    if unit.kind == "definition" and unit.defined_term:
        where.append(f'Definition “{unit.defined_term}”' + (f" (clause {unit.clause_id})" if unit.clause_id else ""))
    elif unit.clause_id:
        where.append(f"Clause {unit.clause_id}" + (f" {unit.heading}" if unit.heading else ""))
    elif unit.kind == "toc":
        where.append("Table of contents")
    if unit.parts > 1:
        where.append(f"part {unit.part}/{unit.parts}")
    if where:
        lines.append(" › ".join(where))
    return "\n".join(lines)


def chunk_keys(unit: Unit, header: str, doc_keys: str) -> str:
    keys = set(doc_keys.split()) | identifier_keys(header + "\n" + unit.text)
    if unit.clause_id:
        keys.add("cl" + unit.clause_id.replace(".", "p"))
        keys.add("art" + unit.clause_id.split(".")[0])
    if unit.article:
        keys.add("art" + squash(unit.article))
    if unit.defined_term:
        keys.add("def" + squash(unit.defined_term))
    for ref in re.findall(r"(?i)\b(?:article|clause|section)\s+(\d{1,3}(?:\.\d{1,3}){0,3})", unit.text):
        keys.add("ref" + ref.replace(".", "p"))
    return " ".join(sorted(keys))


# ------------------------------------------------------------------ ingest

def embedding_hint(exc: Exception, cfg: EmbeddingSettings) -> str:
    """A failure message that says what to do about it."""
    text = str(exc)
    if "not found" in text.lower() and ("404" in text or "model" in text.lower()):
        return (f"OpenAI embedding model '{cfg.model}' was not found. Check EMBEDDING_MODEL in .env, "
                "then run `uv run python main.py reembed`. Keyword search works meanwhile.")
    return f"{text}. Keyword search still works; re-embed later."


def embedding_preflight() -> Optional[str]:
    """One tiny call before a batch, so a broken embedding endpoint fails once, not per file."""
    cfg = embedding_settings()
    if not cfg.enabled:
        return None
    try:
        embed_texts(["ping"], cfg)
        return None
    except Exception as exc:
        return embedding_hint(exc, cfg)


def ingest_file(
    path: Path, *, force: bool = False, use_llm: Optional[bool] = None, embed: bool = True
) -> tuple[str, dict[str, Any]]:
    """Returns (status, info). status is added, updated or skipped."""
    rel = rel_path_of(path)
    stat = path.stat()
    existing = store.find_document_by_path(rel)
    if existing and not force and existing["size"] == stat.st_size and existing["mtime_ns"] == stat.st_mtime_ns:
        return "skipped", {"rel_path": rel}
    sha1 = _sha1(path)
    if existing and not force and existing["sha1"] == sha1:
        return "skipped", {"rel_path": rel}

    extracted = extract(path)
    markdown = path.suffix.lower() in {".md", ".markdown"} or (
        path.suffix.lower() == ".txt" and is_markdown(extracted.pages)
    )
    outline = (parse_markdown if markdown else parse)(extracted.pages, max_chars=CHUNK_MAX_CHARS)
    profile, source = build_profile(path, rel, extracted.pages, INGEST_LLM_PROFILE if use_llm is None else use_llm)
    doc_keys = alias_keys(profile, path.name)

    rows = []
    for unit in outline.units:
        header = context_header(profile, path.name, unit)
        rows.append((unit, header, chunk_keys(unit, header, doc_keys)))

    warnings: list[str] = []
    vectors = None
    cfg = embedding_settings()
    if cfg.enabled and rows and embed:
        try:
            vectors = embed_texts([f"{h}\n{u.text}" for u, h, _ in rows], cfg)
        except Exception as exc:
            warnings.append(f"Embeddings skipped: {embedding_hint(exc, cfg)}")
            log_warning(f"Embedding failed for {path.name}: {exc}")

    with store.tx() as conn:
        if existing:
            ids = [r["id"] for r in conn.execute("SELECT id FROM chunks WHERE doc_id=?", (existing["id"],))]
            conn.executemany("DELETE FROM chunks_fts WHERE rowid=?", [(i,) for i in ids])
            conn.execute("DELETE FROM documents WHERE id=?", (existing["id"],))
        cur = conn.execute(
            """INSERT INTO documents(rel_path, filename, folder, link, family, family_desc, title, short_name, aliases,
               alias_keys, parties, block, doc_date, status, summary, pages, chars, chunks, extractor, empty_pages,
               sha1, size, mtime_ns, profile_source, ingested_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                rel, path.name, profile["folder"], document_link(rel), profile["family"], profile["family_desc"], profile["title"],
                profile["short_name"], json.dumps(profile["aliases"]), doc_keys, json.dumps(profile["parties"]),
                profile["block"], profile["doc_date"], profile["status"], profile["summary"],
                len(extracted.pages), sum(len(p) for p in extracted.pages), len(rows), extracted.extractor,
                json.dumps(extracted.empty_pages), sha1, stat.st_size, stat.st_mtime_ns, source, store.now_iso(),
            ),
        )
        doc_id = cur.lastrowid
        conn.executemany(
            "INSERT INTO pages(doc_id, page_no, text) VALUES(?,?,?)",
            [(doc_id, n, text) for n, text in enumerate(extracted.pages, start=1)],
        )
        conn.executemany(
            "INSERT INTO articles(doc_id, article, title, page_start, page_end) VALUES(?,?,?,?,?)",
            [(doc_id, a["article"], a["title"], a["page_start"], a["page_end"]) for a in outline.articles],
        )
        for ord_, (unit, header, keys) in enumerate(rows):
            cur = conn.execute(
                """INSERT INTO chunks(doc_id, ord, kind, article, article_title, clause_id, heading, defined_term,
                   page_start, page_end, part, parts, context, text) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    doc_id, ord_, unit.kind, unit.article, unit.article_title, unit.clause_id, unit.heading,
                    unit.defined_term, unit.page_start, unit.page_end, unit.part, unit.parts, header, unit.text,
                ),
            )
            chunk_id = cur.lastrowid
            conn.execute(
                "INSERT INTO chunks_fts(rowid, text, context, keys) VALUES(?,?,?,?)",
                (chunk_id, unit.text, header, keys),
            )
            if vectors is not None:
                conn.execute(
                    "INSERT INTO embeddings(chunk_id, model, dim, vec) VALUES(?,?,?,?)",
                    (chunk_id, cfg.model, int(vectors.shape[1]), to_blob(vectors[ord_])),
                )
        if vectors is not None:
            store.set_meta("embedding_model", cfg.model, conn)
        store.bump_version(conn)

    if extracted.empty_pages:
        warnings.append(
            f"{len(extracted.empty_pages)} page(s) have no text layer (scanned?): "
            f"{extracted.empty_pages[:10]}{'…' if len(extracted.empty_pages) > 10 else ''}. OCR them to make them searchable."
        )
    info = {
        "rel_path": rel,
        "doc_id": doc_id,
        "pages": len(extracted.pages),
        "chunks": len(rows),
        "clauses": sum(1 for u, _, _ in rows if u.kind == "clause"),
        "definitions": sum(1 for u, _, _ in rows if u.kind == "definition"),
        "profile": source,
        "warnings": warnings,
    }
    log_info(f"Ingested {rel}: {info['pages']} pages, {info['chunks']} chunks ({source} profile)")
    return ("updated" if existing else "added"), info


def ingest_path(
    path: str | Path | None = None,
    *,
    force: bool = False,
    use_llm: Optional[bool] = None,
    on_progress: Optional[Progress] = None,
    workers: Optional[int] = None,
) -> IngestReport:
    report = IngestReport()
    files = discover(path)
    total = len(files)
    problem = embedding_preflight() if files else None
    if problem:
        report.warnings.append(f"Embeddings skipped for this run: {problem}")
        log_warning(f"Embeddings skipped for this run: {problem}")
    def work(file: Path) -> tuple[str, str, dict[str, Any]]:
        rel = rel_path_of(file)
        try:
            status, info = ingest_file(file, force=force, use_llm=use_llm, embed=problem is None)
        except Exception as exc:  # keep going; report every failure
            log_warning(f"Ingest failed for {rel}: {exc}")
            status, info = "failed", {"error": f"{type(exc).__name__}: {exc}"}
        if status != "skipped":
            _log_document(file, status, info)
        return rel, status, info

    # Each file commits on its own, so an interrupted run resumes: finished files are skipped.
    with ThreadPoolExecutor(max_workers=min(workers or INGEST_WORKERS, max(1, total))) as pool:
        futures = [pool.submit(work, f) for f in files]
        for done, future in enumerate(as_completed(futures), start=1):
            rel, status, info = future.result()
            if on_progress:
                on_progress({"type": "file", "index": done, "total": total, "file": rel})
            if status == "failed":
                report.failed.append({"file": rel, "error": info["error"]})
                continue
            getattr(report, status).append(rel)
            report.chunks += info.get("chunks", 0)
            report.pages += info.get("pages", 0)
            report.warnings.extend(f"{rel}: {w}" for w in info.get("warnings", []))
    # A whole-corpus scan also drops entries for files that were moved or deleted,
    # so a moved file is not left behind with its old path and link.
    if path is None or str(path).strip() in {"", "."}:
        report.removed = remove_missing()
    if on_progress:
        on_progress({"type": "done", "total": total})
    return report


def _log_document(file: Path, status: str, info: dict[str, Any]) -> None:
    """data/import_logs/<file name>.log, written as soon as the file finishes."""
    if status == "failed":
        message = info.get("error", "")
    else:
        message = (
            f"pages {info.get('pages', 0)}, chunks {info.get('chunks', 0)}, clauses {info.get('clauses', 0)}, "
            f"definitions {info.get('definitions', 0)}, profile {info.get('profile', '')}"
        )
        if info.get("warnings"):
            message += "; warnings: " + " | ".join(info["warnings"])
    try:
        write_document_log(file_name=file.name, source_path=rel_path_of(file), status=status, message=message)
    except OSError as exc:
        log_warning(f"Could not write import log for {file.name}: {exc}")


def remove_missing() -> list[str]:
    """Drop index rows whose file no longer exists on disk."""
    removed = []
    for doc in store.list_documents():
        if not (CORPUS_DIR / doc["rel_path"]).is_file():
            store.delete_document(doc["id"])
            removed.append(doc["rel_path"])
    return removed


def reembed(on_progress: Optional[Progress] = None, batch: int = 256, workers: Optional[int] = None) -> dict[str, Any]:
    """Embed every chunk that has no vector for the configured model.

    Batches are embedded in parallel and written as they finish, so an
    interrupted run keeps what it saved and the next run only does the rest.
    """
    cfg = embedding_settings()
    if not cfg.enabled:
        return {"embedded": 0, "note": "Embeddings are disabled (EMBED_ENABLED=false)."}
    with store.tx() as conn:
        conn.execute("DELETE FROM embeddings WHERE model != ?", (cfg.model,))
    rows = store.query(
        "SELECT c.id, c.context, c.text FROM chunks c LEFT JOIN embeddings e ON e.chunk_id = c.id"
        " WHERE e.chunk_id IS NULL ORDER BY c.id"
    )
    done = 0
    parts = [rows[start:start + batch] for start in range(0, len(rows), batch)]

    def embed(part):
        return part, embed_texts([f"{r['context']}\n{r['text']}" for r in part], cfg)

    pool = ThreadPoolExecutor(max_workers=max(1, min(workers or INGEST_WORKERS, len(parts) or 1)))
    for future in as_completed([pool.submit(embed, p) for p in parts]):
        part, vectors = future.result()
        with store.tx() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO embeddings(chunk_id, model, dim, vec) VALUES(?,?,?,?)",
                [(r["id"], cfg.model, int(vectors.shape[1]), to_blob(v)) for r, v in zip(part, vectors)],
            )
            store.set_meta("embedding_model", cfg.model, conn)
            store.bump_version(conn)
        done += len(part)
        if on_progress:
            on_progress({"type": "embed", "done": done, "total": len(rows)})
    pool.shutdown()
    return {"embedded": done, "model": cfg.model}
