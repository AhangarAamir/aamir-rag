"""Custom Agno chunking: inner agent returns legal metadata and a running summary.

Built-in AgenticChunking only returns a split integer. This strategy walks a
document window by window, calls a session-aware Agent with a Pydantic schema,
and stamps each chunk so retrieval can stay on the named instrument.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, IO, List, Optional, Sequence, Union

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.knowledge.chunking.strategy import ChunkingStrategy
from agno.knowledge.document.base import Document
from agno.knowledge.reader.markdown_reader import MarkdownReader
from agno.knowledge.reader.pdf_reader import PDFReader
from agno.knowledge.reader.text_reader import TextReader
from agno.models.openai import OpenAIResponses
from agno.utils.log import log_info, log_warning
from pydantic import BaseModel, Field

from document_catalog import (
    collection_by_code,
    identify_collection,
    normalize_family_code,
)
from config import (
    CHUNKING_MAX_SIZE,
    INCOMING_DIR,
    OPENAI_MODEL,
    ROOT_DIR,
    SESSIONS_DB_FILE,
)
from prompts import LEGAL_CHUNKER_INSTRUCTIONS

GENERIC_QUERY_WORDS = frozenset(
    {
        "psc",
        "joa",
        "jao",
        "pml",
        "pdf",
        "article",
        "clause",
        "section",
        "contract",
        "agreement",
        "the",
        "and",
        "for",
        "in",
        "of",
        "what",
        "states",
        "statues",
    }
)

_chunker_agent: Agent | None = None


class LegalChunkDecision(BaseModel):
    """Structured result from the inner chunking agent for one text window."""

    split_at: int = Field(
        0,
        description=(
            "Character index in THIS window where the current chunk should end. "
            "1..window_length. Prefer a complete numbered clause. "
            "Use the full window length when this is the last remaining text "
            "or the clause does not finish later in the window."
        ),
    )
    instrument_name: str = Field(
        "",
        description="Canonical short name, e.g. KGD6 PSC or NEC-OSN-97-2 JOA.",
    )
    instrument_aliases: List[str] = Field(
        default_factory=list,
        description="Hyphen/space/OCR variants of the instrument name.",
    )
    doc_family: str = Field(
        "",
        description="Catalog code: PSC, JOA, JAO, PML, RSC, PEL, FDP, DGH, MCM, …",
    )
    article: str = Field("", description="Article number only, e.g. 10.")
    clause_id: str = Field("", description="Numbered clause, e.g. 10.7.")
    heading: str = Field("", description="Exact heading attached to this chunk.")
    parties: List[str] = Field(default_factory=list)
    block: str = Field("", description="Block or contract area, e.g. KG-D6.")
    content_type: str = Field(
        "clause",
        description="clause, definition, table, toc, preamble, schedule, annex, other.",
    )
    document_status: str = Field(
        "",
        description="executed, draft, amendment, notification, minutes, playbook, unknown.",
    )
    cross_references: List[str] = Field(default_factory=list)
    defined_terms: List[str] = Field(default_factory=list)
    ocr_uncertainty: bool = False
    running_summary: str = Field(
        "",
        description=(
            "Updated summary of this document so far: identity, parties, block, "
            "current article, status. Keep instrument_name stable across windows."
        ),
    )


@dataclass(frozen=True)
class SourceOrigin:
    filename: str = ""
    display_name: str = ""
    source_path: str = ""
    folder_path: str = ""
    guessed_family: str = ""
    collection_description: str = ""


def normalize_clause(value: str) -> str:
    text = (value or "").strip()
    if not text:
        return ""
    match = re.search(r"\d+(?:\.\d+)*", text)
    return match.group(0) if match else text


def normalize_article(value: str) -> str:
    clause = normalize_clause(value)
    if not clause:
        return ""
    return clause.split(".", 1)[0]


def normalize_family(value: str) -> str:
    return normalize_family_code(value)


def instrument_key_tokens(name: str) -> List[str]:
    parts = re.split(r"[\s_/]+", (name or "").lower())
    keys: List[str] = []
    for part in parts:
        token = re.sub(r"[^a-z0-9-]", "", part.strip(".,()[]"))
        if not token or token in GENERIC_QUERY_WORDS:
            continue
        if len(token) < 3 and not any(ch.isdigit() for ch in token):
            continue
        keys.append(token)
    return keys


def primary_instrument_key(*names: str) -> str:
    for name in names:
        tokens = instrument_key_tokens(name)
        if tokens:
            return tokens[0]
    return ""


def guess_doc_family(*parts: str) -> str:
    hit = identify_collection(*parts)
    return hit.code if hit else ""


def _relative_folder(path: Path) -> str:
    parent = path.resolve().parent
    for root in (INCOMING_DIR.resolve(), ROOT_DIR.resolve()):
        try:
            relative = parent.relative_to(root)
            text = str(relative)
            return "" if text == "." else text
        except ValueError:
            continue
    return "/".join(parent.parts[-4:])


def origin_from_source(
    source: Union[str, Path, IO[Any], None],
    name: Optional[str] = None,
) -> SourceOrigin:
    path: Optional[Path] = None
    if isinstance(source, Path):
        path = source
    elif isinstance(source, str) and source.strip():
        path = Path(source)
    elif source is not None:
        raw_name = getattr(source, "name", "") or ""
        if raw_name:
            path = Path(raw_name)

    filename = ""
    if path is not None:
        filename = path.name
    elif name:
        filename = Path(name).name

    display = Path(name).stem if name else (path.stem if path is not None else filename)
    if display.lower().endswith(".pdf"):
        display = display[:-4]
    folder_path = _relative_folder(path) if path is not None else ""
    hit = identify_collection(folder_path, filename, display)
    family = hit.code if hit else ""
    return SourceOrigin(
        filename=filename or display,
        display_name=display or filename,
        source_path=str(path.resolve()) if path is not None and path.parts else "",
        folder_path=folder_path,
        guessed_family=family,
        collection_description=hit.description if hit else "",
    )


def apply_origin(document: Document, origin: SourceOrigin) -> None:
    meta = dict(document.meta_data or {})
    if origin.filename:
        meta.setdefault("filename", origin.filename)
    if origin.folder_path:
        meta.setdefault("folder_path", origin.folder_path)
    if origin.source_path:
        meta.setdefault("source_path", origin.source_path)
    if origin.guessed_family:
        meta.setdefault("doc_family", origin.guessed_family)
    if origin.collection_description:
        meta.setdefault("collection_description", origin.collection_description)
    if origin.display_name:
        document.name = origin.display_name
        meta.setdefault("document_title", origin.display_name)
    document.meta_data = meta


def locator_prefix(meta: Dict[str, Any], name: str = "") -> str:
    document = (
        meta.get("instrument_name")
        or meta.get("document_title")
        or name
        or meta.get("filename")
        or "unknown"
    )
    lines = [f"[Document: {document}]"]
    filename = meta.get("filename")
    if filename:
        lines.append(f"[Filename: {filename}]")
    family = meta.get("doc_family")
    if family:
        description = meta.get("collection_description")
        if description:
            lines.append(f"[Family: {family} | {description}]")
        else:
            lines.append(f"[Family: {family}]")
    folder = meta.get("folder_path")
    if folder:
        lines.append(f"[Folder: {folder}]")
    location_bits = []
    if meta.get("article"):
        location_bits.append(f"Article {meta['article']}")
    if meta.get("clause_id"):
        location_bits.append(f"Clause {meta['clause_id']}")
    if meta.get("heading"):
        location_bits.append(str(meta["heading"]))
    if location_bits:
        lines.append("[Location: " + " | ".join(location_bits) + "]")
    aliases = meta.get("instrument_aliases")
    if aliases:
        lines.append(f"[Aliases: {aliases}]")
    block = meta.get("block")
    if block:
        lines.append(f"[Block: {block}]")
    return "\n".join(lines) + "\n\n"


def _light_clean(text: str) -> str:
    cleaned = text.replace("\x00", "")
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _fallback_split(window: str, max_size: int) -> int:
    if len(window) <= max_size:
        return len(window)
    cut = window[:max_size]
    for sep in ("\n\n", "\n", ". ", "; ", " "):
        pos = cut.rfind(sep)
        if pos >= max(max_size // 4, 200):
            return pos + len(sep)
    return max_size


def _snap_split(window: str, split_at: int) -> int:
    n = len(window)
    if n == 0:
        return 0
    if split_at <= 0 or split_at > n:
        return _fallback_split(window, min(n, CHUNKING_MAX_SIZE))
    if split_at == n:
        return n
    if split_at < 120 and n > 120:
        return _fallback_split(window, min(n, CHUNKING_MAX_SIZE))
    if n - split_at < 80:
        return n
    if split_at < n and window[split_at].isalnum() and window[split_at - 1].isalnum():
        snapped = window.rfind("\n", 0, split_at)
        if snapped < split_at // 3:
            snapped = window.rfind(" ", 0, split_at)
        if snapped >= split_at // 3:
            return snapped + 1
    return split_at


def _join(values: Sequence[Any]) -> str:
    parts = [str(item).strip() for item in values if str(item).strip()]
    return ", ".join(parts)


def _chroma_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in meta.items():
        if value is None or value == "" or value == []:
            continue
        if isinstance(value, list):
            joined = _join(value)
            if joined:
                out[key] = joined
        else:
            out[key] = value
    return out


def _parse_decision(content: Any) -> Optional[LegalChunkDecision]:
    if isinstance(content, LegalChunkDecision):
        return content
    if isinstance(content, dict):
        try:
            return LegalChunkDecision.model_validate(content)
        except Exception:
            return None
    if isinstance(content, str):
        text = content.strip()
        if not text:
            return None
        try:
            return LegalChunkDecision.model_validate_json(text)
        except Exception:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                try:
                    return LegalChunkDecision.model_validate_json(text[start : end + 1])
                except Exception:
                    return None
    return None


def get_chunker_agent() -> Agent:
    global _chunker_agent
    if _chunker_agent is not None:
        return _chunker_agent
    _chunker_agent = Agent(
        id="legal-chunker",
        name="Legal Chunker",
        model=OpenAIResponses(id=OPENAI_MODEL),
        db=SqliteDb(
            id="chunker-sessions-db",
            db_file=str(SESSIONS_DB_FILE),
            session_table="chunker_sessions",
        ),
        output_schema=LegalChunkDecision,
        enable_session_summaries=True,
        add_session_summary_to_context=True,
        add_history_to_context=False,
        markdown=False,
        instructions=LEGAL_CHUNKER_INSTRUCTIONS,
    )
    return _chunker_agent


class LocatorChunking(ChunkingStrategy):
    """Run an inner splitter, then prefix every chunk with source identity."""

    def __init__(self, inner: ChunkingStrategy):
        self.inner = inner

    def chunk(self, document: Document) -> List[Document]:
        chunks = self.inner.chunk(document)
        name = (document.name or "").strip()
        log_info(f"Chunking document: {name} -> {len(chunks)} chunk(s)")
        for piece in chunks:
            meta = dict(document.meta_data or {})
            meta.update(piece.meta_data or {})
            piece.meta_data = _chroma_meta(meta)
            prefix = locator_prefix(piece.meta_data, name)
            if piece.content and not piece.content.startswith("[Document:"):
                piece.content = prefix + piece.content
        return chunks


class LegalAgenticChunking(ChunkingStrategy):
    """Walk windows, call the chunker agent, stamp metadata, keep a running summary."""

    def __init__(self, max_chunk_size: int = CHUNKING_MAX_SIZE):
        self.chunk_size = max_chunk_size

    def chunk(self, document: Document) -> List[Document]:
        text = _light_clean(document.content or "")
        if not text:
            return []

        origin_meta = dict(document.meta_data or {})
        name = (document.name or origin_meta.get("filename") or "document").strip()
        session_id = f"legal-chunk:{name}:{uuid.uuid4().hex[:10]}"
        agent = get_chunker_agent()

        remaining = text
        chunks: List[Document] = []
        chunk_number = 1
        last = LegalChunkDecision(
            instrument_name=str(origin_meta.get("instrument_name") or name),
            doc_family=str(origin_meta.get("doc_family") or ""),
            running_summary="",
        )

        log_info(f"Legal agentic chunking: {name} ({len(text)} chars), session {session_id}")

        while remaining:
            window = remaining[: self.chunk_size]
            is_tail = len(remaining) <= self.chunk_size
            decision = self._decide_window(
                agent=agent,
                session_id=session_id,
                document=document,
                window=window,
                remaining_len=len(remaining),
                last=last,
                is_tail=is_tail,
            )
            if decision is None:
                split_at = len(window) if is_tail else _fallback_split(window, self.chunk_size)
                decision = last.model_copy()
            else:
                last = decision
                split_at = len(window) if is_tail else _snap_split(window, decision.split_at)

            split_at = max(1, min(split_at, len(window)))
            chunk_text = remaining[:split_at].strip()
            remaining = remaining[split_at:].strip()
            if not chunk_text:
                if not remaining:
                    break
                remaining = remaining[1:]
                continue

            meta = self._chunk_metadata(
                document=document,
                decision=decision,
                chunk_number=chunk_number,
                chunk_text=chunk_text,
            )
            prefix = locator_prefix(meta, name)
            chunks.append(
                Document(
                    id=self._generate_chunk_id(document, chunk_number, chunk_text),
                    name=name,
                    meta_data=meta,
                    content=prefix + chunk_text,
                )
            )
            log_info(
                f"  chunk {chunk_number}: {meta.get('instrument_name')} "
                f"clause={meta.get('clause_id') or '-'} family={meta.get('doc_family') or '-'}"
            )
            chunk_number += 1

        log_info(f"Legal agentic chunking done: {name} -> {len(chunks)} chunk(s)")
        return chunks

    def _decide_window(
        self,
        *,
        agent: Agent,
        session_id: str,
        document: Document,
        window: str,
        remaining_len: int,
        last: LegalChunkDecision,
        is_tail: bool,
    ) -> Optional[LegalChunkDecision]:
        meta = document.meta_data or {}
        prompt = (
            f"SOURCE\n"
            f"- filename: {meta.get('filename') or document.name or ''}\n"
            f"- folder_path: {meta.get('folder_path') or ''}\n"
            f"- guessed_family: {meta.get('doc_family') or ''}\n"
            f"- document.name: {document.name or ''}\n\n"
            f"PRIOR DOCUMENT STATE\n"
            f"- instrument_name: {last.instrument_name}\n"
            f"- article: {last.article}\n"
            f"- clause_id: {last.clause_id}\n"
            f"- running_summary: {last.running_summary}\n\n"
            f"WINDOW\n"
            f"- window_length: {len(window)}\n"
            f"- remaining_characters: {remaining_len}\n"
            f"- is_last_window: {str(is_tail).lower()}\n"
            f"- split_at must be an index in this window, 1..{len(window)}\n\n"
            f"TEXT:\n---\n{window}\n---"
        )
        try:
            response = agent.run(prompt, session_id=session_id)
            return _parse_decision(getattr(response, "content", None))
        except Exception as exc:
            log_warning(f"Chunker agent failed, using fallback split: {exc}")
            return None

    def _chunk_metadata(
        self,
        *,
        document: Document,
        decision: LegalChunkDecision,
        chunk_number: int,
        chunk_text: str,
    ) -> Dict[str, Any]:
        origin = dict(document.meta_data or {})
        instrument = decision.instrument_name.strip() or str(
            origin.get("instrument_name") or document.name or ""
        )
        aliases = list(decision.instrument_aliases)
        if instrument and instrument not in aliases:
            aliases.insert(0, instrument)
        origin_family = str(origin.get("doc_family") or "")
        agent_family = normalize_family(decision.doc_family)
        family = origin_family or agent_family
        catalog = collection_by_code(family)
        clause = normalize_clause(decision.clause_id)
        article = normalize_article(decision.article) or (clause.split(".", 1)[0] if clause else "")
        meta: Dict[str, Any] = {
            **origin,
            "chunk": chunk_number,
            "chunk_size": len(chunk_text),
            "instrument_name": instrument,
            "instrument_aliases": _join(aliases),
            "instrument_key": primary_instrument_key(instrument, *aliases, str(document.name or "")),
            "doc_family": family,
            "collection_description": (
                (catalog.description if catalog else "")
                or str(origin.get("collection_description") or "")
            ),
            "article": article,
            "clause_id": clause,
            "heading": decision.heading.strip(),
            "parties": _join(decision.parties),
            "block": decision.block.strip(),
            "content_type": decision.content_type.strip() or "clause",
            "document_status": decision.document_status.strip(),
            "cross_references": _join(decision.cross_references),
            "defined_terms": _join(decision.defined_terms),
            "ocr_uncertainty": decision.ocr_uncertainty,
            "running_summary": decision.running_summary.strip(),
            "filename": origin.get("filename") or (document.name or ""),
            "document_title": origin.get("document_title") or instrument or (document.name or ""),
        }
        return _chroma_meta(meta)


def document_matches_instrument(document: Document, instrument_name: str) -> bool:
    tokens = instrument_key_tokens(instrument_name)
    if not tokens:
        return True
    blob_parts = [
        document.name or "",
        document.content or "",
        json.dumps(document.meta_data or {}, default=str),
    ]
    blob = " ".join(blob_parts).lower()
    slug = re.sub(r"[^a-z0-9]+", "", blob)
    return any(token.replace("-", "") in slug or token in blob for token in tokens)


def rank_documents(
    documents: List[Document],
    *,
    instrument_name: str = "",
    clause_id: str = "",
    doc_family: str = "",
) -> List[Document]:
    clause = normalize_clause(clause_id)
    family = normalize_family(doc_family)

    def score(doc: Document) -> tuple:
        meta = doc.meta_data or {}
        content = (doc.content or "").lower()
        points = 0
        if instrument_name and document_matches_instrument(doc, instrument_name):
            points += 20
        if clause and (str(meta.get("clause_id") or "") == clause or clause in content):
            points += 8
        if family and str(meta.get("doc_family") or "").upper() == family:
            points += 4
        return (-points, 0)

    return sorted(documents, key=score)


def build_chroma_filters(
    *,
    doc_family: str = "",
    clause_id: str = "",
    article: str = "",
) -> Optional[Dict[str, Any]]:
    filters: Dict[str, Any] = {}
    family = normalize_family(doc_family)
    clause = normalize_clause(clause_id)
    art = normalize_article(article)
    if family:
        filters["doc_family"] = family
    if clause:
        filters["clause_id"] = clause
    if art:
        filters["article"] = art
    return filters or None


def _stamp_then_chunk(
    documents: List[Document],
    origin: SourceOrigin,
    chunk: bool,
    reader: PDFReader | MarkdownReader | TextReader,
) -> List[Document]:
    for doc in documents:
        apply_origin(doc, origin)
    if chunk and documents:
        return reader._build_chunked_documents(documents) if hasattr(reader, "_build_chunked_documents") else [
            piece for doc in documents for piece in reader.chunk_document(doc)
        ]
    return documents


class PathAwarePDFReader(PDFReader):
    """Stamp folder/filename onto the PDF before the custom chunker runs."""

    def read(
        self,
        pdf: Optional[Union[str, Path, IO[Any]]] = None,
        name: Optional[str] = None,
        password: Optional[str] = None,
    ) -> List[Document]:
        origin = origin_from_source(pdf, name)
        was_chunk = self.chunk
        self.chunk = False
        try:
            documents = super().read(pdf, name=name, password=password)
        finally:
            self.chunk = was_chunk
        return _stamp_then_chunk(documents, origin, was_chunk, self)

    async def async_read(
        self,
        pdf: Optional[Union[str, Path, IO[Any]]] = None,
        name: Optional[str] = None,
        password: Optional[str] = None,
    ) -> List[Document]:
        origin = origin_from_source(pdf, name)
        was_chunk = self.chunk
        self.chunk = False
        try:
            documents = await super().async_read(pdf, name=name, password=password)
        finally:
            self.chunk = was_chunk
        return _stamp_then_chunk(documents, origin, was_chunk, self)


def _read_with_origin(
    reader: MarkdownReader | TextReader,
    file: Union[Path, IO[Any]],
    name: Optional[str],
    documents: List[Document],
    was_chunk: bool,
) -> List[Document]:
    origin = origin_from_source(file, name)
    for doc in documents:
        apply_origin(doc, origin)
    if was_chunk:
        return [piece for doc in documents for piece in reader.chunk_document(doc)]
    return documents


class PathAwareMarkdownReader(MarkdownReader):
    def read(self, file: Union[Path, IO[Any]], name: Optional[str] = None) -> List[Document]:
        was_chunk = self.chunk
        self.chunk = False
        try:
            documents = super().read(file, name=name)
        finally:
            self.chunk = was_chunk
        return _read_with_origin(self, file, name, documents, was_chunk)

    async def async_read(self, file: Union[Path, IO[Any]], name: Optional[str] = None) -> List[Document]:
        was_chunk = self.chunk
        self.chunk = False
        try:
            documents = await super().async_read(file, name=name)
        finally:
            self.chunk = was_chunk
        return _read_with_origin(self, file, name, documents, was_chunk)


class PathAwareTextReader(TextReader):
    def read(self, file: Union[Path, IO[Any]], name: Optional[str] = None) -> List[Document]:
        was_chunk = self.chunk
        self.chunk = False
        try:
            documents = super().read(file, name=name)
        finally:
            self.chunk = was_chunk
        return _read_with_origin(self, file, name, documents, was_chunk)

    async def async_read(self, file: Union[Path, IO[Any]], name: Optional[str] = None) -> List[Document]:
        was_chunk = self.chunk
        self.chunk = False
        try:
            documents = await super().async_read(file, name=name)
        finally:
            self.chunk = was_chunk
        return _read_with_origin(self, file, name, documents, was_chunk)
