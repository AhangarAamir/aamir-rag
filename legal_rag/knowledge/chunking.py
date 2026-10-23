"""Custom Agno chunking: inner agent returns legal metadata and a running summary.

Built-in AgenticChunking only returns a split integer. This strategy walks a
document window by window, calls a session-aware Agent with a Pydantic schema,
and stamps each chunk so retrieval can stay on the named instrument.
"""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, IO, List, Optional, Sequence, Union

from agno.agent import Agent
from agno.knowledge.chunking.recursive import RecursiveChunking
from agno.knowledge.chunking.strategy import ChunkingStrategy
from agno.knowledge.document.base import Document
from agno.knowledge.reader.excel_reader import ExcelReader
from agno.knowledge.reader.markdown_reader import MarkdownReader
from agno.knowledge.reader.pdf_reader import PDFReader
from agno.knowledge.reader.text_reader import TextReader
from agno.models.openai import OpenAIResponses
from agno.utils.log import log_info, log_warning
from pydantic import BaseModel, Field

from legal_rag.knowledge.catalog import (
    classify_sites_path,
    collection_by_code,
    identify_collection,
    normalize_family_code,
)
from legal_rag.config import (
    CHUNKER_ASYNC,
    CHUNKER_CONCURRENCY,
    CHUNKING_MAX_SIZE,
    INCOMING_DIR,
    OPENAI_MODEL,
    ROOT_DIR,
)
from legal_rag.knowledge.text_encoding import open_decoded, tag_encoding
from legal_rag.prompts import LEGAL_CHUNKER_INSTRUCTIONS

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

_CLAUSE_LINE = re.compile(
    r"(?im)^\s*(?:article|clause|section)\s+(\d+(?:\.\d+)*)\b([^\n]{0,80})"
)
_NUMBERED_LINE = re.compile(r"(?m)^\s*(\d+\.\d+(?:\.\d+)*)\b([^\n]{0,80})")

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
    block: str = ""
    category: str = ""


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


# Local workspace dump copied under a family folder. It is not a Document Center path.
_WORKSPACE_PATH = re.compile(
    r"aamirWS/data/ril_data/(?:JOA|JAO|PEL|PML|PSC|RSC)/?",
    re.IGNORECASE,
)


def strip_workspace_path(text: str) -> str:
    """Drop aamirWS/data/ril_data/<family> so citations stay family/file."""
    if not text or "aamirws" not in text.lower():
        return text
    return _WORKSPACE_PATH.sub("", text)


def sites_relative_folder(path: Path) -> Optional[str]:
    """Parent folders from the ``sites`` segment down.

    Windows separators are accepted. Returns None when the path has no
    ``sites`` folder, so JOA/PSC paths keep the Document Center rule.
    """
    parts = [part for part in str(path).replace("\\", "/").split("/") if part and part not in {".", ".."}]
    if len(parts) < 2:
        return None
    folders = parts[:-1]
    for index, part in enumerate(folders):
        if part.lower() == "sites":
            return "/".join(folders[index:])
    return None


def _relative_folder(path: Path) -> str:
    anchored = sites_relative_folder(path)
    if anchored is not None:
        return anchored
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
    raw_path = ""
    if isinstance(source, Path):
        raw_path = str(source)
    elif isinstance(source, str) and source.strip():
        raw_path = source.strip()
    elif source is not None:
        raw_path = getattr(source, "name", "") or ""
    if raw_path:
        path = Path(raw_path.replace("\\", "/"))

    filename = ""
    if path is not None:
        filename = path.name
    elif name:
        filename = Path(name).name

    display = Path(name).stem if name else (path.stem if path is not None else filename)
    for suffix in (".pdf", ".xlsx", ".xls", ".xlsm"):
        if display.lower().endswith(suffix):
            display = display[: -len(suffix)]
            break
    folder_path = strip_workspace_path(_relative_folder(path)) if path is not None else ""
    placement = classify_sites_path(folder_path, filename, display)
    if placement is not None:
        hit = collection_by_code(placement.code)
        family = placement.code
        description = hit.description if hit else ""
        block = placement.block
        category = placement.category
    else:
        hit = identify_collection(folder_path, filename, display)
        family = hit.code if hit else ""
        description = hit.description if hit else ""
        block = ""
        category = ""
    return SourceOrigin(
        filename=filename or display,
        display_name=display or filename,
        source_path=str(path.resolve()) if path is not None and path.parts else "",
        folder_path=folder_path,
        guessed_family=family,
        collection_description=description,
        block=block,
        category=category,
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
    if origin.block:
        meta.setdefault("block", origin.block)
    if origin.category:
        meta.setdefault("sites_category", origin.category)
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
    folder = strip_workspace_path(str(meta.get("folder_path") or ""))
    if folder:
        lines.append(f"[Folder: {folder}]")
    category = meta.get("sites_category")
    if category:
        lines.append(f"[Category: {category}]")
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


def _run_sync(coro):
    """Run an async coroutine from sync ingest (no running loop) or a worker thread."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _heuristic_clause(text: str) -> tuple[str, str, str]:
    match = _CLAUSE_LINE.search(text or "")
    if match:
        clause = normalize_clause(match.group(1))
        heading = match.group(2).strip(" :-")
        return clause, normalize_article(clause), heading
    match = _NUMBERED_LINE.search(text or "")
    if match:
        clause = normalize_clause(match.group(1))
        heading = match.group(2).strip(" :-")
        return clause, normalize_article(clause), heading
    return "", "", ""


def get_chunker_agent() -> Agent:
    global _chunker_agent
    if _chunker_agent is not None:
        return _chunker_agent
    # No session DB: parallel arun() must not contend on SQLite, and per-chunk
    # session summaries would add a second LLM call each. Document identity is
    # captured once and passed into every label prompt.
    _chunker_agent = Agent(
        id="legal-chunker",
        name="Legal Chunker",
        model=OpenAIResponses(id=OPENAI_MODEL),
        output_schema=LegalChunkDecision,
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


async def _label_one(
    strategy: "LegalAgenticChunking",
    *,
    agent: Agent,
    document: Document,
    piece: Document,
    index: int,
    count: int,
    identity: LegalChunkDecision,
) -> LegalChunkDecision:
    decision = await strategy._alabel(
        agent=agent,
        document=document,
        window=piece.content or "",
        remaining_len=len(piece.content or ""),
        last=identity,
        is_tail=index == count - 1,
        role="label",
        chunk_index=index + 1,
        chunk_count=count,
    )
    if decision is None:
        clause, article, heading = _heuristic_clause(piece.content or "")
        return identity.model_copy(
            update={
                "split_at": len(piece.content or ""),
                "clause_id": clause,
                "article": article,
                "heading": heading,
            }
        )
    decision.instrument_name = decision.instrument_name or identity.instrument_name
    decision.instrument_aliases = decision.instrument_aliases or list(identity.instrument_aliases)
    decision.doc_family = decision.doc_family or identity.doc_family
    decision.parties = decision.parties or list(identity.parties)
    decision.block = decision.block or identity.block
    decision.document_status = decision.document_status or identity.document_status
    decision.running_summary = identity.running_summary
    return decision


async def _label_in_batches(
    strategy: "LegalAgenticChunking",
    *,
    agent: Agent,
    document: Document,
    pieces: List[Document],
    identity: LegalChunkDecision,
) -> List[LegalChunkDecision]:
    """Label at most CHUNKER_CONCURRENCY pieces at once, then the next batch."""
    labeled: List[LegalChunkDecision] = []
    count = len(pieces)
    batch_size = CHUNKER_CONCURRENCY
    total_batches = (count + batch_size - 1) // batch_size
    for batch_number, offset in enumerate(range(0, count, batch_size), start=1):
        batch = pieces[offset : offset + batch_size]
        last = offset + len(batch)
        log_info(
            f"Legal async chunking batch {batch_number}/{total_batches}: "
            f"chunks {offset + 1}-{last} of {count}"
        )
        labeled.extend(
            await asyncio.gather(
                *[
                    _label_one(
                        strategy,
                        agent=agent,
                        document=document,
                        piece=piece,
                        index=offset + index,
                        count=count,
                        identity=identity,
                    )
                    for index, piece in enumerate(batch)
                ]
            )
        )
    return labeled


class LegalAgenticChunking(ChunkingStrategy):
    """Split locally, then label chunks with parallel async agent calls.

    Sequential window-by-window LLM splits are slow. Default path:
    1. Recursive split (no model).
    2. One identity call on the document head.
    3. Metadata labeling in batches of CHUNKER_CONCURRENCY (default 100).
       The next batch starts only after the current batch finishes.
    """

    def __init__(self, max_chunk_size: int = CHUNKING_MAX_SIZE):
        self.chunk_size = max_chunk_size
        self._splitter = RecursiveChunking(chunk_size=max_chunk_size, overlap=200)

    def chunk(self, document: Document) -> List[Document]:
        return _run_sync(self.achunk(document))

    async def achunk(self, document: Document) -> List[Document]:
        if CHUNKER_ASYNC:
            return await self._chunk_parallel(document)
        return await self._chunk_sequential(document)

    async def _chunk_parallel(self, document: Document) -> List[Document]:
        text = _light_clean(document.content or "")
        if not text:
            return []

        origin_meta = dict(document.meta_data or {})
        name = (document.name or origin_meta.get("filename") or "document").strip()
        split_source = Document(
            name=name,
            id=document.id,
            meta_data=origin_meta,
            content=text,
        )
        pieces = self._splitter.chunk(split_source)
        if not pieces:
            return []

        agent = get_chunker_agent()
        log_info(
            f"Legal async chunking: {name} ({len(text)} chars) -> "
            f"{len(pieces)} split(s), batch={CHUNKER_CONCURRENCY}"
        )

        identity = LegalChunkDecision(
            split_at=len(text),
            instrument_name=str(origin_meta.get("instrument_name") or name),
            doc_family=str(origin_meta.get("doc_family") or ""),
            running_summary="",
        )
        head = text[: min(len(text), max(self.chunk_size, 6000))]
        profile = await self._alabel(
            agent=agent,
            document=document,
            window=head,
            remaining_len=len(text),
            last=identity,
            is_tail=len(text) <= len(head),
            role="identity",
        )
        if profile is not None:
            identity = profile

        if len(pieces) == 1:
            labeled = [identity]
        else:
            labeled = await _label_in_batches(
                self,
                agent=agent,
                document=document,
                pieces=pieces,
                identity=identity,
            )

        chunks: List[Document] = []
        for index, (piece, decision) in enumerate(zip(pieces, labeled), start=1):
            chunk_text = (piece.content or "").strip()
            if not chunk_text:
                continue
            meta = self._chunk_metadata(
                document=document,
                decision=decision,
                chunk_number=index,
                chunk_text=chunk_text,
            )
            chunks.append(
                Document(
                    id=self._generate_chunk_id(document, index, chunk_text),
                    name=name,
                    meta_data=meta,
                    content=locator_prefix(meta, name) + chunk_text,
                )
            )
            log_info(
                f"  chunk {index}: {meta.get('instrument_name')} "
                f"clause={meta.get('clause_id') or '-'} family={meta.get('doc_family') or '-'}"
            )
        log_info(f"Legal async chunking done: {name} -> {len(chunks)} chunk(s)")
        return chunks

    async def _chunk_sequential(self, document: Document) -> List[Document]:
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

        log_info(f"Legal sequential chunking: {name} ({len(text)} chars), session {session_id}")

        while remaining:
            window = remaining[: self.chunk_size]
            is_tail = len(remaining) <= self.chunk_size
            decision = await self._alabel(
                agent=agent,
                document=document,
                window=window,
                remaining_len=len(remaining),
                last=last,
                is_tail=is_tail,
                role="split",
                session_id=session_id,
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
            chunks.append(
                Document(
                    id=self._generate_chunk_id(document, chunk_number, chunk_text),
                    name=name,
                    meta_data=meta,
                    content=locator_prefix(meta, name) + chunk_text,
                )
            )
            log_info(
                f"  chunk {chunk_number}: {meta.get('instrument_name')} "
                f"clause={meta.get('clause_id') or '-'} family={meta.get('doc_family') or '-'}"
            )
            chunk_number += 1

        log_info(f"Legal sequential chunking done: {name} -> {len(chunks)} chunk(s)")
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
        return _run_sync(
            self._alabel(
                agent=agent,
                document=document,
                window=window,
                remaining_len=remaining_len,
                last=last,
                is_tail=is_tail,
                role="split",
                session_id=session_id,
            )
        )

    async def _alabel(
        self,
        *,
        agent: Agent,
        document: Document,
        window: str,
        remaining_len: int,
        last: LegalChunkDecision,
        is_tail: bool,
        role: str,
        session_id: Optional[str] = None,
        chunk_index: int = 0,
        chunk_count: int = 0,
    ) -> Optional[LegalChunkDecision]:
        meta = document.meta_data or {}
        task = {
            "identity": (
                "ROLE: document identity. Read the opening text and fill instrument_name, "
                "aliases, doc_family, parties, block, document_status, running_summary. "
                "Set split_at to window_length."
            ),
            "label": (
                f"ROLE: label pre-split chunk {chunk_index}/{chunk_count}. "
                "Reuse PRIOR instrument_name/doc_family. Extract clause_id, article, "
                "heading, content_type, cross_references, defined_terms. "
                "Set split_at to window_length."
            ),
            "split": (
                "ROLE: choose split_at in this window and extract metadata. "
                "Keep instrument_name stable."
            ),
        }.get(role, "Extract metadata.")
        prompt = (
            f"{task}\n\n"
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
        run_session = session_id or f"legal-chunk:{role}:{uuid.uuid4().hex[:10]}"
        try:
            response = await agent.arun(prompt, session_id=run_session)
            return _parse_decision(getattr(response, "content", None))
        except Exception as exc:
            log_warning(f"Chunker agent failed ({role}), using fallback: {exc}")
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
            "block": decision.block.strip() or str(origin.get("block") or ""),
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
    heading: str = "",
    parties: str = "",
    block: str = "",
    content_type: str = "",
    document_status: str = "",
    defined_terms: str = "",
    cross_references: str = "",
) -> List[Document]:
    clause = normalize_clause(clause_id)
    family = normalize_family(doc_family)

    def _has(meta: Dict[str, Any], key: str, needle: str) -> bool:
        if not needle:
            return False
        return needle.lower() in str(meta.get(key) or "").lower()

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
        if _has(meta, "heading", heading):
            points += 3
        if _has(meta, "block", block):
            points += 3
        if _has(meta, "parties", parties):
            points += 2
        if _has(meta, "defined_terms", defined_terms):
            points += 3
        if _has(meta, "cross_references", cross_references):
            points += 2
        if content_type and str(meta.get("content_type") or "").lower() == content_type.lower():
            points += 3
        if document_status and str(meta.get("document_status") or "").lower() == document_status.lower():
            points += 2
        return (-points, 0)

    return sorted(documents, key=score)


# LegalChunkDecision fields stored on each chunk. Skip running_summary and
# instrument_aliases: they are long joined blobs, not exact Chroma keys.
# instrument_key is derived in _chunk_metadata from instrument_name.
_EXACT_FILTER_FIELDS = (
    "heading",
    "parties",
    "block",
    "content_type",
    "document_status",
    "defined_terms",
    "cross_references",
)


def _parse_optional_bool(value: Any) -> Optional[bool]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "on"}:
        return True
    if text in {"false", "0", "no", "off"}:
        return False
    return None


def _chroma_eq(part: Dict[str, Any]) -> Dict[str, Any]:
    """Wrap a simple {field: value} as {field: {$eq: value}}; leave $or/$and."""
    if any(key.startswith("$") for key in part):
        return part
    return {key: {"$eq": value} for key, value in part.items()}


def build_chroma_filters(
    *,
    doc_family: str = "",
    clause_id: str = "",
    article: str = "",
    instrument_name: str = "",
    heading: str = "",
    parties: str = "",
    block: str = "",
    content_type: str = "",
    document_status: str = "",
    defined_terms: str = "",
    cross_references: str = "",
    ocr_uncertainty: Any = "",
) -> Optional[Dict[str, Any]]:
    """Build a Chroma ``where`` clause from chunker-agent metadata.

    Only LegalChunkDecision fields (plus derived ``instrument_key``) are
    accepted. Chroma internals and origin fields such as ``_agno.*``,
    ``chroma:document``, ``filename`` and ``running_summary`` are ignored.

    Chroma allows exactly one top-level operator. Two metadata keys must be
    wrapped in ``$and``. A lone simple key is left as ``{field: value}`` so
    Agno can add ``$eq``.
    """
    family = normalize_family(doc_family)
    clause = normalize_clause(clause_id)
    # clause_id already implies the article; sending both over-filters when
    # stored clause_id is "21.5" and the model passed "21.5.5" + article 21.
    art = "" if clause else normalize_article(article)
    parts: List[Dict[str, Any]] = []
    if family:
        parts.append({"doc_family": family})
    if clause:
        parts.append({"clause_id": clause})
    if art:
        parts.append({"article": art})

    instrument = (instrument_name or "").strip()
    if instrument:
        inst_parts: List[Dict[str, Any]] = [{"instrument_name": instrument}]
        key = primary_instrument_key(instrument)
        if key:
            inst_parts.append({"instrument_key": key})
        if len(inst_parts) == 1:
            parts.append(inst_parts[0])
        else:
            parts.append({"$or": [_chroma_eq(item) for item in inst_parts]})

    extras = {
        "heading": (heading or "").strip(),
        "parties": (parties or "").strip(),
        "block": (block or "").strip(),
        "content_type": (content_type or "").strip(),
        "document_status": (document_status or "").strip(),
        "defined_terms": (defined_terms or "").strip(),
        "cross_references": (cross_references or "").strip(),
    }
    for field in _EXACT_FILTER_FIELDS:
        value = extras[field]
        if value:
            parts.append({field: value})

    ocr = _parse_optional_bool(ocr_uncertainty)
    if ocr is not None:
        parts.append({"ocr_uncertainty": ocr})

    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return {"$and": [_chroma_eq(part) for part in parts]}


def _stamp_then_chunk(
    documents: List[Document],
    origin: SourceOrigin,
    chunk: bool,
    reader: PDFReader | MarkdownReader | TextReader | ExcelReader,
) -> List[Document]:
    for doc in documents:
        apply_origin(doc, origin)
    if chunk and documents:
        return [
            piece
            for doc in documents
            for piece in reader.chunk_document(doc)
        ]
    return documents


async def _stamp_then_achunk(
    documents: List[Document],
    origin: SourceOrigin,
    chunk: bool,
    reader: PDFReader | MarkdownReader | TextReader | ExcelReader,
) -> List[Document]:
    for doc in documents:
        apply_origin(doc, origin)
    if chunk and documents:
        return [
            piece
            for group in await asyncio.gather(*[reader.achunk_document(doc) for doc in documents])
            for piece in group
        ]
    return documents


def _joined_text(documents: List[Document]) -> str:
    return "\n".join(str(document.content or "") for document in documents).strip()


def _pdf_path(pdf: Optional[Union[str, Path, IO[Any]]]) -> Optional[Path]:
    if isinstance(pdf, Path):
        return pdf
    if isinstance(pdf, str) and pdf.strip():
        return Path(pdf)
    return None


def _pdftotext(path: Path) -> str:
    """Poppler text extract. Empty when the tool is missing or the PDF has no text layer."""
    try:
        completed = subprocess.run(
            ["pdftotext", "-layout", "-enc", "UTF-8", str(path), "-"],
            capture_output=True,
            timeout=180,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout.decode("utf-8", errors="replace")


def _pypdf_layout_text(pdf: Optional[Union[str, Path, IO[Any]]], password: Optional[str]) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        return ""
    try:
        reader = PdfReader(pdf, password=password) if password else PdfReader(pdf)
        pages = [page.extract_text(extraction_mode="layout") or "" for page in reader.pages]
    except Exception:
        return ""
    return "\n\n".join(pages)


def recover_empty_pdf_text(
    pdf: Optional[Union[str, Path, IO[Any]]],
    name: Optional[str],
    password: Optional[str],
    documents: List[Document],
) -> List[Document]:
    """Replace an empty PDF extract with pdftotext, or fail the file.

    An empty extract used to continue into "Async Upserting 0 documents", so the
    file looked imported and nothing was searchable.
    """
    if _joined_text(documents):
        return documents
    label = name or (_pdf_path(pdf).name if _pdf_path(pdf) else "pdf")
    log_warning(f"PDF reader extracted no text from {label}. Trying pdftotext.")
    text = ""
    source = "pdftotext"
    path = _pdf_path(pdf)
    if path is not None and path.is_file():
        text = _pdftotext(path).strip()
    if not text:
        source = "PDF layout extract"
        text = _pypdf_layout_text(pdf, password).strip()
    if not text:
        raise RuntimeError(f"PDF reader extracted no text from {label}. The file was not indexed.")
    log_warning(f"Loaded {label} with {source} after the PDF reader extracted no text.")
    return [Document(name=label, id=str(uuid.uuid4()), content=text)]


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
            documents = recover_empty_pdf_text(pdf, name, password, documents)
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
            documents = await asyncio.to_thread(
                recover_empty_pdf_text, pdf, name, password, documents
            )
        finally:
            self.chunk = was_chunk
        return await _stamp_then_achunk(documents, origin, was_chunk, self)


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
        stream, encoding = open_decoded(file, name)
        was_chunk = self.chunk
        self.chunk = False
        try:
            documents = super().read(stream, name=name)
        finally:
            self.chunk = was_chunk
        tag_encoding(documents, encoding)
        return _read_with_origin(self, file, name, documents, was_chunk)

    async def async_read(self, file: Union[Path, IO[Any]], name: Optional[str] = None) -> List[Document]:
        stream, encoding = open_decoded(file, name)
        origin = origin_from_source(file, name)
        was_chunk = self.chunk
        self.chunk = False
        try:
            documents = await super().async_read(stream, name=name)
        finally:
            self.chunk = was_chunk
        tag_encoding(documents, encoding)
        return await _stamp_then_achunk(documents, origin, was_chunk, self)


class PathAwareTextReader(TextReader):
    def read(self, file: Union[Path, IO[Any]], name: Optional[str] = None) -> List[Document]:
        stream, encoding = open_decoded(file, name)
        was_chunk = self.chunk
        self.chunk = False
        try:
            documents = super().read(stream, name=name)
        finally:
            self.chunk = was_chunk
        tag_encoding(documents, encoding)
        return _read_with_origin(self, file, name, documents, was_chunk)

    async def async_read(self, file: Union[Path, IO[Any]], name: Optional[str] = None) -> List[Document]:
        stream, encoding = open_decoded(file, name)
        origin = origin_from_source(file, name)
        was_chunk = self.chunk
        self.chunk = False
        try:
            documents = await super().async_read(stream, name=name)
        finally:
            self.chunk = was_chunk
        tag_encoding(documents, encoding)
        return await _stamp_then_achunk(documents, origin, was_chunk, self)


class PathAwareExcelReader(ExcelReader):
    """Stamp sites/family folder metadata onto spreadsheet rows."""

    def read(self, file: Union[Path, IO[Any]], name: Optional[str] = None) -> List[Document]:
        origin = origin_from_source(file, name)
        was_chunk = self.chunk
        self.chunk = False
        try:
            documents = super().read(file, name=name)
        finally:
            self.chunk = was_chunk
        return _stamp_then_chunk(documents, origin, was_chunk, self)

    async def async_read(
        self,
        file: Union[Path, IO[Any]],
        name: Optional[str] = None,
    ) -> List[Document]:
        origin = origin_from_source(file, name)
        was_chunk = self.chunk
        self.chunk = False
        try:
            documents = await super().async_read(file, name=name)
        finally:
            self.chunk = was_chunk
        return await _stamp_then_achunk(documents, origin, was_chunk, self)
