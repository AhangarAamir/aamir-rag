"""One shared Knowledge base.

Ingest writes here. Query searches it as a tool at runtime
(KnowledgeTools), not by stuffing chunks into the prompt.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

from agno.db.sqlite import SqliteDb
from agno.knowledge.chunking.agentic import AgenticChunking
from agno.knowledge.chunking.document import DocumentChunking
from agno.knowledge.chunking.recursive import RecursiveChunking
from agno.knowledge.chunking.strategy import ChunkingStrategy
from agno.knowledge.document.base import Document
from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.knowledge.knowledge import Knowledge
from agno.knowledge.reader.markdown_reader import MarkdownReader
from agno.knowledge.reader.pdf_reader import PDFReader
from agno.knowledge.reader.text_reader import TextReader
from agno.models.openai import OpenAIResponses
from agno.utils.log import log_info
from agno.vectordb.chroma import ChromaDb
from agno.vectordb.search import SearchType

from alias_catalog import register_document
from config import (
    CHUNKING_STRATEGY,
    COLLECTION_NAME,
    DATA_DIR,
    EMBEDDING_MODEL,
    OPENAI_MODEL,
    VECTOR_DB_DIR,
    require_api_key,
)
from folder_tags import folder_tags_from_path
from metadata_agent import (
    build_search_prefix,
    chunk_metadata,
    enrich_chunks,
)
from prompts import CHUNKING_INSTRUCTIONS

_knowledge: Knowledge | None = None


class EnrichingChunking(ChunkingStrategy):
    """Split, then stamp uniqueness metadata and a search prefix before embed."""

    def __init__(self, inner: ChunkingStrategy):
        self.inner = inner
        self._source_path: Optional[Path] = None

    def set_source_path(self, path: Path) -> None:
        self._source_path = Path(path)

    def chunk(self, document: Document) -> List[Document]:
        chunks = self.inner.chunk(document)
        name = (document.name or "").strip()
        log_info(f"Chunking document: {name} -> {len(chunks)} chunk(s)")
        if not chunks:
            return chunks

        folder_tags = {}
        if self._source_path is not None:
            folder_tags = folder_tags_from_path(self._source_path)
        elif name:
            folder_tags = {"filename": name}

        filename = folder_tags.get("filename") or name
        log_info(f"Enriching uniqueness metadata for {filename}")
        full_text = document.content or ""
        chunk_texts = [piece.content or "" for piece in chunks]
        document_card, cards = enrich_chunks(filename, folder_tags, full_text, chunk_texts)
        register_document(
            document_card.instrument_key,
            filename,
            document_card.document_title or filename,
            document_card.aliases,
        )

        for piece, card in zip(chunks, cards):
            meta = chunk_metadata(document_card, card, folder_tags)
            piece.meta_data = {**(piece.meta_data or {}), **meta}
            prefix = build_search_prefix(document_card, card, folder_tags)
            body = piece.content or ""
            if not body.startswith("[Document:"):
                piece.content = prefix + body
        return chunks


def build_chunking_strategy() -> EnrichingChunking:
    if CHUNKING_STRATEGY == "document":
        inner: ChunkingStrategy = DocumentChunking()
    elif CHUNKING_STRATEGY == "agentic":
        inner = AgenticChunking(
            model=OpenAIResponses(id=OPENAI_MODEL),
            custom_prompt=CHUNKING_INSTRUCTIONS,
            max_chunk_size=4000,
        )
    else:
        inner = RecursiveChunking(chunk_size=1800, overlap=200)
    return EnrichingChunking(inner)


class PathAwarePDFReader(PDFReader):
    def read(self, pdf, name: Optional[str] = None, password: Optional[str] = None):
        self._stamp_source(pdf)
        return super().read(pdf, name=name, password=password)

    async def async_read(self, pdf, name: Optional[str] = None, password: Optional[str] = None):
        self._stamp_source(pdf)
        return await super().async_read(pdf, name=name, password=password)

    def _stamp_source(self, pdf: Any) -> None:
        if isinstance(self.chunking_strategy, EnrichingChunking) and isinstance(pdf, (str, Path)):
            self.chunking_strategy.set_source_path(Path(pdf))


class PathAwareMarkdownReader(MarkdownReader):
    def read(self, path, name: Optional[str] = None):
        self._stamp_source(path)
        return super().read(path, name=name)

    async def async_read(self, path, name: Optional[str] = None):
        self._stamp_source(path)
        return await super().async_read(path, name=name)

    def _stamp_source(self, path: Any) -> None:
        if isinstance(self.chunking_strategy, EnrichingChunking) and isinstance(path, (str, Path)):
            self.chunking_strategy.set_source_path(Path(path))


class PathAwareTextReader(TextReader):
    def read(self, path, name: Optional[str] = None):
        self._stamp_source(path)
        return super().read(path, name=name)

    async def async_read(self, path, name: Optional[str] = None):
        self._stamp_source(path)
        return await super().async_read(path, name=name)

    def _stamp_source(self, path: Any) -> None:
        if isinstance(self.chunking_strategy, EnrichingChunking) and isinstance(path, (str, Path)):
            self.chunking_strategy.set_source_path(Path(path))


def _attach_readers(knowledge: Knowledge) -> None:
    strategy = build_chunking_strategy()
    if knowledge.readers is None:
        knowledge.readers = {}
    knowledge.readers["pdf"] = PathAwarePDFReader(
        name="Legal PDF Reader",
        split_on_pages=False,
        chunking_strategy=strategy,
    )
    knowledge.readers["markdown"] = PathAwareMarkdownReader(
        name="Legal Markdown Reader",
        chunking_strategy=strategy,
    )
    knowledge.readers["text"] = PathAwareTextReader(
        name="Legal Text Reader",
        chunking_strategy=strategy,
    )


def _reader_for_path(knowledge: Knowledge, path: Path) -> object | None:
    suffix = path.suffix.lower()
    mapping = {".pdf": "pdf", ".md": "markdown", ".markdown": "markdown", ".txt": "text"}
    key = mapping.get(suffix)
    if key and knowledge.readers:
        return knowledge.readers.get(key)
    return None


def _patch_insert_to_use_legal_readers(knowledge: Knowledge) -> None:
    """Agno insert() uses ReaderFactory unless reader= is passed. Force our enriching readers."""
    original_insert = knowledge.insert
    original_ainsert = knowledge.ainsert

    def insert_with_reader(*args, **kwargs):
        path = kwargs.get("path")
        if path and kwargs.get("reader") is None:
            reader = _reader_for_path(knowledge, Path(str(path)))
            if reader is not None:
                kwargs["reader"] = reader
        return original_insert(*args, **kwargs)

    async def ainsert_with_reader(*args, **kwargs):
        path = kwargs.get("path")
        if path and kwargs.get("reader") is None:
            reader = _reader_for_path(knowledge, Path(str(path)))
            if reader is not None:
                kwargs["reader"] = reader
        return await original_ainsert(*args, **kwargs)

    knowledge.insert = insert_with_reader  # type: ignore[method-assign]
    knowledge.ainsert = ainsert_with_reader  # type: ignore[method-assign]


def get_knowledge() -> Knowledge:
    global _knowledge
    if _knowledge is not None:
        return _knowledge

    require_api_key()
    knowledge = Knowledge(
        name="Legal contract knowledge",
        description="PSC, JOA, assignment, notification, and related oil & gas documents.",
        contents_db=SqliteDb(db_file=str(DATA_DIR / "contents.db")),
        vector_db=ChromaDb(
            collection=COLLECTION_NAME,
            path=str(VECTOR_DB_DIR),
            persistent_client=True,
            search_type=SearchType.hybrid,
            embedder=OpenAIEmbedder(id=EMBEDDING_MODEL),
        ),
    )
    _attach_readers(knowledge)
    _patch_insert_to_use_legal_readers(knowledge)
    _knowledge = knowledge
    return knowledge
