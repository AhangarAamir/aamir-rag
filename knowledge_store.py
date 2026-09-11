"""One shared Knowledge base.

Ingest agent writes here. Query agent searches it as a tool at runtime
(KnowledgeTools), not by stuffing chunks into the prompt.
"""

from __future__ import annotations

from typing import List

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

from config import (
    CHUNKING_STRATEGY,
    COLLECTION_NAME,
    DATA_DIR,
    EMBEDDING_MODEL,
    OPENAI_MODEL,
    VECTOR_DB_DIR,
    require_api_key,
)
from prompts import CHUNKING_INSTRUCTIONS

_knowledge: Knowledge | None = None


class NamedChunking(ChunkingStrategy):
    """Run an inner splitter, then stamp the source filename on every chunk.

    Markdown/text ingest never used the PDF reader, so the old PDF-only
    prefix never ran. This wrapper applies to every reader we register.
    """

    def __init__(self, inner: ChunkingStrategy):
        self.inner = inner

    def chunk(self, document: Document) -> List[Document]:
        chunks = self.inner.chunk(document)
        name = (document.name or "").strip()
        log_info(f"Chunking document: {name} -> {len(chunks)} chunk(s)")
        if not name:
            return chunks
        prefix = f"[Document: {name}]\n"
        for piece in chunks:
            if piece.content and not piece.content.startswith("[Document:"):
                piece.content = prefix + piece.content
        return chunks


def build_chunking_strategy() -> ChunkingStrategy:
    if CHUNKING_STRATEGY == "document":
        inner: ChunkingStrategy = DocumentChunking()
    elif CHUNKING_STRATEGY == "recursive":
        inner = RecursiveChunking(chunk_size=1800, overlap=200)
    else:
        inner = AgenticChunking(
            model=OpenAIResponses(id=OPENAI_MODEL),
            custom_prompt=CHUNKING_INSTRUCTIONS,
            max_chunk_size=4000,
        )
    return NamedChunking(inner)


def _attach_readers(knowledge: Knowledge) -> None:
    strategy = build_chunking_strategy()
    if knowledge.readers is None:
        knowledge.readers = {}
    knowledge.readers["pdf"] = PDFReader(
        name="Legal PDF Reader",
        split_on_pages=False,
        chunking_strategy=strategy,
    )
    knowledge.readers["markdown"] = MarkdownReader(
        name="Legal Markdown Reader",
        chunking_strategy=strategy,
    )
    knowledge.readers["text"] = TextReader(
        name="Legal Text Reader",
        chunking_strategy=strategy,
    )


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
    _knowledge = knowledge
    return knowledge
