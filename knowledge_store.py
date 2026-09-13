"""One shared Knowledge base.

Ingest agent writes here. Query agent searches it as a tool at runtime
(KnowledgeTools), not by stuffing chunks into the prompt.

Agno's path ingest uses ReaderFactory, not Knowledge.readers, so custom
readers are registered in both places.
"""

from __future__ import annotations

from agno.db.sqlite import SqliteDb
from agno.knowledge.chunking.document import DocumentChunking
from agno.knowledge.chunking.recursive import RecursiveChunking
from agno.knowledge.chunking.strategy import ChunkingStrategy
from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.knowledge.knowledge import Knowledge
from agno.knowledge.reader.reader_factory import ReaderFactory
from agno.vectordb.chroma import ChromaDb
from agno.vectordb.search import SearchType

from config import (
    CHUNKING_MAX_SIZE,
    CHUNKING_STRATEGY,
    COLLECTION_NAME,
    DATA_DIR,
    EMBEDDING_MODEL,
    VECTOR_DB_DIR,
    require_api_key,
)
from legal_chunking import (
    LegalAgenticChunking,
    LocatorChunking,
    PathAwareMarkdownReader,
    PathAwarePDFReader,
    PathAwareTextReader,
)

_knowledge: Knowledge | None = None


def build_chunking_strategy() -> ChunkingStrategy:
    if CHUNKING_STRATEGY == "document":
        return LocatorChunking(DocumentChunking())
    if CHUNKING_STRATEGY == "recursive":
        return LocatorChunking(RecursiveChunking(chunk_size=1800, overlap=200))
    return LegalAgenticChunking(max_chunk_size=CHUNKING_MAX_SIZE)


def _attach_readers(knowledge: Knowledge) -> None:
    strategy = build_chunking_strategy()
    pdf_reader = PathAwarePDFReader(
        name="Legal PDF Reader",
        split_on_pages=False,
        chunking_strategy=strategy,
    )
    markdown_reader = PathAwareMarkdownReader(
        name="Legal Markdown Reader",
        chunking_strategy=strategy,
    )
    text_reader = PathAwareTextReader(
        name="Legal Text Reader",
        chunking_strategy=strategy,
    )
    if knowledge.readers is None:
        knowledge.readers = {}
    knowledge.readers["pdf"] = pdf_reader
    knowledge.readers["markdown"] = markdown_reader
    knowledge.readers["text"] = text_reader
    # Path ingest uses ReaderFactory.get_reader_for_extension, which caches
    # a default PDFReader. Replace that cache so ingest_path uses this strategy.
    ReaderFactory._reader_cache["pdf"] = pdf_reader
    ReaderFactory._reader_cache["markdown"] = markdown_reader
    ReaderFactory._reader_cache["text"] = text_reader


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
