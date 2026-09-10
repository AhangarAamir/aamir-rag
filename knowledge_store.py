"""One shared Knowledge base.

Ingest agent writes here. Query agent searches it as a tool at runtime
(KnowledgeTools), not by stuffing chunks into the prompt.
"""

from __future__ import annotations

from agno.db.sqlite import SqliteDb
from agno.knowledge.chunking.agentic import AgenticChunking
from agno.knowledge.chunking.document import DocumentChunking
from agno.knowledge.chunking.recursive import RecursiveChunking
from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.knowledge.knowledge import Knowledge
from agno.knowledge.reader.pdf_reader import PDFReader
from agno.models.openai import OpenAIChat
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


def build_pdf_reader() -> PDFReader:
    """Chunking used at ingest time. The query agent never re-chunks."""
    if CHUNKING_STRATEGY == "document":
        strategy = DocumentChunking()
    elif CHUNKING_STRATEGY == "recursive":
        strategy = RecursiveChunking(chunk_size=1800, overlap=200)
    else:
        strategy = AgenticChunking(
            model=OpenAIChat(id=OPENAI_MODEL),
            custom_prompt=CHUNKING_INSTRUCTIONS,
            max_chunk_size=4000,
        )

    return PDFReader(
        name="Legal PDF Reader",
        split_on_pages=False,
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
    # So ingest_path() uses agentic/clause-aware PDF splits instead of the default reader.
    if knowledge.readers is None:
        knowledge.readers = {}
    knowledge.readers["pdf"] = build_pdf_reader()
    _knowledge = knowledge
    return knowledge
