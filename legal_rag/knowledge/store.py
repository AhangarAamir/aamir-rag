"""Approved-lessons knowledge for the learning loop.

The contract corpus is not here: it lives in data/index.db (legal_rag.index).
Reviewer-approved lessons stay in their own Chroma collection so they never
mix with contract passages.
"""

from __future__ import annotations

from agno.db.sqlite import SqliteDb
from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.chroma import ChromaDb
from agno.vectordb.search import SearchType

from legal_rag.config import DATA_DIR, EMBEDDING_MODEL, OPENAI_API_BASE, OPENAI_API_KEY, require_api_key

LEARNINGS_COLLECTION = "legal_learnings"
LEARNINGS_VECTOR_DB_DIR = DATA_DIR / "learnings_vectordb"

_learnings: Knowledge | None = None


def build_embedder() -> OpenAIEmbedder:
    return OpenAIEmbedder(
        id=EMBEDDING_MODEL,
        base_url=OPENAI_API_BASE,
        api_key=OPENAI_API_KEY or None,
        enable_batch=True,
        batch_size=64,
    )


def get_learnings_knowledge() -> Knowledge:
    """Approved lessons only. Kept out of the contract corpus."""
    global _learnings
    if _learnings is not None:
        return _learnings

    require_api_key()
    LEARNINGS_VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
    _learnings = Knowledge(
        name="Approved legal lessons",
        description="Reviewer-approved lessons for the query agent. Not the contract corpus.",
        contents_db=SqliteDb(db_file=str(DATA_DIR / "learnings_contents.db")),
        vector_db=ChromaDb(
            collection=LEARNINGS_COLLECTION,
            path=str(LEARNINGS_VECTOR_DB_DIR),
            persistent_client=True,
            search_type=SearchType.hybrid,
            embedder=build_embedder(),
        ),
    )
    return _learnings
