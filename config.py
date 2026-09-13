"""Shared paths and model settings for ingest + query agents."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env")

DATA_DIR = ROOT_DIR / "data"
INCOMING_DIR = DATA_DIR / "incoming"
VECTOR_DB_DIR = DATA_DIR / "vectordb"
COLLECTION_NAME = "legal_contracts"
ALIASES_DB_FILE = DATA_DIR / "aliases.db"

INCOMING_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
CHUNKING_STRATEGY = os.getenv("CHUNKING_STRATEGY", "recursive").strip().lower()
QUERY_SESSION_ID = os.getenv("QUERY_SESSION_ID", "legal-query-default")
NUM_HISTORY_RUNS = int(os.getenv("NUM_HISTORY_RUNS", "5"))
METADATA_BATCH_SIZE = int(os.getenv("METADATA_BATCH_SIZE", "4"))
DOCUMENT_CARD_CHARS = int(os.getenv("DOCUMENT_CARD_CHARS", "12000"))
SESSIONS_DB_FILE = DATA_DIR / "sessions.db"


def _quiet_openai_httpx_finalizers() -> None:
    """OpenAI 3.x wraps httpx2. Client GC calls __del__ -> is_closed -> _state.

    If the wrapper is already half-destroyed, _state is gone and Python prints
    'Exception ignored'. Searches still succeed. Swallow that destructor noise.
    """
    try:
        from openai._base_client import AsyncHttpxClientWrapper, SyncHttpxClientWrapper
    except ImportError:
        return

    def _safe_sync_del(self) -> None:
        try:
            if getattr(self, "_state", None) is None or self.is_closed:
                return
            self.close()
        except Exception:
            pass

    def _safe_async_del(self) -> None:
        try:
            if getattr(self, "_state", None) is None or self.is_closed:
                return
            import asyncio

            loop = asyncio.get_running_loop()
            loop.create_task(self.aclose())
        except Exception:
            pass

    SyncHttpxClientWrapper.__del__ = _safe_sync_del  # type: ignore[method-assign]
    AsyncHttpxClientWrapper.__del__ = _safe_async_del  # type: ignore[method-assign]


_quiet_openai_httpx_finalizers()


def require_api_key() -> None:
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Copy .env.example to .env and set the key."
        )
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY # 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
