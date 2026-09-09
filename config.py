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

INCOMING_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
CHUNKING_STRATEGY = os.getenv("CHUNKING_STRATEGY", "agentic").strip().lower()
QUERY_SESSION_ID = os.getenv("QUERY_SESSION_ID", "legal-query-default")
NUM_HISTORY_RUNS = int(os.getenv("NUM_HISTORY_RUNS", "5"))
SESSIONS_DB_FILE = DATA_DIR / "sessions.db"


def require_api_key() -> None:
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Copy .env.example to .env and set the key."
        )
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY # 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
