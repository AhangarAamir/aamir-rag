"""Paths and settings for the ingest pipeline, the index and the agents.

Models are OpenAI only: one chat model (Responses API) and one embedding
model, both read from .env.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")


def _path(name: str, default: Path) -> Path:
    raw = os.getenv(name, "").strip()
    path = Path(raw).expanduser() if raw else default
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


DATA_DIR = _path("DATA_DIR", ROOT_DIR / "data")
INCOMING_DIR = DATA_DIR / "incoming"
# Files the agent answers from (.pdf, .md, .txt). Folder paths under it become
# the Document Center links stored in documents.link.
CORPUS_DIR = _path("CORPUS_DIR", INCOMING_DIR)
INDEX_DB = DATA_DIR / "index.db"
SESSIONS_DB_FILE = DATA_DIR / "sessions.db"

for _dir in (DATA_DIR, INCOMING_DIR, CORPUS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

from legal_rag.knowledge.catalog import ensure_incoming_layout  # noqa: E402

ensure_incoming_layout(INCOMING_DIR)

# ---- OpenAI models (chat + embeddings)
# The base URL is fixed on purpose. The openai client would otherwise pick up
# OPENAI_BASE_URL from the environment (for example an old Ollama setting).
OPENAI_API_BASE = "https://api.openai.com/v1"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
# none | minimal | low | medium | high | xhigh. Empty = model default.
OPENAI_REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT", "").strip()
# Model for the one profile call per document at ingest. Empty = OPENAI_MODEL.
INGEST_MODEL = os.getenv("INGEST_MODEL", "").strip() or OPENAI_MODEL
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
EMBED_ENABLED = _bool("EMBED_ENABLED", True)
EMBED_BATCH_SIZE = max(1, int(os.getenv("EMBED_BATCH_SIZE", "64")))

# ---- Retrieval / agent tuning
SEARCH_TOP_K = int(os.getenv("SEARCH_TOP_K", "8"))
CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "2400"))
NUM_HISTORY_RUNS = int(os.getenv("NUM_HISTORY_RUNS", "4"))
TOOL_CALL_LIMIT = int(os.getenv("TOOL_CALL_LIMIT", "18"))
QUERY_SESSION_ID = os.getenv("QUERY_SESSION_ID", "legal-query-default")
# One LLM call per document at ingest to read title, parties, block, date and
# status from the first pages. False = filename / text heuristics only.
INGEST_LLM_PROFILE = _bool("INGEST_LLM_PROFILE", True)
# Files processed in parallel during ingest. Above 1, concurrent SQLite writes
# can fail with "database is locked".
INGEST_WORKERS = max(1, int(os.getenv("INGEST_WORKERS", "1")))

# ---- Document links (documents.link and References), same rules as the old
# prompt: files under sites/ link from SITES_BASE_URL, every other file from
# DOC_CENTER_BASE_URL plus its folder path. Set both empty to link to the
# local file served at /files/.
DOC_CENTER_BASE_URL = os.getenv(
    "DOC_CENTER_BASE_URL", "https://enp1.ril.com/sites/Regulatory/Document%20Center"
).strip()
SITES_BASE_URL = os.getenv("SITES_BASE_URL", "https://enp1.ril.com").strip()

# ---- Server
HOST = os.getenv("HOST") or os.getenv("AGENT_OS_HOST") or "127.0.0.1"
PORT = int(os.getenv("PORT") or os.getenv("AGENT_OS_PORT") or "8000")

_DEFAULT_CORS_ORIGINS = (
    "https://uracs.enp.ril.com",
    "https://uracs.ucg.ril.com",
)


def _parse_cors_origins(raw: str | None) -> list[str]:
    """Comma-separated origins. Bare hostnames get an https:// prefix."""
    values = _DEFAULT_CORS_ORIGINS if raw is None else raw.split(",")
    origins: list[str] = []
    seen: set[str] = set()
    for value in values:
        origin = value.strip().rstrip("/")
        if not origin:
            continue
        if "://" not in origin:
            origin = f"https://{origin}"
        if origin not in seen:
            seen.add(origin)
            origins.append(origin)
    return origins


CORS_ALLOWED_ORIGINS = _parse_cors_origins(os.getenv("CORS_ALLOWED_ORIGINS"))


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
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
