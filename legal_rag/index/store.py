"""SQLite index: documents, pages, clause-level chunks, FTS5 (BM25) and vectors.

One file (data/index.db) holds everything, so the index can be copied,
inspected with any SQLite tool, and rebuilt without a vector server.

documents.link is the Document Center URL of the file (legal_rag.domain.document_link).
The References section under every answer uses it.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

from legal_rag.config import INDEX_DB

_SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    rel_path TEXT UNIQUE NOT NULL,
    filename TEXT NOT NULL,
    folder TEXT NOT NULL DEFAULT '',
    link TEXT NOT NULL DEFAULT '',
    family TEXT NOT NULL DEFAULT '',
    family_desc TEXT NOT NULL DEFAULT '',
    title TEXT NOT NULL DEFAULT '',
    short_name TEXT NOT NULL DEFAULT '',
    aliases TEXT NOT NULL DEFAULT '[]',
    alias_keys TEXT NOT NULL DEFAULT '',
    parties TEXT NOT NULL DEFAULT '[]',
    block TEXT NOT NULL DEFAULT '',
    doc_date TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT '',
    summary TEXT NOT NULL DEFAULT '',
    pages INTEGER NOT NULL DEFAULT 0,
    chars INTEGER NOT NULL DEFAULT 0,
    chunks INTEGER NOT NULL DEFAULT 0,
    extractor TEXT NOT NULL DEFAULT '',
    empty_pages TEXT NOT NULL DEFAULT '[]',
    sha1 TEXT NOT NULL DEFAULT '',
    size INTEGER NOT NULL DEFAULT 0,
    mtime_ns INTEGER NOT NULL DEFAULT 0,
    profile_source TEXT NOT NULL DEFAULT '',
    ingested_at TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS pages (
    doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_no INTEGER NOT NULL,
    text TEXT NOT NULL,
    PRIMARY KEY (doc_id, page_no)
);
CREATE TABLE IF NOT EXISTS articles (
    doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    article TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    page_start INTEGER NOT NULL,
    page_end INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS articles_doc ON articles(doc_id, article);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ord INTEGER NOT NULL,
    kind TEXT NOT NULL,
    article TEXT NOT NULL DEFAULT '',
    article_title TEXT NOT NULL DEFAULT '',
    clause_id TEXT NOT NULL DEFAULT '',
    heading TEXT NOT NULL DEFAULT '',
    defined_term TEXT NOT NULL DEFAULT '',
    page_start INTEGER NOT NULL,
    page_end INTEGER NOT NULL,
    part INTEGER NOT NULL DEFAULT 1,
    parts INTEGER NOT NULL DEFAULT 1,
    context TEXT NOT NULL,
    text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc_id, ord);
CREATE INDEX IF NOT EXISTS chunks_clause ON chunks(clause_id);
CREATE INDEX IF NOT EXISTS chunks_term ON chunks(defined_term COLLATE NOCASE);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, context, keys,
    tokenize = 'porter unicode61 remove_diacritics 2'
);
CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vec BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    run_id TEXT, session_id TEXT, user_id TEXT,
    rating INTEGER NOT NULL DEFAULT 0,
    comment TEXT NOT NULL DEFAULT '',
    query TEXT NOT NULL DEFAULT '',
    answer TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL DEFAULT ''
);
"""

_LOCK = threading.RLock()
_local = threading.local()
_db_path: Path = INDEX_DB


def configure(path: Path) -> None:
    """Point the store at another file (tests, evaluation)."""
    global _db_path
    _db_path = Path(path)
    _local.__dict__.clear()


def _connect() -> sqlite3.Connection:
    conn = getattr(_local, "conn", None)
    if conn is not None and getattr(_local, "path", None) == _db_path:
        return conn
    _db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_db_path), check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(_SCHEMA)
    _migrate(conn)
    _local.conn = conn
    _local.path = _db_path
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Add documents.link to an index created before the column existed and fill it."""
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(documents)")}
    if "link" in columns:
        return
    from legal_rag.domain import document_link

    with _LOCK:
        conn.execute("ALTER TABLE documents ADD COLUMN link TEXT NOT NULL DEFAULT ''")
        conn.executemany(
            "UPDATE documents SET link=? WHERE id=?",
            [(document_link(r["rel_path"]), r["id"]) for r in conn.execute("SELECT id, rel_path FROM documents")],
        )
        conn.commit()


def refresh_links() -> int:
    """Rebuild documents.link for every row (after changing the base URLs in .env)."""
    from legal_rag.domain import document_link

    with tx() as conn:
        rows = conn.execute("SELECT id, rel_path FROM documents").fetchall()
        conn.executemany("UPDATE documents SET link=? WHERE id=?", [(document_link(r["rel_path"]), r["id"]) for r in rows])
    return len(rows)


@contextmanager
def tx() -> Iterator[sqlite3.Connection]:
    conn = _connect()
    with _LOCK:
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise


def query(sql: str, params: tuple | list = ()) -> list[sqlite3.Row]:
    return _connect().execute(sql, params).fetchall()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------ meta

def get_meta(key: str, default: str = "") -> str:
    row = _connect().execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row["value"] if row else default


def set_meta(key: str, value: str, conn: Optional[sqlite3.Connection] = None) -> None:
    (conn or _connect()).execute(
        "INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )


def bump_version(conn: sqlite3.Connection) -> None:
    current = conn.execute("SELECT value FROM meta WHERE key='index_version'").fetchone()
    set_meta("index_version", str(int(current["value"]) + 1 if current else 1), conn)


def index_version() -> int:
    return int(get_meta("index_version", "0") or 0)


# ------------------------------------------------------------------ documents

def document_row(doc: sqlite3.Row | dict) -> dict[str, Any]:
    data = dict(doc)
    for key in ("aliases", "parties", "empty_pages"):
        try:
            data[key] = json.loads(data.get(key) or "[]")
        except (TypeError, json.JSONDecodeError):
            data[key] = []
    return data


def list_documents() -> list[dict[str, Any]]:
    rows = query("SELECT * FROM documents ORDER BY folder, filename")
    return [document_row(r) for r in rows]


def get_document(doc_id: int) -> Optional[dict[str, Any]]:
    rows = query("SELECT * FROM documents WHERE id=?", (doc_id,))
    return document_row(rows[0]) if rows else None


def find_document_by_path(rel_path: str) -> Optional[dict[str, Any]]:
    rows = query("SELECT * FROM documents WHERE rel_path=?", (rel_path,))
    return document_row(rows[0]) if rows else None


def delete_document(doc_id: int) -> None:
    with tx() as conn:
        ids = [r["id"] for r in conn.execute("SELECT id FROM chunks WHERE doc_id=?", (doc_id,))]
        conn.executemany("DELETE FROM chunks_fts WHERE rowid=?", [(i,) for i in ids])
        conn.execute("DELETE FROM documents WHERE id=?", (doc_id,))
        bump_version(conn)


def stats() -> dict[str, Any]:
    conn = _connect()
    one = lambda sql: conn.execute(sql).fetchone()[0]  # noqa: E731
    return {
        "documents": one("SELECT COUNT(*) FROM documents"),
        "pages": one("SELECT COALESCE(SUM(pages),0) FROM documents"),
        "chunks": one("SELECT COUNT(*) FROM chunks"),
        "embedded": one("SELECT COUNT(*) FROM embeddings"),
        "embedding_model": get_meta("embedding_model"),
        "empty_pages": one(
            "SELECT COALESCE(SUM(json_array_length(empty_pages)),0) FROM documents"
        ),
    }


def add_feedback(**fields: Any) -> int:
    with tx() as conn:
        cur = conn.execute(
            "INSERT INTO feedback(created_at, run_id, session_id, user_id, rating, comment, query, answer, model)"
            " VALUES(?,?,?,?,?,?,?,?,?)",
            (
                now_iso(),
                fields.get("run_id"),
                fields.get("session_id"),
                fields.get("user_id"),
                int(fields.get("rating") or 0),
                fields.get("comment") or "",
                fields.get("query") or "",
                fields.get("answer") or "",
                fields.get("model") or "",
            ),
        )
        return int(cur.lastrowid)
