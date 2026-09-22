"""SQLite store for Legal RAG observability runs, tool spans, and comments."""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from config import DATA_DIR

OBS_DB_FILE = DATA_DIR / "observability.db"

_LOCK = threading.Lock()
_CONN: sqlite3.Connection | None = None

_SCHEMA = """
CREATE TABLE IF NOT EXISTS obs_runs (
    run_id TEXT PRIMARY KEY,
    app TEXT NOT NULL DEFAULT 'legal-rag',
    agent_id TEXT,
    agent_name TEXT,
    route TEXT,
    query_class TEXT,
    status TEXT NOT NULL,
    model TEXT,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    duration_ms REAL,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cached_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0,
    session_id TEXT,
    user_id TEXT,
    input_text TEXT,
    output_text TEXT
);

CREATE INDEX IF NOT EXISTS idx_obs_runs_started ON obs_runs(started_at);
CREATE INDEX IF NOT EXISTS idx_obs_runs_app ON obs_runs(app, started_at);
CREATE INDEX IF NOT EXISTS idx_obs_runs_route ON obs_runs(route);
CREATE INDEX IF NOT EXISTS idx_obs_runs_class ON obs_runs(query_class);

CREATE TABLE IF NOT EXISTS obs_spans (
    span_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    name TEXT,
    kind TEXT,
    status TEXT,
    started_at TEXT,
    duration_ms REAL,
    detail TEXT,
    FOREIGN KEY (run_id) REFERENCES obs_runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_obs_spans_run ON obs_spans(run_id);

CREATE TABLE IF NOT EXISTS obs_comments (
    comment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    created_at TEXT NOT NULL,
    comment TEXT NOT NULL,
    query TEXT,
    history TEXT
);
"""


def _connect() -> sqlite3.Connection:
    global _CONN
    if _CONN is None:
        OBS_DB_FILE.parent.mkdir(parents=True, exist_ok=True)
        _CONN = sqlite3.connect(str(OBS_DB_FILE), check_same_thread=False)
        _CONN.row_factory = sqlite3.Row
        _CONN.execute("PRAGMA journal_mode=WAL")
        _CONN.execute("PRAGMA foreign_keys=ON")
        _CONN.executescript(_SCHEMA)
        _CONN.commit()
    return _CONN


def configure_db(path: str | Path) -> None:
    """Point the store at another SQLite file (tests)."""
    global OBS_DB_FILE, _CONN
    with _LOCK:
        if _CONN is not None:
            _CONN.close()
            _CONN = None
        OBS_DB_FILE = Path(path)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def upsert_run(row: dict[str, Any]) -> None:
    cols = (
        "run_id",
        "app",
        "agent_id",
        "agent_name",
        "route",
        "query_class",
        "status",
        "model",
        "started_at",
        "ended_at",
        "duration_ms",
        "input_tokens",
        "output_tokens",
        "cached_tokens",
        "total_tokens",
        "cost_usd",
        "session_id",
        "user_id",
        "input_text",
        "output_text",
    )
    values = [row.get(col) for col in cols]
    placeholders = ",".join("?" * len(cols))
    assignments = ",".join(f"{col}=excluded.{col}" for col in cols if col != "run_id")
    sql = (
        f"INSERT INTO obs_runs ({','.join(cols)}) VALUES ({placeholders}) "
        f"ON CONFLICT(run_id) DO UPDATE SET {assignments}"
    )
    with _LOCK:
        conn = _connect()
        conn.execute(sql, values)
        conn.commit()


def replace_spans(run_id: str, spans: Iterable[dict[str, Any]]) -> None:
    rows = list(spans)
    with _LOCK:
        conn = _connect()
        conn.execute("DELETE FROM obs_spans WHERE run_id = ?", (run_id,))
        conn.executemany(
            """
            INSERT INTO obs_spans (
                span_id, run_id, name, kind, status, started_at, duration_ms, detail
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    span["span_id"],
                    run_id,
                    span.get("name"),
                    span.get("kind"),
                    span.get("status"),
                    span.get("started_at"),
                    span.get("duration_ms"),
                    span.get("detail"),
                )
                for span in rows
            ],
        )
        conn.commit()


def add_comment(comment: str, run_id: str | None = None, query: str = "", history: str = "") -> dict[str, Any]:
    created = iso(utc_now())
    with _LOCK:
        conn = _connect()
        cur = conn.execute(
            """
            INSERT INTO obs_comments (run_id, created_at, comment, query, history)
            VALUES (?, ?, ?, ?, ?)
            """,
            (run_id, created, comment, query, history),
        )
        conn.commit()
        comment_id = cur.lastrowid
    return {
        "comment_id": comment_id,
        "run_id": run_id,
        "created_at": created,
        "comment": comment,
        "query": query,
        "history": history,
    }


def list_comments(since: datetime, app: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
    sql = """
        SELECT c.comment_id, c.run_id, c.created_at, c.comment, c.query, c.history
        FROM obs_comments c
        LEFT JOIN obs_runs r ON r.run_id = c.run_id
        WHERE c.created_at >= ?
    """
    params: list[Any] = [iso(since)]
    if app:
        sql += " AND (r.app = ? OR c.run_id IS NULL)"
        params.append(app)
    sql += " ORDER BY c.created_at DESC LIMIT ?"
    params.append(limit)
    with _LOCK:
        conn = _connect()
        rows = conn.execute(sql, params).fetchall()
    return [dict(row) for row in rows]


def list_apps() -> list[str]:
    with _LOCK:
        conn = _connect()
        rows = conn.execute(
            "SELECT DISTINCT app FROM obs_runs ORDER BY app"
        ).fetchall()
    apps = [row["app"] for row in rows if row["app"]]
    return apps or ["legal-rag"]


def get_run(run_id: str) -> dict[str, Any] | None:
    with _LOCK:
        conn = _connect()
        row = conn.execute("SELECT * FROM obs_runs WHERE run_id = ?", (run_id,)).fetchone()
    return dict(row) if row else None


def list_run_ids(status: str | None = None) -> list[str]:
    sql = "SELECT run_id FROM obs_runs"
    params: list[Any] = []
    if status:
        sql += " WHERE status = ?"
        params.append(status)
    with _LOCK:
        conn = _connect()
        rows = conn.execute(sql, params).fetchall()
    return [row["run_id"] for row in rows]


def list_runs(
    since: datetime,
    app: str | None = None,
    until: datetime | None = None,
) -> list[dict[str, Any]]:
    sql = "SELECT * FROM obs_runs WHERE started_at >= ?"
    params: list[Any] = [iso(since)]
    if until:
        sql += " AND started_at < ?"
        params.append(iso(until))
    if app:
        sql += " AND app = ?"
        params.append(app)
    sql += " ORDER BY started_at ASC"
    with _LOCK:
        conn = _connect()
        rows = conn.execute(sql, params).fetchall()
    return [dict(row) for row in rows]


def window_start(days: int) -> datetime:
    return utc_now() - timedelta(days=max(1, days))
