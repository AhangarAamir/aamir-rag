"""SQLite queue for incidents, candidate lessons, and eval cases.

Approved lesson text is copied into the learned-knowledge vector store
only after a reviewer accepts it. This database is the review queue.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from legal_rag.config import DATA_DIR

LEARNING_DB_FILE = DATA_DIR / "learning.db"

_LOCK = threading.Lock()
_CONN: sqlite3.Connection | None = None

_SCHEMA = """
CREATE TABLE IF NOT EXISTS incidents (
    incident_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    type TEXT NOT NULL,
    area TEXT NOT NULL DEFAULT '',
    route TEXT NOT NULL DEFAULT '',
    query TEXT NOT NULL DEFAULT '',
    feedback TEXT NOT NULL DEFAULT '',
    run_id TEXT,
    session_id TEXT,
    user_id TEXT,
    status TEXT NOT NULL DEFAULT 'open',
    source_key TEXT NOT NULL UNIQUE
);

CREATE INDEX IF NOT EXISTS idx_incidents_created ON incidents(created_at);
CREATE INDEX IF NOT EXISTS idx_incidents_status ON incidents(status, created_at);

CREATE TABLE IF NOT EXISTS lessons (
    lesson_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'candidate',
    title TEXT NOT NULL,
    lesson TEXT NOT NULL,
    severity TEXT NOT NULL DEFAULT 'medium',
    tags TEXT NOT NULL DEFAULT '[]',
    confidence REAL NOT NULL DEFAULT 0.5,
    agent_id TEXT NOT NULL DEFAULT 'query-agent',
    files TEXT NOT NULL DEFAULT '[]',
    incident_id TEXT,
    run_id TEXT,
    query TEXT NOT NULL DEFAULT '',
    answer TEXT NOT NULL DEFAULT '',
    feedback TEXT NOT NULL DEFAULT '',
    context TEXT NOT NULL DEFAULT '',
    source_key TEXT NOT NULL UNIQUE
);

CREATE INDEX IF NOT EXISTS idx_lessons_status ON lessons(status, created_at);

CREATE TABLE IF NOT EXISTS eval_cases (
    eval_id TEXT PRIMARY KEY,
    lesson_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    query TEXT NOT NULL DEFAULT '',
    feedback TEXT NOT NULL DEFAULT '',
    lesson TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_eval_lesson ON eval_cases(lesson_id);
"""


def _connect() -> sqlite3.Connection:
    global _CONN
    if _CONN is None:
        LEARNING_DB_FILE.parent.mkdir(parents=True, exist_ok=True)
        _CONN = sqlite3.connect(str(LEARNING_DB_FILE), check_same_thread=False)
        _CONN.row_factory = sqlite3.Row
        _CONN.execute("PRAGMA journal_mode=WAL")
        _CONN.executescript(_SCHEMA)
        _CONN.commit()
    return _CONN


def configure_db(path: str | Path) -> None:
    """Point the queue at another SQLite file (tests)."""
    global LEARNING_DB_FILE, _CONN
    with _LOCK:
        if _CONN is not None:
            _CONN.close()
            _CONN = None
        LEARNING_DB_FILE = Path(path)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _loads(raw: str | None) -> list[Any]:
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return value if isinstance(value, list) else []


def _lesson_row(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)
    item["tags"] = _loads(item.get("tags"))
    item["files"] = _loads(item.get("files"))
    return item


def get_incident_by_source(source_key: str) -> dict[str, Any] | None:
    with _LOCK:
        conn = _connect()
        row = conn.execute(
            "SELECT * FROM incidents WHERE source_key = ?",
            (source_key,),
        ).fetchone()
    return dict(row) if row else None


def get_lesson_by_source(source_key: str) -> dict[str, Any] | None:
    with _LOCK:
        conn = _connect()
        row = conn.execute(
            "SELECT * FROM lessons WHERE source_key = ?",
            (source_key,),
        ).fetchone()
    return _lesson_row(row) if row else None


def get_lesson(lesson_id: str) -> dict[str, Any] | None:
    with _LOCK:
        conn = _connect()
        row = conn.execute(
            "SELECT * FROM lessons WHERE lesson_id = ?",
            (lesson_id,),
        ).fetchone()
    return _lesson_row(row) if row else None


def insert_incident(
    *,
    type: str,
    area: str,
    route: str,
    query: str,
    feedback: str,
    source_key: str,
    run_id: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    existing = get_incident_by_source(source_key)
    if existing:
        return existing
    incident_id = uuid.uuid4().hex
    created = created_at or iso(utc_now())
    with _LOCK:
        conn = _connect()
        conn.execute(
            """
            INSERT INTO incidents (
                incident_id, created_at, type, area, route, query, feedback,
                run_id, session_id, user_id, status, source_key
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?)
            """,
            (
                incident_id,
                created,
                type,
                area,
                route,
                query,
                feedback,
                run_id,
                session_id,
                user_id,
                source_key,
            ),
        )
        conn.commit()
    return get_incident_by_source(source_key) or {}


def upsert_candidate(
    *,
    source_key: str,
    incident_id: str | None,
    title: str,
    lesson: str,
    severity: str,
    tags: list[str],
    confidence: float,
    agent_id: str,
    files: list[str],
    run_id: str | None,
    query: str,
    answer: str,
    feedback: str,
    context: str,
) -> dict[str, Any]:
    existing = get_lesson_by_source(source_key)
    now = iso(utc_now())
    tags_json = json.dumps(tags)
    files_json = json.dumps(files)
    with _LOCK:
        conn = _connect()
        if existing and existing["status"] == "candidate":
            conn.execute(
                """
                UPDATE lessons SET
                    title = ?, lesson = ?, severity = ?, tags = ?, confidence = ?,
                    agent_id = ?, files = ?, incident_id = ?, run_id = ?,
                    query = ?, answer = ?, feedback = ?, context = ?, updated_at = ?
                WHERE source_key = ? AND status = 'candidate'
                """,
                (
                    title,
                    lesson,
                    severity,
                    tags_json,
                    confidence,
                    agent_id,
                    files_json,
                    incident_id,
                    run_id,
                    query,
                    answer,
                    feedback,
                    context,
                    now,
                    source_key,
                ),
            )
        elif existing:
            return existing
        else:
            conn.execute(
                """
                INSERT INTO lessons (
                    lesson_id, created_at, updated_at, status, title, lesson,
                    severity, tags, confidence, agent_id, files, incident_id,
                    run_id, query, answer, feedback, context, source_key
                ) VALUES (?, ?, ?, 'candidate', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uuid.uuid4().hex,
                    now,
                    now,
                    title,
                    lesson,
                    severity,
                    tags_json,
                    confidence,
                    agent_id,
                    files_json,
                    incident_id,
                    run_id,
                    query,
                    answer,
                    feedback,
                    context,
                    source_key,
                ),
            )
        conn.commit()
    return get_lesson_by_source(source_key) or {}


def list_lessons(status: str | None = None, since: datetime | None = None) -> list[dict[str, Any]]:
    sql = "SELECT * FROM lessons"
    clauses: list[str] = []
    params: list[Any] = []
    if status:
        clauses.append("status = ?")
        params.append(status)
    if since is not None:
        clauses.append("created_at >= ?")
        params.append(iso(since))
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY created_at DESC"
    with _LOCK:
        conn = _connect()
        rows = conn.execute(sql, params).fetchall()
    return [_lesson_row(row) for row in rows]


def list_incidents(since: datetime | None = None, status: str | None = None) -> list[dict[str, Any]]:
    sql = "SELECT * FROM incidents"
    clauses: list[str] = []
    params: list[Any] = []
    if since is not None:
        clauses.append("created_at >= ?")
        params.append(iso(since))
    if status:
        clauses.append("status = ?")
        params.append(status)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY created_at DESC"
    with _LOCK:
        conn = _connect()
        rows = conn.execute(sql, params).fetchall()
    return [dict(row) for row in rows]


def set_lesson_status(lesson_id: str, status: str) -> dict[str, Any] | None:
    now = iso(utc_now())
    with _LOCK:
        conn = _connect()
        conn.execute(
            "UPDATE lessons SET status = ?, updated_at = ? WHERE lesson_id = ?",
            (status, now, lesson_id),
        )
        row = conn.execute(
            "SELECT incident_id FROM lessons WHERE lesson_id = ?",
            (lesson_id,),
        ).fetchone()
        if row and row["incident_id"]:
            conn.execute(
                "UPDATE incidents SET status = 'closed' WHERE incident_id = ?",
                (row["incident_id"],),
            )
        conn.commit()
    return get_lesson(lesson_id)


def add_eval_case(lesson: dict[str, Any]) -> dict[str, Any]:
    eval_id = uuid.uuid4().hex
    created = iso(utc_now())
    with _LOCK:
        conn = _connect()
        conn.execute(
            """
            INSERT INTO eval_cases (eval_id, lesson_id, created_at, query, feedback, lesson)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                eval_id,
                lesson["lesson_id"],
                created,
                lesson.get("query") or "",
                lesson.get("feedback") or "",
                lesson.get("lesson") or "",
            ),
        )
        conn.commit()
    return {
        "eval_id": eval_id,
        "lesson_id": lesson["lesson_id"],
        "created_at": created,
        "query": lesson.get("query") or "",
        "feedback": lesson.get("feedback") or "",
        "lesson": lesson.get("lesson") or "",
    }


def list_eval_cases() -> list[dict[str, Any]]:
    with _LOCK:
        conn = _connect()
        rows = conn.execute(
            "SELECT * FROM eval_cases ORDER BY created_at ASC"
        ).fetchall()
    return [dict(row) for row in rows]


def count_eval_cases() -> int:
    with _LOCK:
        conn = _connect()
        row = conn.execute("SELECT COUNT(*) AS n FROM eval_cases").fetchone()
    return int(row["n"]) if row else 0


def cited_code_files(status: str = "approved") -> list[str]:
    seen: list[str] = []
    known: set[str] = set()
    for lesson in list_lessons(status=status):
        for path in lesson.get("files") or []:
            if path not in known:
                known.add(path)
                seen.append(path)
    return seen
