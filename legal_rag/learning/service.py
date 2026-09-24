"""Feedback, log analysis, and the approve/reject gate."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from legal_rag.config import ROOT_DIR
from legal_rag.learning.extract import extract_lesson, source_files
from legal_rag.learning.runtime import publish_lesson, search_lessons
from legal_rag.learning.store import (
    add_eval_case,
    get_lesson,
    get_lesson_by_source,
    insert_incident,
    list_incidents,
    list_lessons,
    set_lesson_status,
    upsert_candidate,
)
from legal_rag.observability.store import list_comments, list_runs, list_spans, window_start

SLOW_RUN_MS = 60_000.0
ANALYZE_LIMIT = 40


def _draft_fields(draft: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": draft["title"],
        "lesson": draft["lesson"],
        "severity": draft["severity"],
        "tags": draft["tags"],
        "confidence": draft["confidence"],
        "files": draft["files"],
        "context": draft["context"],
    }


def submit_feedback(
    *,
    feedback: str,
    query: str = "",
    answer: str = "",
    run_id: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
) -> dict[str, Any]:
    source_key = f"feedback:{run_id}" if run_id else f"feedback:{uuid.uuid4().hex}"
    draft = extract_lesson(
        query=query,
        answer=answer,
        feedback=feedback,
        incident_type="user_feedback",
    )
    incident = insert_incident(
        type="user_feedback",
        area=draft["area"],
        route="query-agent",
        query=query,
        feedback=feedback,
        source_key=source_key,
        run_id=run_id,
        session_id=session_id,
        user_id=user_id,
    )
    lesson = upsert_candidate(
        source_key=source_key,
        incident_id=incident.get("incident_id"),
        agent_id="query-agent",
        run_id=run_id,
        query=query,
        answer=answer,
        feedback=feedback,
        **_draft_fields(draft),
    )
    return {"incident": incident, "lesson": lesson}


def approve_lesson(lesson_id: str) -> dict[str, Any]:
    lesson = get_lesson(lesson_id)
    if lesson is None:
        raise KeyError(lesson_id)
    if lesson["status"] == "approved":
        return lesson
    if lesson["status"] != "candidate":
        raise ValueError(f"lesson is {lesson['status']}")
    if not publish_lesson(lesson):
        raise RuntimeError("approved lesson was not saved to learned knowledge")
    updated = set_lesson_status(lesson_id, "approved")
    add_eval_case(updated or lesson)
    return updated or lesson


def reject_lesson(lesson_id: str) -> dict[str, Any]:
    lesson = get_lesson(lesson_id)
    if lesson is None:
        raise KeyError(lesson_id)
    if lesson["status"] == "rejected":
        return lesson
    if lesson["status"] != "candidate":
        raise ValueError(f"lesson is {lesson['status']}")
    updated = set_lesson_status(lesson_id, "rejected")
    return updated or lesson


def _incident_kind(run: dict[str, Any], spans: list[dict[str, Any]]) -> tuple[str, str] | None:
    status = str(run.get("status") or "")
    if status == "fail":
        return "raw_error", "Run failed"
    for span in spans:
        span_status = str(span.get("status") or "").lower()
        if "error" in span_status or span_status == "fail":
            detail = span.get("detail") or span.get("name") or "tool error"
            return "tool_error", str(detail)
    duration = run.get("duration_ms")
    if duration is not None and float(duration) >= SLOW_RUN_MS:
        return "slow_query", f"Run took {float(duration):.0f} ms"
    return None


def _record_case(
    *,
    source_key: str,
    incident_type: str,
    area: str,
    route: str,
    query: str,
    answer: str,
    feedback: str,
    run_id: str | None,
    session_id: str | None = None,
    user_id: str | None = None,
    created_at: str | None = None,
) -> bool:
    if get_lesson_by_source(source_key):
        return False
    draft = extract_lesson(
        query=query,
        answer=answer,
        feedback=feedback,
        incident_type=incident_type,
    )
    incident = insert_incident(
        type=incident_type,
        area=area or draft["area"],
        route=route or "query-agent",
        query=query,
        feedback=feedback,
        source_key=source_key,
        run_id=run_id,
        session_id=session_id,
        user_id=user_id,
        created_at=created_at,
    )
    upsert_candidate(
        source_key=source_key,
        incident_id=incident.get("incident_id"),
        agent_id=route or "query-agent",
        run_id=run_id,
        query=query,
        answer=answer,
        feedback=feedback,
        **_draft_fields(draft),
    )
    return True


def analyze_logs(since: datetime) -> dict[str, int]:
    runs = list_runs(since)
    spans = list_spans([run["run_id"] for run in runs if run.get("run_id")])
    by_run: dict[str, list[dict[str, Any]]] = {}
    for span in spans:
        by_run.setdefault(span["run_id"], []).append(span)

    created = 0
    for run in runs:
        if created >= ANALYZE_LIMIT:
            break
        kind = _incident_kind(run, by_run.get(run["run_id"], []))
        if kind is None:
            continue
        incident_type, detail = kind
        source_key = f"log:{run['run_id']}:{incident_type}"
        if _record_case(
            source_key=source_key,
            incident_type=incident_type,
            area=str(run.get("query_class") or "contract"),
            route=str(run.get("route") or run.get("agent_id") or "query-agent"),
            query=str(run.get("input_text") or ""),
            answer=str(run.get("output_text") or ""),
            feedback=detail,
            run_id=run.get("run_id"),
            session_id=run.get("session_id"),
            user_id=run.get("user_id"),
            created_at=run.get("started_at"),
        ):
            created += 1

    for comment in list_comments(since):
        if created >= ANALYZE_LIMIT:
            break
        source_key = f"comment:{comment['comment_id']}"
        if _record_case(
            source_key=source_key,
            incident_type="user_feedback",
            area="contract",
            route="query-agent",
            query=str(comment.get("query") or ""),
            answer=str(comment.get("history") or ""),
            feedback=str(comment.get("comment") or ""),
            run_id=comment.get("run_id"),
            created_at=comment.get("created_at"),
        ):
            created += 1

    return {"created": created}


def preview_lessons(query: str, limit: int = 5) -> list[dict[str, Any]]:
    approved = {row["title"]: row for row in list_lessons(status="approved")}
    hits = search_lessons(query, limit=limit)
    found: list[dict[str, Any]] = []
    for hit in hits:
        title = getattr(hit, "title", None) or ""
        row = approved.get(title)
        if row is None:
            continue
        found.append(
            {
                "lesson_id": row["lesson_id"],
                "title": row["title"],
                "lesson": getattr(hit, "learning", None) or row["lesson"],
                "context": getattr(hit, "context", None) or row.get("context") or "",
                "tags": row.get("tags") or [],
                "confidence": row.get("confidence"),
                "files": row.get("files") or [],
                "severity": row.get("severity"),
            }
        )
    return found


def _feedback_lesson(lesson: dict[str, Any]) -> bool:
    tags = lesson.get("tags") or []
    source = str(lesson.get("source_key") or "")
    return "user_feedback" in tags or source.startswith(("feedback:", "comment:"))


def summary(since: datetime) -> dict[str, Any]:
    incidents = list_incidents(since=since)
    feedbacks = [row for row in incidents if row.get("type") == "user_feedback"]
    approved = [
        row for row in list_lessons(status="approved", since=since) if _feedback_lesson(row)
    ]
    return {
        "total_feedbacks": len(feedbacks),
        "approved_feedback": len(approved),
        "incidents": len(incidents),
    }


def search_code(query: str, limit: int = 20) -> list[dict[str, Any]]:
    needle = query.strip().lower()
    if not needle:
        return []
    matches: list[dict[str, Any]] = []
    for rel in source_files():
        if not rel.endswith((".py", ".md")):
            continue
        file_path = ROOT_DIR / rel
        try:
            lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        hits = []
        for number, line in enumerate(lines, start=1):
            if needle in line.lower():
                hits.append({"line": number, "text": line.strip()[:240]})
                if len(hits) >= 3:
                    break
        if hits:
            matches.append({"path": rel, "matches": hits})
        if len(matches) >= limit:
            break
    return matches
