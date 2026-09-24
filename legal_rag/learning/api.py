"""Review API for chat feedback, log analysis, and approved lessons."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from legal_rag.learning.service import (
    analyze_logs,
    approve_lesson,
    preview_lessons,
    reject_lesson,
    search_code,
    submit_feedback,
    summary,
)
from legal_rag.learning.store import list_eval_cases, list_incidents, list_lessons
from legal_rag.observability.store import window_start

router = APIRouter(prefix="/learning", tags=["learning"])


class FeedbackIn(BaseModel):
    feedback: str = Field(min_length=1)
    query: str = ""
    answer: str = ""
    run_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None


def _days(days: int) -> int:
    return min(max(days, 1), 90)


@router.get("/summary")
def learning_summary(days: int = Query(default=14, ge=1, le=90)) -> dict:
    since = window_start(_days(days))
    payload = summary(since)
    payload["days"] = _days(days)
    payload["candidates_items"] = list_lessons(status="candidate")
    payload["approved_items"] = list_lessons(status="approved")
    payload["incidents_items"] = list_incidents(since=since)
    return payload


@router.get("/lessons")
def learning_lessons(status: str = Query(default="candidate")) -> list[dict]:
    if status not in {"candidate", "approved", "rejected"}:
        raise HTTPException(status_code=400, detail="unknown lesson status")
    return list_lessons(status=status)


@router.get("/incidents")
def learning_incidents(days: int = Query(default=14, ge=1, le=90)) -> list[dict]:
    return list_incidents(since=window_start(_days(days)))


@router.post("/feedback")
def learning_feedback(body: FeedbackIn) -> dict:
    return submit_feedback(
        feedback=body.feedback.strip(),
        query=body.query,
        answer=body.answer,
        run_id=body.run_id,
        session_id=body.session_id,
        user_id=body.user_id,
    )


@router.post("/lessons/{lesson_id}/approve")
def learning_approve(lesson_id: str) -> dict:
    try:
        return approve_lesson(lesson_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="lesson not found") from None
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/lessons/{lesson_id}/reject")
def learning_reject(lesson_id: str) -> dict:
    try:
        return reject_lesson(lesson_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="lesson not found") from None
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/analyze")
def learning_analyze(days: int = Query(default=14, ge=1, le=90)) -> dict:
    return analyze_logs(window_start(_days(days)))


@router.get("/code")
def learning_code(q: str = Query(min_length=1)) -> dict:
    return {"files": search_code(q)}


@router.get("/preview")
def learning_preview(q: str = Query(min_length=1), limit: int = Query(default=5, ge=1, le=20)) -> dict:
    return {"lessons": preview_lessons(q, limit=limit)}


@router.get("/eval.jsonl")
def learning_eval_jsonl() -> PlainTextResponse:
    import json

    lines = [json.dumps(row, ensure_ascii=False) for row in list_eval_cases()]
    body = "\n".join(lines)
    if body:
        body += "\n"
    return PlainTextResponse(
        body,
        media_type="application/x-ndjson",
        headers={"Content-Disposition": 'attachment; filename="eval.jsonl"'},
    )
