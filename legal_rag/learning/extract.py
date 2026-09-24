"""Turn feedback or a bad run into one candidate lesson.

The model call is replaceable so tests never hit OpenAI.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from legal_rag.config import OPENAI_MODEL, ROOT_DIR, require_api_key

SEVERITIES = {"low", "medium", "high", "critical"}
_SKIP_DIRS = {".venv", "node_modules", "dist", "data", "__pycache__", ".git"}

Extractor = Callable[[str], str]
_extractor: Extractor | None = None

EXTRACT_SYSTEM = """You write one reusable lesson for a legal contract query agent.
Reply with JSON only, no markdown:
{
  "title": "short searchable title",
  "lesson": "For similar legal assistant questions, ...",
  "severity": "low|medium|high|critical",
  "tags": ["short_tag"],
  "confidence": 0.0,
  "files": ["prompts.py"],
  "area": "contract",
  "context": "when this lesson applies"
}
The lesson must be a general rule a later question can follow. Do not store the user's private facts.
Cite only files from the list you are given.
"""


def configure_extractor(fn: Extractor | None) -> None:
    global _extractor
    _extractor = fn


def source_files() -> list[str]:
    found: list[str] = []
    for path in ROOT_DIR.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in {".py", ".jsx", ".js", ".css", ".md"}:
            continue
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        found.append(path.relative_to(ROOT_DIR).as_posix())
    return sorted(found)


def _clip(text: str, limit: int = 4000) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


def build_prompt(
    *,
    query: str,
    answer: str,
    feedback: str,
    incident_type: str,
    files: list[str],
) -> str:
    catalog = "\n".join(f"- {name}" for name in files[:80])
    return (
        f"Incident type: {incident_type}\n"
        f"User question:\n{_clip(query)}\n\n"
        f"Assistant answer:\n{_clip(answer)}\n\n"
        f"Feedback or error:\n{_clip(feedback)}\n\n"
        f"Source files you may cite:\n{catalog}\n"
    )


def _model_text(prompt: str) -> str:
    require_api_key()
    from agno.models.message import Message
    from agno.models.openai import OpenAIResponses

    model = OpenAIResponses(id=OPENAI_MODEL)
    response = model.response(
        messages=[
            Message(role="system", content=EXTRACT_SYSTEM),
            Message(role="user", content=prompt),
        ]
    )
    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return str(content)


def _parse_json(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("lesson JSON must be an object")
    return data


def _fallback(feedback: str, incident_type: str) -> dict[str, Any]:
    note = (feedback or incident_type or "the previous answer was wrong").strip()
    return {
        "title": f"Correction: {incident_type or 'user_feedback'}",
        "lesson": (
            "For similar legal assistant questions, treat the user correction as "
            "evidence of a failure and re-check the instrument, clause, and citations "
            f"before answering. User correction: {note}"
        ),
        "severity": "medium" if incident_type != "raw_error" else "high",
        "tags": [incident_type or "user_feedback"],
        "confidence": 0.45,
        "files": ["legal_rag/agents/query.py", "legal_rag/prompts.py"],
        "area": "contract",
        "context": "Questions similar to the one that drew this correction",
    }


def normalize_draft(data: dict[str, Any], *, incident_type: str, feedback: str) -> dict[str, Any]:
    fallback = _fallback(feedback, incident_type)
    title = str(data.get("title") or fallback["title"]).strip()[:180]
    lesson = str(data.get("lesson") or fallback["lesson"]).strip()
    severity = str(data.get("severity") or "medium").strip().lower()
    if severity not in SEVERITIES:
        severity = "medium"
    tags_raw = data.get("tags") or []
    if isinstance(tags_raw, str):
        tags_raw = [tags_raw]
    tags = [str(tag).strip()[:40] for tag in tags_raw if str(tag).strip()][:6]
    if incident_type and incident_type not in tags:
        tags.insert(0, incident_type)
    try:
        confidence = float(data.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))
    allowed = set(source_files())
    files_raw = data.get("files") or []
    if isinstance(files_raw, str):
        files_raw = [files_raw]
    files = []
    for name in files_raw:
        rel = str(name).strip().lstrip("./")
        if rel in allowed and rel not in files:
            files.append(rel)
    if not files:
        files = [name for name in ("legal_rag/agents/query.py", "legal_rag/prompts.py") if name in allowed]
    area = str(data.get("area") or "contract").strip()[:40] or "contract"
    context = str(data.get("context") or fallback["context"]).strip()
    return {
        "title": title or fallback["title"],
        "lesson": lesson or fallback["lesson"],
        "severity": severity,
        "tags": tags[:6],
        "confidence": confidence,
        "files": files[:8],
        "area": area,
        "context": context,
    }


def extract_lesson(
    *,
    query: str,
    answer: str,
    feedback: str,
    incident_type: str,
) -> dict[str, Any]:
    prompt = build_prompt(
        query=query,
        answer=answer,
        feedback=feedback,
        incident_type=incident_type,
        files=source_files(),
    )
    try:
        raw = (_extractor or _model_text)(prompt)
        return normalize_draft(_parse_json(raw), incident_type=incident_type, feedback=feedback)
    except Exception:
        return normalize_draft({}, incident_type=incident_type, feedback=feedback)
