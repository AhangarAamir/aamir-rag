"""Custom Agno adapter: persist each agent run (and tool spans) to SQLite."""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from agno.agent import Agent
from agno.run.agent import RunOutput
from agno.utils.log import log_warning

from config import DATA_DIR, OPENAI_MODEL, SESSIONS_DB_FILE
from document_catalog import QUERY_MIX_CODES, identify_collection
from obs_pricing import estimate_cost, model_rates, refresh_prices
from obs_store import get_run, iso, list_run_ids, replace_spans, upsert_run

OBS_APP = os.getenv("OBS_APP", "legal-rag").strip() or "legal-rag"
OBS_TRACE_INNER = os.getenv("OBS_TRACE_INNER", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

_INNER_AGENT_IDS = {"legal-chunker", "query-topic-shift"}
SESSION_IMPORT_ENABLED = os.getenv("OBS_IMPORT_SESSIONS", "true").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}


def resolve_model_id(run_output: RunOutput, agent: Optional[Agent] = None) -> str:
    raw = getattr(run_output, "model", None)
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    ident = getattr(raw, "id", None) if raw is not None else None
    if ident:
        return str(ident)
    if agent is not None:
        model_obj = getattr(agent, "model", None)
        ident = getattr(model_obj, "id", None) if model_obj is not None else None
        if ident:
            return str(ident)
    return OPENAI_MODEL


def classify_query_class(text: str, agent_id: str | None = None) -> str:
    """Return a document-family code (JOA/PEL/PML/PSC/RSC) or empty if unknown."""
    hit = identify_collection(text or "")
    if hit and hit.code in QUERY_MIX_CODES:
        return hit.code
    return ""


def _input_text(run_output: RunOutput) -> str:
    payload = getattr(run_output, "input", None)
    content = getattr(payload, "input_content", None) if payload is not None else None
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, default=str)
    except TypeError:
        return str(content)


def _output_text(run_output: RunOutput) -> str:
    content = run_output.content
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, default=str)
    except TypeError:
        return str(content)


def _unix_to_iso(value: Any) -> str | None:
    if value is None:
        return None
    try:
        stamp = float(value)
    except (TypeError, ValueError):
        return None
    if stamp > 10_000_000_000:
        stamp = stamp / 1000.0
    return iso(datetime.fromtimestamp(stamp, tz=timezone.utc))


def _metrics_tokens(run_output: RunOutput) -> tuple[int, int, int, int, float | None, float | None]:
    metrics = run_output.metrics
    if metrics is None:
        return 0, 0, 0, 0, None, None
    input_tokens = int(getattr(metrics, "input_tokens", 0) or 0)
    output_tokens = int(getattr(metrics, "output_tokens", 0) or 0)
    cached = int(getattr(metrics, "cache_read_tokens", 0) or 0)
    total = int(getattr(metrics, "total_tokens", 0) or (input_tokens + output_tokens))
    cost = getattr(metrics, "cost", None)
    duration = getattr(metrics, "duration", None)
    timer = getattr(metrics, "timer", None)
    if duration is None and timer is not None:
        duration = getattr(timer, "elapsed", None) or getattr(timer, "elapsed_time", None)
    return input_tokens, output_tokens, cached, total, cost, duration


def status_from_raw(raw: Any, *, post_hook: bool = False) -> str:
    """Map Agno/session status onto dashboard buckets.

    Agent post-hooks run *before* Agno sets COMPLETED, so streaming runs still
    look like RUNNING when we record them. Treat that as ok for the hook only.
    """
    text = str(getattr(raw, "value", raw) or "").upper()
    if text in {"ERROR", "CANCELLED"}:
        return "fail"
    if text in {"PENDING", "PAUSED"}:
        return "run"
    if text == "RUNNING" and not post_hook:
        return "run"
    return "ok"


def _status(run_output: RunOutput) -> str:
    return status_from_raw(getattr(run_output, "status", None), post_hook=True)


def _tool_get(tool: Any, name: str, default: Any = None) -> Any:
    if isinstance(tool, dict):
        return tool.get(name, default)
    return getattr(tool, name, default)


def _spans_from_tools(tools: Any) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for tool in tools or []:
        metrics = _tool_get(tool, "metrics")
        duration = None
        if isinstance(metrics, dict):
            duration = metrics.get("duration")
        elif metrics is not None:
            duration = getattr(metrics, "duration", None)
        if duration is not None:
            duration = float(duration) * 1000.0
        args = _tool_get(tool, "tool_args") or {}
        if not isinstance(args, dict):
            args = {}
        detail = args.get("thought") or args.get("query") or args.get("analysis") or ""
        result = _tool_get(tool, "result")
        if not detail and result:
            detail = str(result)[:400]
        spans.append(
            {
                "span_id": _tool_get(tool, "tool_call_id") or uuid.uuid4().hex,
                "name": _tool_get(tool, "tool_name") or "tool",
                "kind": "tool",
                "status": "fail" if _tool_get(tool, "tool_call_error", False) else "ok",
                "started_at": _unix_to_iso(_tool_get(tool, "created_at")),
                "duration_ms": duration,
                "detail": str(detail)[:2000] if detail else "",
            }
        )
    return spans


def _tool_spans(run_output: RunOutput) -> list[dict[str, Any]]:
    return _spans_from_tools(run_output.tools)


def record_agent_run(
    run_output: RunOutput,
    agent: Optional[Agent] = None,
    **_: Any,
) -> None:
    """Agno post-hook: write the finished run and its tool spans to SQLite."""
    try:
        agent_id = (
            getattr(run_output, "agent_id", None)
            or (getattr(agent, "id", None) if agent is not None else None)
            or "unknown"
        )
        if agent_id in _INNER_AGENT_IDS and not OBS_TRACE_INNER:
            return
        input_text = _input_text(run_output)
        input_tokens, output_tokens, cached, total, reported_cost, duration_s = _metrics_tokens(
            run_output
        )
        model = resolve_model_id(run_output, agent)
        started = _unix_to_iso(getattr(run_output, "created_at", None)) or iso(
            datetime.now(timezone.utc)
        )
        duration_ms = float(duration_s) * 1000.0 if duration_s else None
        ended = None
        if duration_ms is not None:
            start_dt = datetime.fromisoformat(started.replace("Z", "+00:00"))
            ended = iso(
                datetime.fromtimestamp(start_dt.timestamp() + duration_ms / 1000.0, tz=timezone.utc)
            )
        run_id = getattr(run_output, "run_id", None) or uuid.uuid4().hex
        upsert_run(
            {
                "run_id": run_id,
                "app": OBS_APP,
                "agent_id": agent_id,
                "agent_name": getattr(run_output, "agent_name", None)
                or (getattr(agent, "name", None) if agent is not None else None),
                "route": agent_id,
                "query_class": classify_query_class(input_text, agent_id),
                "status": _status(run_output),
                "model": model,
                "started_at": started,
                "ended_at": ended,
                "duration_ms": duration_ms,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cached_tokens": cached,
                "total_tokens": total,
                "cost_usd": estimate_cost(
                    model, input_tokens, output_tokens, cached, reported_cost
                ),
                "session_id": getattr(run_output, "session_id", None),
                "user_id": getattr(run_output, "user_id", None),
                "input_text": input_text[:8000],
                "output_text": _output_text(run_output)[:8000],
            }
        )
        replace_spans(run_id, _tool_spans(run_output))
    except Exception as exc:
        log_warning(f"Observability adapter failed to record run: {exc}")


def _payload_input_text(payload: dict[str, Any]) -> str:
    raw = payload.get("input")
    if isinstance(raw, dict):
        content = raw.get("input_content")
        if isinstance(content, str):
            return content
        if content is not None:
            try:
                return json.dumps(content, default=str)
            except TypeError:
                return str(content)
    if isinstance(raw, str):
        return raw
    return ""


def _payload_output_text(payload: dict[str, Any]) -> str:
    content = payload.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, default=str)
    except TypeError:
        return str(content)


def _row_from_session(table_row: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any] | None:
    agent_id = payload.get("agent_id") or table_row.get("agent_id") or "unknown"
    if agent_id in _INNER_AGENT_IDS and not OBS_TRACE_INNER:
        return None
    run_id = payload.get("run_id") or table_row.get("run_id")
    if not run_id:
        return None
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    input_tokens = int(metrics.get("input_tokens") or 0)
    output_tokens = int(metrics.get("output_tokens") or 0)
    cached = int(metrics.get("cache_read_tokens") or 0)
    total = int(metrics.get("total_tokens") or (input_tokens + output_tokens))
    duration_s = metrics.get("duration")
    model = payload.get("model") or OPENAI_MODEL
    started = _unix_to_iso(payload.get("created_at") or table_row.get("created_at")) or iso(
        datetime.now(timezone.utc)
    )
    duration_ms = float(duration_s) * 1000.0 if duration_s else None
    ended = _unix_to_iso(table_row.get("updated_at"))
    if duration_ms is not None:
        start_dt = datetime.fromisoformat(started.replace("Z", "+00:00"))
        ended = iso(
            datetime.fromtimestamp(start_dt.timestamp() + duration_ms / 1000.0, tz=timezone.utc)
        )
    input_text = _payload_input_text(payload)
    return {
        "run_id": run_id,
        "app": OBS_APP,
        "agent_id": agent_id,
        "agent_name": payload.get("agent_name"),
        "route": agent_id,
        "query_class": classify_query_class(input_text, agent_id),
        "status": status_from_raw(payload.get("status") or table_row.get("status")),
        "model": model,
        "started_at": started,
        "ended_at": ended,
        "duration_ms": duration_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_tokens": cached,
        "total_tokens": total,
        "cost_usd": estimate_cost(
            model, input_tokens, output_tokens, cached, metrics.get("cost")
        ),
        "session_id": payload.get("session_id") or table_row.get("session_id"),
        "user_id": payload.get("user_id") or table_row.get("user_id"),
        "input_text": input_text[:8000],
        "output_text": _payload_output_text(payload)[:8000],
        "spans": _spans_from_tools(payload.get("tools")),
    }


def _run_tables(conn: sqlite3.Connection) -> list[str]:
    names: list[str] = []
    for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table'"):
        if not str(name).endswith("_runs"):
            continue
        cols = {row[1] for row in conn.execute(f'PRAGMA table_info("{name}")')}
        if {"run_id", "run_data", "status"} <= cols:
            names.append(str(name))
    return names


_SESSION_WATERMARK: dict[str, int] = {}


def import_session_runs(db_files: list[str | Path] | None = None) -> int:
    """Copy completed Agno session runs into observability.db.

    Historical chats live in sessions.db; the dashboard only reads obs_runs.
    Also repairs rows that were recorded as running because the post-hook fired
    before Agno marked the run COMPLETED.
    """
    if not SESSION_IMPORT_ENABLED:
        return 0
    paths = [Path(p) for p in (db_files or [SESSIONS_DB_FILE, DATA_DIR / "agentos.db"])]
    imported = 0
    stuck = set(list_run_ids("run"))
    for path in paths:
        if not path.is_file():
            continue
        key = str(path.resolve())
        watermark = _SESSION_WATERMARK.get(key, 0)
        try:
            conn = sqlite3.connect(str(path), timeout=5)
            conn.row_factory = sqlite3.Row
        except sqlite3.Error as exc:
            log_warning(f"Could not read session db {path}: {exc}")
            continue
        try:
            max_updated = watermark
            for table in _run_tables(conn):
                cols = {row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')}
                wanted = [
                    col
                    for col in (
                        "run_id",
                        "session_id",
                        "agent_id",
                        "user_id",
                        "status",
                        "run_data",
                        "created_at",
                        "updated_at",
                    )
                    if col in cols
                ]
                for raw in conn.execute(f'SELECT {", ".join(wanted)} FROM "{table}"'):
                    row = dict(raw)
                    updated = int(row.get("updated_at") or row.get("created_at") or 0)
                    if updated > max_updated:
                        max_updated = updated
                    if updated <= watermark and row.get("run_id") not in stuck:
                        continue
                    try:
                        payload = json.loads(row.get("run_data") or "{}")
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(payload, dict):
                        continue
                    mapped = _row_from_session(row, payload)
                    if not mapped:
                        continue
                    existing = get_run(mapped["run_id"])
                    if existing and existing.get("status") == "ok" and mapped["status"] == "run":
                        continue
                    spans = mapped.pop("spans", [])
                    upsert_run(mapped)
                    if spans:
                        replace_spans(mapped["run_id"], spans)
                    imported += 1
            _SESSION_WATERMARK[key] = max_updated
        finally:
            conn.close()
    return imported


def attach_observability(agent: Agent) -> Agent:
    refresh_prices()
    try:
        import_session_runs()
    except Exception as exc:
        log_warning(f"Observability session import failed: {exc}")
    hooks = list(agent.post_hooks or [])
    if record_agent_run not in hooks:
        hooks.append(record_agent_run)
    agent.post_hooks = hooks
    return agent
