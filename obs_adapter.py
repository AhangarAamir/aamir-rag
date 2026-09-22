"""Custom Agno adapter: persist each agent run (and tool spans) to SQLite."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from agno.agent import Agent
from agno.run.agent import RunOutput
from agno.utils.log import log_warning

from config import OPENAI_MODEL
from document_catalog import identify_collection
from obs_pricing import estimate_cost, model_rates, refresh_prices
from obs_store import iso, replace_spans, upsert_run

OBS_APP = os.getenv("OBS_APP", "legal-rag").strip() or "legal-rag"
OBS_TRACE_INNER = os.getenv("OBS_TRACE_INNER", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

_INNER_AGENT_IDS = {"legal-chunker", "query-topic-shift"}


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
    if agent_id == "ingest-agent":
        return "ingest"
    hit = identify_collection(text or "")
    if hit:
        return hit.code
    lowered = (text or "").lower()
    if "participating interest" in lowered or " working interest" in lowered:
        return "PI / economics"
    if "workover" in lowered:
        return "workover"
    if "gain/loss" in lowered or "gain or loss" in lowered:
        return "gain/loss/cost"
    if "export" in lowered or "report" in lowered:
        return "export/report"
    if "production" in lowered:
        return "actual production"
    if "econom" in lowered or "royalt" in lowered:
        return "economics"
    return "legal query"


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


def _status(run_output: RunOutput) -> str:
    raw = getattr(run_output.status, "value", run_output.status)
    text = str(raw or "").upper()
    if text in {"ERROR", "CANCELLED"}:
        return "fail"
    if text in {"RUNNING", "PENDING", "PAUSED"}:
        return "run"
    return "ok"


def _tool_spans(run_output: RunOutput) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for tool in run_output.tools or []:
        metrics = getattr(tool, "metrics", None)
        duration = None
        if metrics is not None:
            duration = getattr(metrics, "duration", None)
            if duration is not None:
                duration = float(duration) * 1000.0
        args = getattr(tool, "tool_args", None) or {}
        detail = args.get("thought") or args.get("query") or args.get("analysis") or ""
        if not detail and getattr(tool, "result", None):
            detail = str(tool.result)[:400]
        spans.append(
            {
                "span_id": getattr(tool, "tool_call_id", None) or uuid.uuid4().hex,
                "name": getattr(tool, "tool_name", None) or "tool",
                "kind": "tool",
                "status": "fail" if getattr(tool, "tool_call_error", False) else "ok",
                "started_at": _unix_to_iso(getattr(tool, "created_at", None)),
                "duration_ms": duration,
                "detail": str(detail)[:2000] if detail else "",
            }
        )
    return spans


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


def attach_observability(agent: Agent) -> Agent:
    refresh_prices()
    hooks = list(agent.post_hooks or [])
    if record_agent_run not in hooks:
        hooks.append(record_agent_run)
    agent.post_hooks = hooks
    return agent
