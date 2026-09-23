"""HTTP API that feeds the custom observability dashboard from SQLite traces."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from statistics import median
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from obs_adapter import classify_query_class, import_session_runs
from document_catalog import QUERY_MIX_CODES
from obs_pricing import catalog_status, model_rates, pricing_source
from obs_store import add_comment, list_apps, list_comments, list_runs, utc_now, window_start

router = APIRouter(prefix="/observability", tags=["observability"])

# Tokens-vs-latency is fixed at 0–200s. Longer values are bad durations, not real runs.
MAX_PLOT_LATENCY_MS = 200_000.0


class CommentIn(BaseModel):
    comment: str = Field(min_length=1)
    run_id: Optional[str] = None
    query: str = ""
    history: str = ""


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (pct / 100.0)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _plot_duration_ms(row: dict[str, Any]) -> float | None:
    raw = row.get("duration_ms")
    if raw is None:
        return None
    value = float(raw)
    if value < 0 or value > MAX_PLOT_LATENCY_MS:
        return None
    return value


def _day_key(started_at: str) -> str:
    dt = datetime.fromisoformat(started_at.replace("Z", "+00:00")).astimezone(timezone.utc)
    return dt.date().isoformat()


def _build_summary(runs: list[dict[str, Any]], comments: list[dict[str, Any]], days: int) -> dict[str, Any]:
    now = utc_now()
    ok = [row for row in runs if row["status"] == "ok"]
    fail = [row for row in runs if row["status"] == "fail"]
    running = [row for row in runs if row["status"] == "run"]
    durations = [ms for row in runs if (ms := _plot_duration_ms(row)) is not None]
    tokens_in = sum(int(row["input_tokens"] or 0) for row in runs)
    tokens_out = sum(int(row["output_tokens"] or 0) for row in runs)
    tokens_cached = sum(int(row["cached_tokens"] or 0) for row in runs)
    tokens_total = sum(int(row["total_tokens"] or 0) for row in runs) or (tokens_in + tokens_out)
    cost = sum(float(row["cost_usd"] or 0) for row in runs)
    success_rate = (len(ok) / len(runs) * 100.0) if runs else 0.0
    avg_ms = (sum(durations) / len(durations)) if durations else 0.0
    p95_ms = _percentile(durations, 95)
    med_ms = median(durations) if durations else 0.0

    by_day: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "ok": 0,
            "fail": 0,
            "run": 0,
            "durations": [],
            "cost": 0.0,
            "cost_input": 0.0,
            "cost_cached": 0.0,
            "cost_output": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_tokens": 0,
        }
    )
    by_class: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "ok": 0, "duration_sum": 0.0, "duration_n": 0}
    )
    by_route: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "ok": 0, "duration_sum": 0.0, "duration_n": 0}
    )
    by_model: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "runs": 0,
            "input_tokens": 0,
            "cached_tokens": 0,
            "output_tokens": 0,
            "cost": 0.0,
        }
    )
    scatter: list[dict[str, Any]] = []

    for row in runs:
        day = _day_key(row["started_at"])
        bucket = by_day[day]
        status = row["status"] or "ok"
        bucket[status] = int(bucket.get(status, 0)) + 1
        duration_ms = _plot_duration_ms(row)
        if duration_ms is not None:
            bucket["durations"].append(duration_ms)
        bucket["cost"] += float(row["cost_usd"] or 0)
        inp_tok = int(row["input_tokens"] or 0)
        out_tok = int(row["output_tokens"] or 0)
        cached_tok = int(row["cached_tokens"] or 0)
        bucket["input_tokens"] += inp_tok
        bucket["output_tokens"] += out_tok
        bucket["cached_tokens"] += cached_tok
        inp_rate, cached_rate, out_rate = model_rates(row.get("model"))
        billed_input = max(inp_tok - cached_tok, 0)
        bucket["cost_input"] += billed_input * inp_rate / 1_000_000.0
        bucket["cost_cached"] += cached_tok * cached_rate / 1_000_000.0
        bucket["cost_output"] += out_tok * out_rate / 1_000_000.0

        cls = classify_query_class(row.get("input_text") or "", row.get("agent_id"))
        if not cls:
            stored = row.get("query_class") or ""
            cls = stored if stored in QUERY_MIX_CODES else ""
        if not cls:
            cls = None
        if cls:
            by_class[cls]["count"] += 1
            if status == "ok":
                by_class[cls]["ok"] += 1
            if duration_ms is not None:
                by_class[cls]["duration_sum"] += duration_ms
                by_class[cls]["duration_n"] += 1

        route = row.get("route") or row.get("agent_id") or "unknown"
        by_route[route]["count"] += 1
        if status == "ok":
            by_route[route]["ok"] += 1
        if duration_ms is not None:
            by_route[route]["duration_sum"] += duration_ms
            by_route[route]["duration_n"] += 1

        model = row.get("model") or "unknown"
        by_model[model]["runs"] += 1
        by_model[model]["input_tokens"] += int(row["input_tokens"] or 0)
        by_model[model]["cached_tokens"] += int(row["cached_tokens"] or 0)
        by_model[model]["output_tokens"] += int(row["output_tokens"] or 0)
        by_model[model]["cost"] += float(row["cost_usd"] or 0)

        if duration_ms is not None:
            scatter.append(
                {
                    "tokens": int(row["total_tokens"] or 0)
                    or (int(row["input_tokens"] or 0) + int(row["output_tokens"] or 0)),
                    "latency_s": duration_ms / 1000.0,
                    "latency_ms": duration_ms,
                    "run_id": row["run_id"],
                    "route": route,
                }
            )

    peak_day = ""
    peak_count = 0
    for day, bucket in by_day.items():
        total = int(bucket.get("ok", 0)) + int(bucket.get("fail", 0)) + int(bucket.get("run", 0))
        if total > peak_count:
            peak_count = total
            peak_day = day

    slowest_class = ""
    slowest_class_ms = 0.0
    for name, stats in by_class.items():
        if stats["duration_n"]:
            avg = stats["duration_sum"] / stats["duration_n"]
            if avg > slowest_class_ms:
                slowest_class_ms = avg
                slowest_class = name

    slowest_route = ""
    slowest_route_ms = 0.0
    for name, stats in by_route.items():
        if stats["duration_n"]:
            avg = stats["duration_sum"] / stats["duration_n"]
            if avg > slowest_route_ms:
                slowest_route_ms = avg
                slowest_route = name

    top_model = ""
    top_model_runs = 0
    for name, stats in by_model.items():
        if stats["runs"] > top_model_runs:
            top_model_runs = stats["runs"]
            top_model = name

    days_series = []
    for day in sorted(by_day):
        bucket = by_day[day]
        durations = bucket["durations"]
        days_series.append(
            {
                "date": day,
                "ok": int(bucket.get("ok", 0)),
                "fail": int(bucket.get("fail", 0)),
                "run": int(bucket.get("run", 0)),
                "avg_ms": (sum(durations) / len(durations)) if durations else 0.0,
                "p95_ms": _percentile(durations, 95),
                "cost": float(bucket["cost"]),
                "cost_input": float(bucket["cost_input"]),
                "cost_cached": float(bucket["cost_cached"]),
                "cost_output": float(bucket["cost_output"]),
                "input_tokens": int(bucket["input_tokens"]),
                "output_tokens": int(bucket["output_tokens"]),
                "cached_tokens": int(bucket["cached_tokens"]),
            }
        )

    query_mix = []
    for name in QUERY_MIX_CODES:
        stats = by_class.get(name) or {"count": 0, "ok": 0, "duration_sum": 0.0, "duration_n": 0}
        query_mix.append(
            {
                "name": name,
                "count": stats["count"],
                "success": (stats["ok"] / stats["count"] * 100.0) if stats["count"] else 0.0,
                "avg_ms": (stats["duration_sum"] / stats["duration_n"]) if stats["duration_n"] else 0.0,
            }
        )
    query_mix.sort(key=lambda row: row["count"])

    routes = []
    for name, stats in sorted(by_route.items(), key=lambda item: item[1]["count"], reverse=True):
        routes.append(
            {
                "name": name,
                "count": stats["count"],
                "success": (stats["ok"] / stats["count"] * 100.0) if stats["count"] else 0.0,
                "avg_ms": (stats["duration_sum"] / stats["duration_n"]) if stats["duration_n"] else 0.0,
            }
        )

    models = []
    for name, stats in sorted(by_model.items(), key=lambda item: item[1]["cost"], reverse=True):
        inp, cached, out = model_rates(name)
        models.append(
            {
                "name": name,
                "runs": stats["runs"],
                "input_tokens": stats["input_tokens"],
                "cached_tokens": stats["cached_tokens"],
                "output_tokens": stats["output_tokens"],
                "cost": stats["cost"],
                "input_rate": inp,
                "cached_rate": cached,
                "output_rate": out,
                "source": pricing_source(name),
            }
        )

    return {
        "days": days,
        "generated_at": now.isoformat(),
        "runs": len(runs),
        "ok": len(ok),
        "fail": len(fail),
        "running": len(running),
        "success_rate": success_rate,
        "avg_ms": avg_ms,
        "median_ms": med_ms,
        "p95_ms": p95_ms,
        "tokens": tokens_total,
        "input_tokens": tokens_in,
        "output_tokens": tokens_out,
        "cached_tokens": tokens_cached,
        "cost": cost,
        "feedback": len(comments),
        "feedback_rate": (len(comments) / len(runs) * 100.0) if runs else 0.0,
        "peak_day": peak_day,
        "peak_day_runs": peak_count,
        "slowest_class": slowest_class,
        "slowest_class_ms": slowest_class_ms,
        "slowest_route": slowest_route,
        "slowest_route_ms": slowest_route_ms,
        "model": top_model,
        "days_series": days_series,
        "query_mix": query_mix,
        "routes": routes,
        "models": models,
        "pricing": catalog_status(),
        "scatter": scatter,
        "comments": comments,
        "apps": list_apps(),
    }


@router.get("/summary")
def observability_summary(
    days: int = Query(14, ge=1, le=365),
    app: Optional[str] = Query(None),
):
    try:
        import_session_runs()
    except Exception:
        pass
    since = window_start(days)
    runs = list_runs(since, app=app)
    comments = list_comments(since, app=app)
    return _build_summary(runs, comments, days)


@router.get("/runs")
def observability_runs(
    days: int = Query(14, ge=1, le=365),
    app: Optional[str] = Query(None),
    limit: int = Query(200, ge=1, le=2000),
):
    since = window_start(days)
    runs = list_runs(since, app=app)
    return {"data": runs[-limit:]}


@router.get("/apps")
def observability_apps():
    return {"data": list_apps()}


@router.get("/pricing")
def observability_pricing():
    return catalog_status()


@router.post("/comments")
def observability_add_comment(body: CommentIn):
    try:
        return add_comment(
            comment=body.comment.strip(),
            run_id=body.run_id,
            query=body.query,
            history=body.history,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
