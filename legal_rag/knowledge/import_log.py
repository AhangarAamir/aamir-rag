"""One timestamped log file for each ingest run."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from agno.knowledge.utils import get_agno_metadata
from agno.utils.log import log_warning

from legal_rag.config import DATA_DIR

IMPORT_LOG_DIR = DATA_DIR / "import_logs"


def _status_text(status: Any) -> str:
    return str(getattr(status, "value", status) or "").lower()


def _failure_from_row(row: Any) -> Optional[dict[str, str]]:
    if "fail" not in _status_text(getattr(row, "status", "")):
        return None
    meta = getattr(row, "metadata", None) or {}
    path = get_agno_metadata(meta, "source_path") or getattr(row, "name", "") or ""
    error = str(getattr(row, "status_message", "") or "").strip() or "failed"
    return {"path": str(path), "error": error}


def failed_files_from_folder(knowledge: Any, site_id: str, user_id: Optional[str]) -> list[dict[str, str]]:
    """Every failed file for one folder ingest.

    Child rows keep the full status message. The parent summary only stores
    the first 25 failures, so this walks the child ids instead.
    """
    row = knowledge.get_content_by_id(site_id, user_id=user_id)
    if row is None:
        return []
    meta = getattr(row, "metadata", None) or {}
    children = get_agno_metadata(meta, "children") or []
    failures: list[dict[str, str]] = []
    if isinstance(children, list) and children:
        for child_id in children:
            child = knowledge.get_content_by_id(child_id, user_id=user_id)
            if child is None:
                continue
            item = _failure_from_row(child)
            if item is not None:
                failures.append(item)
    if not failures:
        own = _failure_from_row(row)
        if own is not None:
            failures.append(own)
    if not failures:
        recorded = get_agno_metadata(meta, "failed") or []
        if isinstance(recorded, list):
            for entry in recorded:
                if not isinstance(entry, dict):
                    continue
                failures.append(
                    {
                        "path": str(entry.get("path") or entry.get("url") or ""),
                        "error": str(entry.get("error") or "failed"),
                    }
                )
    return failures


def render_import_log(
    *,
    source_path: str,
    started_at: datetime,
    finished_at: datetime,
    status: str,
    loaded: int,
    failures: Sequence[Mapping[str, str]],
    note: str = "",
) -> str:
    lines = [
        f"started_at: {started_at.isoformat(timespec='seconds')}",
        f"finished_at: {finished_at.isoformat(timespec='seconds')}",
        f"source_path: {source_path}",
        f"status: {status}",
        f"loaded: {loaded}",
        f"failed: {len(failures)}",
    ]
    if note:
        lines.append(f"note: {note}")
    lines.append("")
    if not failures:
        lines.append("No file failures.")
    else:
        lines.append("FAILED FILES")
        for index, item in enumerate(failures, start=1):
            lines.append(f"[{index}] path: {item.get('path') or ''}")
            lines.append(f"    error: {item.get('error') or 'failed'}")
    lines.append("")
    return "\n".join(lines)


def write_import_log(
    *,
    source_path: str,
    started_at: datetime,
    finished_at: datetime,
    status: str,
    loaded: int,
    failures: Sequence[Mapping[str, str]],
    note: str = "",
    directory: Optional[Path] = None,
) -> Path:
    """Write a new log. A second run in the same second gets a numeric suffix."""
    root = directory or IMPORT_LOG_DIR
    root.mkdir(parents=True, exist_ok=True)
    stamp = finished_at.strftime("%Y-%m-%d_%H%M%S")
    path = root / f"import_{stamp}.log"
    suffix = 2
    while path.exists():
        path = root / f"import_{stamp}_{suffix}.log"
        suffix += 1
    path.write_text(
        render_import_log(
            source_path=source_path,
            started_at=started_at,
            finished_at=finished_at,
            status=status,
            loaded=loaded,
            failures=failures,
            note=note,
        ),
        encoding="utf-8",
    )
    return path


def record_import(
    knowledge: Any,
    *,
    source_path: str,
    started_at: datetime,
    result: str,
    user_id: Optional[str] = None,
    directory: Optional[Path] = None,
) -> str:
    """Write one import log for this run and add its path to the tool JSON."""
    try:
        finished_at = datetime.now()
        try:
            payload = json.loads(result)
        except json.JSONDecodeError:
            payload = {"ok": False, "error": result}
        if not isinstance(payload, dict):
            payload = {"ok": False, "error": result}

        failures: list[dict[str, str]] = []
        site_id = payload.get("site_id")
        if site_id:
            failures = failed_files_from_folder(knowledge, str(site_id), user_id)
        if not failures:
            recorded = payload.get("failed") or []
            if isinstance(recorded, list):
                for entry in recorded:
                    if isinstance(entry, dict):
                        failures.append(
                            {
                                "path": str(entry.get("path") or entry.get("url") or source_path),
                                "error": str(entry.get("error") or "failed"),
                            }
                        )
        if not failures and payload.get("ok") is False:
            error = str(payload.get("error") or payload.get("status_message") or "failed")
            failures.append({"path": source_path, "error": error})

        page_rows = payload.get("page_rows")
        if isinstance(page_rows, int):
            loaded = max(0, page_rows - len(failures))
        elif payload.get("ok") is True and not failures:
            loaded = 1
        else:
            loaded = 0
        status = str(payload.get("status") or ("failed" if payload.get("ok") is False else "completed"))
        log_path = write_import_log(
            source_path=source_path,
            started_at=started_at,
            finished_at=finished_at,
            status=status,
            loaded=loaded,
            failures=failures,
            directory=directory,
        )
        return attach_log_path(json.dumps(payload), log_path)
    except Exception as exc:
        log_warning(f"Could not write import log for {source_path}: {exc}")
        return result


def attach_log_path(result: str, log_path: Path) -> str:
    try:
        payload = json.loads(result)
    except json.JSONDecodeError:
        payload = {"ok": False, "error": result}
    if not isinstance(payload, dict):
        payload = {"ok": False, "error": result}
    payload["import_log"] = str(log_path)
    return json.dumps(payload)
