"""One log file per imported document.

The file is written when that document finishes. Its name is the document
name plus ``.log``, for example ``budget.0.1.pdf.log``.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from agno.knowledge.utils import get_agno_metadata
from agno.utils.log import log_warning

from legal_rag.config import DATA_DIR

IMPORT_LOG_DIR = DATA_DIR / "import_logs"
_ILLEGAL_LOG_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


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


def document_log_name(file_name: str) -> str:
    """``budget.0.1.pdf`` becomes ``budget.0.1.pdf.log``."""
    raw = str(file_name or "").replace("\\", "/")
    name = Path(raw).name.strip()
    name = _ILLEGAL_LOG_CHARS.sub("_", name).rstrip(" .")
    return f"{name or 'document'}.log"


def write_document_log(
    *,
    file_name: str,
    source_path: str,
    status: str,
    message: str = "",
    finished_at: Optional[datetime] = None,
    directory: Optional[Path] = None,
) -> Path:
    """Write ``<file name>.log``. A second import of the same name keeps both logs."""
    root = directory or IMPORT_LOG_DIR
    root.mkdir(parents=True, exist_ok=True)
    when = finished_at or datetime.now()
    path = _unique_log_path(root, document_log_name(file_name), when)
    lines = [
        f"file: {source_path}",
        f"finished_at: {when.isoformat(timespec='seconds')}",
        f"status: {status}",
    ]
    if message:
        lines.append(f"message: {message}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _unique_log_path(root: Path, log_name: str, finished_at: datetime) -> Path:
    path = root / log_name
    if not path.exists():
        return path
    stamp = finished_at.strftime("%Y-%m-%d_%H%M%S")
    stem = log_name[:-4] if log_name.lower().endswith(".log") else log_name
    stamped = root / f"{stem}_{stamp}.log"
    if not stamped.exists():
        return stamped
    suffix = 2
    while True:
        candidate = root / f"{stem}_{stamp}_{suffix}.log"
        if not candidate.exists():
            return candidate
        suffix += 1


def _content_file(content: Any) -> Optional[Path]:
    raw = getattr(content, "path", None)
    if not raw:
        return None
    path = Path(str(raw))
    if path.is_file():
        return path
    return None


def _log_finished_document(
    knowledge: Any,
    content: Any,
    path: Path,
    error: Optional[BaseException],
) -> None:
    if error is not None:
        status = "failed"
        message = f"{type(error).__name__}: {error}"
    else:
        status = _status_text(getattr(content, "status", "")) or "completed"
        message = str(getattr(content, "status_message", "") or "").strip()
        if "fail" in status:
            status = "failed"
    write_document_log(
        file_name=path.name,
        source_path=str(path),
        status=status,
        message=message,
        finished_at=datetime.now(),
        directory=getattr(knowledge, "_import_log_dir", None),
    )


def install_per_file_import_logs(knowledge: Any) -> None:
    """Write one log as soon as each file finishes inside path ingest."""
    if getattr(knowledge, "_document_import_logs", False):
        return
    original_async = knowledge._aload_from_path
    original_sync = knowledge._load_from_path

    async def _aload_from_path(content, upsert, skip_if_exists, include=None, exclude=None):
        path = _content_file(content)
        try:
            await original_async(content, upsert, skip_if_exists, include, exclude)
        except Exception as exc:
            if path is not None:
                _log_finished_document(knowledge, content, path, exc)
            raise
        if path is not None:
            _log_finished_document(knowledge, content, path, None)

    def _load_from_path(content, upsert, skip_if_exists, include=None, exclude=None):
        path = _content_file(content)
        try:
            original_sync(content, upsert, skip_if_exists, include, exclude)
        except Exception as exc:
            if path is not None:
                _log_finished_document(knowledge, content, path, exc)
            raise
        if path is not None:
            _log_finished_document(knowledge, content, path, None)

    knowledge._aload_from_path = _aload_from_path
    knowledge._load_from_path = _load_from_path
    knowledge._document_import_logs = True


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
    """Point the tool result at the per-document logs written during ingest."""
    try:
        finished_at = datetime.now()
        try:
            payload = json.loads(result)
        except json.JSONDecodeError:
            payload = {"ok": False, "error": result}
        if not isinstance(payload, dict):
            payload = {"ok": False, "error": result}

        root = directory or IMPORT_LOG_DIR
        payload["import_log_dir"] = str(root)
        # Per-file logs are written as each document finishes. This covers a
        # path ingest that fails before any document row is created.
        if payload.get("ok") is False and not payload.get("site_id"):
            error = str(payload.get("error") or payload.get("status_message") or "failed")
            log_path = write_document_log(
                file_name=source_path,
                source_path=source_path,
                status="failed",
                message=error,
                finished_at=finished_at,
                directory=root,
            )
            payload["import_log"] = str(log_path)
        return json.dumps(payload)
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
