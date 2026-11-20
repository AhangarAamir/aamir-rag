"""One log file per imported document.

The file is written when that document finishes. Its name is the document
name plus ``.log``, for example ``budget.0.1.pdf.log``.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from legal_rag.config import DATA_DIR

IMPORT_LOG_DIR = DATA_DIR / "import_logs"
_ILLEGAL_LOG_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


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
