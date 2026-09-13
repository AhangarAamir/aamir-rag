"""Per-file ingest so JAO/PSC/PML folder tags stay on the right document."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agno.utils.log import log_error, log_info

from config import INCOMING_DIR
from folder_tags import chroma_safe, folder_tags_from_path

SUPPORTED_SUFFIXES = {".pdf", ".md", ".markdown", ".txt"}


def iter_legal_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files = [
        item
        for item in sorted(path.rglob("*"))
        if item.is_file() and item.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    return files


def ingest_legal_tree(
    path: str | None = None,
    knowledge: Any | None = None,
    reset_aliases: bool = False,
    replace_all: bool = False,
) -> str:
    """Insert each file with folder-derived metadata. Returns a JSON report."""
    from knowledge_store import get_knowledge

    if knowledge is None:
        knowledge = get_knowledge()
    root = Path(path or INCOMING_DIR).expanduser().resolve()
    if not root.exists():
        return json.dumps({"ok": False, "error": f"path does not exist: {root}"})

    if reset_aliases:
        from alias_catalog import reset_aliases as _reset

        _reset()

    if replace_all:
        log_info("Removing existing knowledge content before re-ingest")
        knowledge.remove_all_content()

    files = iter_legal_files(root)
    loaded: list[str] = []
    failed: list[dict[str, str]] = []
    tree_root = root if root.is_dir() else root.parent

    for file_path in files:
        tags = chroma_safe(folder_tags_from_path(file_path, root=tree_root))
        try:
            log_info(f"Ingesting {file_path} with tags {tags}")
            knowledge.insert(
                path=str(file_path),
                name=file_path.name,
                metadata=tags,
                upsert=True,
                skip_if_exists=False,
            )
            loaded.append(file_path.name)
        except Exception as exc:
            log_error(f"ingest failed for {file_path}: {exc}")
            failed.append({"file": file_path.name, "error": str(exc)})

    return json.dumps(
        {
            "ok": not failed,
            "path": str(root),
            "files_seen": len(files),
            "loaded": loaded,
            "failed": failed,
        }
    )
