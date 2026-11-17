"""Current user text for tools the model calls without arguments."""

from __future__ import annotations

from typing import Any


def latest_user_text(run_context: Any) -> str:
    """Return the latest user message on this run."""
    messages = getattr(run_context, "messages", None) or []
    for message in reversed(list(messages)):
        role = str(getattr(message, "role", "") or "").lower()
        if role not in {"user", "human"} and not role.endswith("user"):
            continue
        content = getattr(message, "content", "") or ""
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    parts.append(str(item.get("text") or item.get("content") or ""))
                else:
                    parts.append(str(getattr(item, "text", "") or ""))
            content = " ".join(part for part in parts if part)
        text = str(content).strip()
        if text:
            return text
    return ""
