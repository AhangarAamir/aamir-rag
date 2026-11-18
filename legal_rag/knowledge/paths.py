"""Citation path helpers that do not import the chunker or the model."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

# Local workspace dump copied under a family folder. It is not a Document Center path.
_WORKSPACE_PATH = re.compile(
    r"aamirWS/data/ril_data/(?:JOA|JAO|PEL|PML|PSC|RSC)/?",
    re.IGNORECASE,
)


def strip_workspace_path(text: str) -> str:
    """Drop aamirWS/data/ril_data/<family> so citations stay family/file."""
    if not text or "aamirws" not in text.lower():
        return text
    return _WORKSPACE_PATH.sub("", text)


def sites_relative_folder(path: Path) -> Optional[str]:
    """Parent folders from the ``sites`` segment down.

    Windows separators are accepted. Returns None when the path has no
    ``sites`` folder, so JOA/PSC paths keep the Document Center rule.
    """
    parts = [part for part in str(path).replace("\\", "/").split("/") if part and part not in {".", ".."}]
    if len(parts) < 2:
        return None
    folders = parts[:-1]
    for index, part in enumerate(folders):
        if part.lower() == "sites":
            return "/".join(folders[index:])
    return None
