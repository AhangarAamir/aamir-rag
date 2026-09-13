"""Folder-path tags from the live ingest tree (JOA / PSC / PML / …)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

# Keys match folder-name slugs after lowercasing and replacing _ with space.
# Canonical values follow the old ingest collections in
# Ingestion_Agent-Batch-remove_duplicates.ipynb (self.collections).
FAMILY_NAMES = {
    "qpr": "QPR",
    "nec25": "NEC25",
    "nec 25": "NEC25",
    "nec-25": "NEC25",
    "mcr": "MCR",
    "mom": "MOM",
    "polar satelite": "POLAR_SATELITE",
    "polar satellite": "POLAR_SATELITE",
    "audited statement": "AUDITED_STATEMENT",
    "audited statements": "AUDITED_STATEMENT",
    "psc": "PSC",
    "ocr": "OCR",
    "mcm": "MCM",
    "budget": "BUDGET",
    "pel": "PEL",
    "fdp": "FDP",
    "auditedaccounts": "AUDITED_ACCOUNTS",
    "audited accounts": "AUDITED_ACCOUNTS",
    "monthlystatusreport": "MONTHLY_STATUS_REPORT",
    "monthly status report": "MONTHLY_STATUS_REPORT",
    "dgh": "DGH",
    "wpb": "WPB",
    "range clearence": "RANGE_CLEARENCE",
    "range clearance": "RANGE_CLEARENCE",
    "joa": "JOA",
    "jao": "JOA",
    "pml": "PML",
    "rsc": "RSC",
}
VINTAGE_NAMES = {
    "new": "NEW",
    "old": "OLD",
    "new joa": "NEW",
    "old joa": "OLD",
    "new jao": "NEW",
    "old jao": "OLD",
}
REGION_NAMES = {
    "domestic": "Domestic",
    "international": "International",
    "exploration": "Exploration",
}


def chroma_safe(data: Dict[str, object]) -> Dict[str, str | int | float | bool]:
    """Flatten values Chroma will accept (no None, lists, or nested dicts)."""
    out: Dict[str, str | int | float | bool] = {}
    for key, value in data.items():
        if value is None or value == "":
            continue
        if isinstance(value, bool):
            out[key] = value
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            out[key] = value
        elif isinstance(value, (list, tuple)):
            joined = " | ".join(str(item).strip() for item in value if str(item).strip())
            if joined:
                out[key] = joined
        else:
            text = str(value).strip()
            if text:
                out[key] = text
    return out


def folder_tags_from_path(path: Path, root: Path | None = None) -> Dict[str, str]:
    """Turn PSC/Domestic/.../file.pdf into filterable folder metadata."""
    path = Path(path)
    parts: list[str]
    if root is not None:
        try:
            parts = list(path.relative_to(root).parts)
        except ValueError:
            parts = list(path.parts)
    else:
        parts = list(path.parts)
    dirs = [part for part in parts[:-1]] if path.suffix else list(parts)

    family = ""
    vintage = ""
    region = ""
    for raw in dirs:
        key = " ".join(raw.strip().lower().replace("_", " ").split())
        if key in FAMILY_NAMES:
            family = FAMILY_NAMES[key]
        if key in VINTAGE_NAMES:
            vintage = VINTAGE_NAMES[key]
        if key in REGION_NAMES:
            region = REGION_NAMES[key]
        if "new" in key and ("jao" in key or "joa" in key):
            family = "JOA"
            vintage = "NEW"
        elif "old" in key and ("jao" in key or "joa" in key):
            family = "JOA"
            vintage = "OLD"

    tags = {
        "filename": path.name,
        "folder_family": family,
        "folder_vintage": vintage,
        "folder_region": region,
        "source_path": str(path),
    }
    return {key: value for key, value in tags.items() if value}
