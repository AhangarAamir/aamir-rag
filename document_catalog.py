"""Document-type catalog from the old RMS ingest notebook.

Source: Ingestion_Agent-Batch-remove_duplicates.ipynb (`PDFIngestPipeline.collections`
and `identify_collection`). Folder names from the aquaforest tree (JOA/New JOA,
PSC/Domestic, PML) plus JAO NEW/OLD from the current corpus layout.

Used at ingest to stamp `doc_family` from path/filename before the chunker agent runs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class DocumentCollection:
    code: str
    description: str
    aliases: Tuple[str, ...]
    incoming_folders: Tuple[str, ...] = ()


# Keep notebook keys (psc, joa, pml, …) as aliases so old paths still classify.
COLLECTIONS: Tuple[DocumentCollection, ...] = (
    DocumentCollection(
        "PSC",
        "Production Sharing Contract",
        ("psc", "production sharing contract", "production sharing contracts"),
        ("PSC/Domestic",),
    ),
    DocumentCollection(
        "JOA",
        "Joint Operating Agreement",
        ("joa", "joint operating agreement", "new joa", "old joa"),
        ("JOA/New JOA", "JOA/Old JOA"),
    ),
    DocumentCollection(
        "JAO",
        "Joint Account / JOA working-interest folders (NEW JAO / OLD JAO)",
        ("jao", "new jao", "old jao", "joint account"),
        ("JAO/NEW JAO", "JAO/OLD JAO"),
    ),
    DocumentCollection(
        "PML",
        "Petroleum Mining Lease",
        ("pml", "petroleum mining lease", "mining lease"),
        ("PML",),
    ),
    DocumentCollection(
        "RSC",
        "Revenue Sharing Contract",
        ("rsc", "revenue sharing contract"),
        ("RSC",),
    ),
    DocumentCollection(
        "PEL",
        "Petroleum Exploration License",
        ("pel", "petroleum exploration license", "petroleum exploration licence"),
        ("PEL",),
    ),
    DocumentCollection(
        "FDP",
        "Field Development Plan",
        ("fdp", "field development plan"),
        ("FDP",),
    ),
    DocumentCollection(
        "DGH",
        "Directorate General of Hydrocarbons",
        ("dgh", "director general hydrocarbons", "directorate general of hydrocarbons"),
        ("DGH",),
    ),
    DocumentCollection(
        "MCM",
        "Management Committee Meeting / Minutes",
        ("mcm", "management committee meeting", "management committee minutes"),
        ("MCM",),
    ),
    DocumentCollection(
        "MCR",
        "Management Committee Resolution",
        ("mcr", "management committee resolution"),
        ("MCR",),
    ),
    DocumentCollection(
        "OCR",
        "Operating Committee Resolution",
        ("operating committee resolution",),
        ("OCR",),
    ),
    DocumentCollection(
        "MOM",
        "Minutes of Meeting",
        ("mom", "minutes of meeting", "minutes of meetings"),
        ("MOM",),
    ),
    DocumentCollection(
        "QPR",
        "Quarterly Performance Report",
        ("qpr", "quarterly performance report", "quarterly performance reports"),
        ("QPR",),
    ),
    DocumentCollection(
        "WPB",
        "Work Programme and Budget",
        ("wpb", "work programme budget", "work program budget", "work programme and budget"),
        ("WPB",),
    ),
    DocumentCollection(
        "BUDGET",
        "Budget data",
        ("budget",),
        ("budget",),
    ),
    DocumentCollection(
        "AUDITED_STATEMENT",
        "Audited statements",
        ("audited_statement", "audited statement", "audited statements"),
        ("audited_statement",),
    ),
    DocumentCollection(
        "AUDITED_ACCOUNTS",
        "Audited accounts",
        ("auditedaccounts", "audited accounts", "audited account"),
        ("auditedaccounts",),
    ),
    DocumentCollection(
        "MONTHLY_STATUS_REPORT",
        "Monthly status reports",
        ("monthlystatusreport", "monthly status report", "monthly status reports"),
        ("monthlystatusreport",),
    ),
    DocumentCollection(
        "RANGE_CLEARANCE",
        "Range clearance",
        ("range_clearence", "range_clearance", "range clearance", "range clearence"),
        ("range_clearence",),
    ),
    DocumentCollection(
        "POLAR_SATELLITE",
        "Polar satellite launch data",
        ("polar_satelite", "polar_satellite", "polar satelite", "polar satellite"),
        ("polar_satelite",),
    ),
    DocumentCollection(
        "NEC25",
        "NEC-OSN-97-2 (NEC-25) block documents",
        ("nec25", "nec-25", "nec 25", "nec-osn-97-2", "nec osn 97 2", "nec-osn 97/2"),
        ("NEC25",),
    ),
    DocumentCollection(
        "ASSIGNMENT",
        "Assignment / farm-in / farm-out",
        ("assignment", "farm-in", "farmin", "farm-out", "farmout"),
        (),
    ),
    DocumentCollection(
        "NOTIFICATION",
        "Government / MoPNG notification",
        ("notification", "mopng"),
        (),
    ),
    DocumentCollection(
        "AMENDMENT",
        "Contract amendment",
        ("amendment",),
        (),
    ),
)

# Longest alias first so "new joa" wins over "joa".
_ALIAS_INDEX: Tuple[Tuple[str, DocumentCollection], ...] = tuple(
    sorted(
        ((alias.lower(), collection) for collection in COLLECTIONS for alias in collection.aliases),
        key=lambda item: len(item[0]),
        reverse=True,
    )
)

# Instrument / folder families beat status words like "amendment" in a JOA filename.
_PRIMARY_CODES = {
    "PSC",
    "JOA",
    "JAO",
    "PML",
    "RSC",
    "PEL",
    "FDP",
    "DGH",
    "MCM",
    "MCR",
    "OCR",
    "MOM",
    "QPR",
    "WPB",
    "BUDGET",
    "AUDITED_STATEMENT",
    "AUDITED_ACCOUNTS",
    "MONTHLY_STATUS_REPORT",
    "RANGE_CLEARANCE",
    "POLAR_SATELLITE",
}

_CODE_INDEX = {collection.code.upper(): collection for collection in COLLECTIONS}
for collection in COLLECTIONS:
    _CODE_INDEX[collection.code.lower()] = collection
    _CODE_INDEX[collection.code.replace("_", "").lower()] = collection


def _hay(text: str) -> str:
    return _NON_ALNUM.sub(" ", (text or "").lower()).strip()


def _alias_in_hay(alias: str, hay: str) -> bool:
    needle = _hay(alias)
    if not needle:
        return False
    if " " in needle:
        return needle in hay
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", hay))


def match_collection_text(text: str, *, primary_only: bool = False) -> Optional[DocumentCollection]:
    hay = _hay(text)
    if not hay:
        return None
    for alias, collection in _ALIAS_INDEX:
        if primary_only and collection.code not in _PRIMARY_CODES:
            continue
        if _alias_in_hay(alias, hay):
            return collection
    compact = hay.replace(" ", "")
    hit = _CODE_INDEX.get(compact)
    if hit and (not primary_only or hit.code in _PRIMARY_CODES):
        return hit
    return None


def identify_collection(*parts: str) -> Optional[DocumentCollection]:
    """Pick a collection from folder path, then filename.

    Call with folder first so ``JOA/New JOA/NEC-25....pdf`` stays JOA.
    Leaf folders win inside a path (``PSC/Domestic`` -> PSC).
    """

    def _scan(primary_only: bool) -> Optional[DocumentCollection]:
        for part in parts:
            if not part:
                continue
            segments = re.split(r"[\\/]+", str(part))
            for segment in reversed(segments):
                hit = match_collection_text(segment, primary_only=primary_only)
                if hit:
                    return hit
            hit = match_collection_text(str(part), primary_only=primary_only)
            if hit:
                return hit
        return None

    return _scan(True) or _scan(False)


def identify_from_path(path: Path) -> Optional[DocumentCollection]:
    return identify_collection(str(path.parent), path.name, path.stem)


def normalize_family_code(value: str) -> str:
    if not value:
        return ""
    hit = match_collection_text(value) or _CODE_INDEX.get(value.strip().upper())
    if hit:
        return hit.code
    compact = re.sub(r"[^A-Z0-9]+", "", value.strip().upper())
    return compact


def collection_by_code(code: str) -> Optional[DocumentCollection]:
    if not code:
        return None
    return _CODE_INDEX.get(code.strip().upper()) or match_collection_text(code)


def family_codes() -> Tuple[str, ...]:
    return tuple(collection.code for collection in COLLECTIONS)


def family_prompt_block() -> str:
    return "\n".join(f"      - {item.code}: {item.description}" for item in COLLECTIONS)


def incoming_layout() -> Tuple[str, ...]:
    folders: list[str] = []
    for collection in COLLECTIONS:
        folders.extend(collection.incoming_folders)
    return tuple(dict.fromkeys(folders))


def ensure_incoming_layout(root: Path, folders: Optional[Sequence[str]] = None) -> None:
    for relative in folders if folders is not None else incoming_layout():
        target = root / relative
        target.mkdir(parents=True, exist_ok=True)
        keep = target / ".gitkeep"
        if not keep.exists():
            keep.write_text("", encoding="utf-8")
