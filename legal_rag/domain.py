"""Corpus knowledge that should not depend on the model: abbreviations,
document families, and citation links.

aamir-rag asked the model to expand abbreviations and build Document Center
URLs from long prompt rules. Small open-source models skip those rules, so
here they are code: search expands the query itself, and every hit carries a
ready-made link.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Iterable, Optional
from urllib.parse import quote

from legal_rag.config import DOC_CENTER_BASE_URL, SITES_BASE_URL

# --------------------------------------------------------------- glossary
# Each group is a set of equivalent names. The first entry is the label
# shown to the model. Entries marked with a leading "!" only match when the
# user typed them in capitals (short tokens such as AP, MC, ML are common
# English otherwise).
GLOSSARY: tuple[tuple[str, ...], ...] = (
    ("Production Sharing Contract", "PSC"),
    ("Joint Operating Agreement", "JOA"),
    ("Revenue Sharing Contract", "RSC"),
    ("Petroleum Exploration Licence", "Petroleum Exploration License", "PEL"),
    ("Petroleum Mining Lease", "PML", "Mining Lease", "!ML"),
    ("Field Development Plan", "FDP"),
    ("Appraisal Programme", "Appraisal Program", "!AP"),
    ("Notification of Discovery", "NoD", "NOD"),
    ("Potential Commercial Discovery", "!PoD"),
    ("Potential Commercial Interest", "PCI"),
    ("Declaration of Commerciality", "!DoC"),
    ("New Exploration Licensing Policy", "NELP"),
    ("Reliance Industries Limited", "RIL", "Reliance"),
    ("Government of India", "GoI", "GOI", "Government"),
    ("Directorate General of Hydrocarbons", "DGH"),
    ("Ministry of Petroleum and Natural Gas", "MoPNG", "MOPNG", "MoP&NG", "moping"),
    ("Management Committee", "!MC", "MCM", "MCR"),
    ("Operating Committee", "!OC", "OCR"),
    ("Participating Interest", "PI"),
    ("Krishna Godavari D6", "KG-D6", "KGD6", "KG D6", "KG-DWN-98/3", "KG-DWN-98-3"),
    ("NEC-OSN-97/2", "NEC25", "NEC-25", "NEC-OSN-97-2"),
    ("KG-UDWHP-2018/1", "KGUDWHP-2018-1"),
    ("KG-UDWHP-2022/1", "KGUDWHP-2022-1"),
    ("Dhirubhai-1", "D1"),
    ("Dhirubhai-2", "D2"),
    ("Dhirubhai-3", "D3"),
    ("D1-D3", "D1D2D3", "D1 D3"),
    ("Dhirubhai-34", "D-34", "D34", "R-Cluster", "R Cluster"),
    ("Satellite fields", "Satellite Cluster", "SATs", "!Sat"),
    ("Other Satellite fields", "OSW", "OSWs"),
    ("Profit Petroleum", "Profit Oil", "Profit Gas"),
    ("Cost Petroleum", "Cost Recovery"),
    ("Investment Multiple", "!IM"),
    ("Royalty", "well-head value"),
    ("Assignment", "transfer of participating interest", "farm-in", "farm-out"),
    ("Relinquishment", "relinquish"),
    ("Appendix", "Annex", "Annexure", "Schedule"),
    ("Work Programme and Budget", "Work Program and Budget", "WPB"),
    ("Minutes of Meeting", "MOM", "minutes"),
    ("Quarterly Performance Report", "QPR"),
)

_TYPOS = {
    "reserviou": "reservoir",
    "resevoir": "reservoir",
    "statues": "states",
    "relinquishement": "relinquishment",
    "participating intrest": "participating interest",
    "particpating": "participating",
    "aggrement": "agreement",
    "agrement": "agreement",
    "royality": "royalty",
}


def _variant_pattern(variant: str) -> tuple[re.Pattern[str], str]:
    case_sensitive = variant.startswith("!")
    text = variant.lstrip("!")
    body = re.escape(text).replace(r"\ ", r"[\s-]*").replace(r"\-", r"[\s-]*")
    flags = 0 if case_sensitive else re.IGNORECASE
    return re.compile(rf"(?<![A-Za-z0-9]){body}(?![A-Za-z0-9])", flags), text


_GLOSSARY_PATTERNS = [[_variant_pattern(v) for v in group] for group in GLOSSARY]


def fix_typos(text: str) -> str:
    out = text
    for wrong, right in _TYPOS.items():
        out = re.sub(rf"(?i)\b{re.escape(wrong)}\b", right, out)
    return out


def expansions(query: str) -> list[str]:
    """Equivalent names for every glossary entry found in the query."""
    found: list[str] = []
    for group in _GLOSSARY_PATTERNS:
        if any(pattern.search(query) for pattern, _ in group):
            for pattern, text in group:
                if not pattern.search(query) and text not in found:
                    found.append(text)
    return found


def glossary_variants(text: str) -> list[str]:
    """Every name in each glossary group that the text mentions (including the match)."""
    out: list[str] = []
    for group in _GLOSSARY_PATTERNS:
        if any(pattern.search(text or "") for pattern, _ in group):
            out.extend(t for _, t in group if t not in out)
    return out


def glossary_prompt() -> str:
    lines = []
    for group in GLOSSARY:
        names = [v.lstrip("!") for v in group]
        lines.append(f"- {names[0]} = {', '.join(names[1:])}")
    return "\n".join(lines)


# --------------------------------------------------------------- families

@dataclass(frozen=True)
class Family:
    code: str
    description: str
    aliases: tuple[str, ...]


FAMILIES: tuple[Family, ...] = (
    Family("PSC", "Production Sharing Contract", ("psc", "production sharing contract")),
    Family("JOA", "Joint Operating Agreement", ("joa", "joint operating agreement", "new joa", "old joa")),
    Family("JAO", "Joint Account / JOA working-interest", ("jao", "new jao", "old jao")),
    Family("PML", "Petroleum Mining Lease", ("pml", "petroleum mining lease", "mining lease")),
    Family("RSC", "Revenue Sharing Contract", ("rsc", "revenue sharing contract")),
    Family("PEL", "Petroleum Exploration Licence", ("pel", "petroleum exploration licence", "petroleum exploration license")),
    Family("FDP", "Field Development Plan", ("fdp", "field development plan")),
    Family("DGH", "Directorate General of Hydrocarbons", ("dgh", "directorate general of hydrocarbons")),
    Family("MCR_MCM", "Management Committee Meetings and Resolutions (KG-UDWHP)", ("mcr-mcm", "mcm-mcr")),
    Family("MINUTES_RESOLUTIONS", "Minutes & Resolutions (KGD6 / NEC25)", ("minutesnresolutions", "minutes & resolutions", "minutes and resolutions")),
    Family("MCM", "Management Committee Meeting / Minutes", ("mcm", "management committee meeting", "management committee minutes")),
    Family("MCR", "Management Committee Resolution", ("mcr", "management committee resolution")),
    Family("OCR", "Operating Committee Resolution", ("ocr", "operating committee resolution")),
    Family("MOM", "Minutes of Meeting", ("mom", "minutes of meeting")),
    Family("QPR", "Quarterly Performance Report", ("qpr", "quarterly performance report")),
    Family("WPB", "Work Programme and Budget", ("wpb", "workprogrammenbudget", "work programme and budget")),
    Family("BUDGET", "Budget", ("budget",)),
    Family("AUDITED_ACCOUNTS", "Audited accounts / statements", ("auditedaccounts", "audited accounts", "audited_statement", "audited statement")),
    Family("ASSIGNMENT", "Assignment / farm-in / farm-out", ("assignment", "farm-in", "farmin", "farm-out", "farmout")),
    Family("NOTIFICATION", "Government / MoPNG notification", ("notification", "gazette")),
    Family("AMENDMENT", "Contract amendment", ("amendment", "addendum")),
)
_FAMILY_BY_CODE = {f.code: f for f in FAMILIES}
_ALIASES = sorted(((a, f) for f in FAMILIES for a in f.aliases), key=lambda x: len(x[0]), reverse=True)
# Folder-type families win over words such as "amendment" in a filename.
_SECONDARY = {"ASSIGNMENT", "NOTIFICATION", "AMENDMENT", "BUDGET"}


def _hay(text: str) -> str:
    return " " + re.sub(r"[^a-z0-9&]+", " ", (text or "").lower()).strip() + " "


def family_for(*parts: str) -> Optional[Family]:
    """Family from folder names first, then from the file name."""
    segments = [seg for part in parts for seg in re.split(r"[\\/]", part or "") if seg]
    if any(s.lower() == "minutesnresolutions" for s in segments):
        return _FAMILY_BY_CODE["MINUTES_RESOLUTIONS"]
    if any(s.lower() in {"mcr-mcm", "mcm-mcr"} for s in segments):
        return _FAMILY_BY_CODE["MCR_MCM"]
    secondary: Optional[Family] = None
    for segment in segments:
        hay = _hay(segment)
        compact = re.sub(r"[^a-z0-9]", "", segment.lower())
        for alias, family in _ALIASES:
            if _hay(alias) in hay or compact == re.sub(r"[^a-z0-9]", "", alias):
                if family.code in _SECONDARY:
                    secondary = secondary or family
                    continue
                return family
    return secondary


def family_by_code(code: str) -> Optional[Family]:
    return _FAMILY_BY_CODE.get((code or "").strip().upper().replace(" ", "_"))


def family_codes() -> list[str]:
    return [f.code for f in FAMILIES]


# --------------------------------------------------------------- citations

_WORKSPACE = re.compile(r"aamirWS/data/ril_data/(?:JOA|JAO|PEL|PML|PSC|RSC)/?", re.IGNORECASE)


def clean_rel_path(rel_path: str) -> str:
    """Drop a local workspace dump segment so citations stay family/file."""
    path = rel_path.replace("\\", "/")
    return _WORKSPACE.sub("", path).strip("/")


def citation_path(rel_path: str) -> str:
    """Label used in References: folder path + file name, workspace dump removed.

    Sites files are labelled from the `sites` folder down.
    """
    path = clean_rel_path(rel_path)
    parts = PurePosixPath(path).parts
    lowered = [p.lower() for p in parts]
    if "sites" in lowered:
        return "/".join(parts[lowered.index("sites"):])
    return path


def _encode(path: str) -> str:
    # Spaces -> %20 (and #, %, ? escaped); characters SharePoint paths use stay as they are.
    return "/".join(quote(part, safe="-_.~()!$&'*+,;=:@") for part in path.split("/"))


def document_link(rel_path: str) -> str:
    """Document Center link for a corpus file. Stored in documents.link.

    Same rules the old query prompt (prompts.py, "References") gave the model:
      - drop the workspace segment aamirWS/data/ril_data/<FAMILY>/
      - folder_path starts with sites/ -> https://enp1.ril.com/ + path from sites
        down to the file (Document Center only if it is already in the path)
      - anything else (JOA, JAO, PEL, PML, PSC, RSC, ...) ->
        https://enp1.ril.com/sites/Regulatory/Document%20Center/ + full folder
        hierarchy + "/" + filename
      - spaces encoded as %20, no folder flattened or invented
    Base URLs come from SITES_BASE_URL / DOC_CENTER_BASE_URL. With both empty,
    the local file served by this app at /files/.
    """
    path = citation_path(rel_path)
    if SITES_BASE_URL and path.lower().startswith("sites/"):
        return SITES_BASE_URL.rstrip("/") + "/" + _encode(path)
    if DOC_CENTER_BASE_URL:
        return DOC_CENTER_BASE_URL.rstrip("/") + "/" + _encode(path)
    return local_url(rel_path)


def citation_url(rel_path: str, page: Optional[int] = None, text: str = "") -> str:
    """Document link that opens at `page` and searches for `text` (see page_link)."""
    return page_link(document_link(rel_path), page, text)


def local_url(rel_path: str, page: Optional[int] = None) -> str:
    return page_link("/files/" + _encode(rel_path.replace("\\", "/")), page)


_SNIPPET_WORDS = 8


def search_snippet(text: str, words: int = _SNIPPET_WORDS) -> str:
    """First few words of a passage, cleaned for a PDF viewer's text search.

    Short on purpose: a long phrase is more likely to cross a line break or a
    hyphenation the viewer reads differently, and then nothing is found.
    """
    # Words are kept as printed ("(200)", "10.7", "per cent."): the viewer searches the
    # PDF's own text. Only quote marks, markdown emphasis and [C…] tags are dropped.
    cleaned = re.sub(r"\[\s*C\d+[^\]]*\]", " ", text or "")
    cleaned = re.sub(r"[“”\"*]", " ", cleaned)
    tokens = [t for t in cleaned.split() if not re.fullmatch(r"[\W_]+", t)]
    picked: list[str] = []
    for token in tokens:
        # A word split by end-of-line hyphenation would not match the rendered text.
        if token.endswith("-") and len(token) > 1:
            break
        picked.append(token)
        if len(picked) >= words:
            break
    if picked:
        picked[-1] = picked[-1].rstrip(",;:")
    return " ".join(picked).strip()


def page_link(link: str, page: Optional[int] = None, text: str = "") -> str:
    """Add PDF open parameters to a document link.

    #page=N              opens the page: Chrome, Edge, Firefox, Acrobat
    &search=...&phrase=  finds and highlights the words: Firefox (PDF.js), Acrobat;
                         other viewers ignore it and still open the page
    Any fragment already on the link is replaced.
    """
    base = (link or "").split("#", 1)[0]
    params: list[str] = []
    if page:
        params.append(f"page={int(page)}")
    snippet = search_snippet(text) if text else ""
    if snippet:
        params.append("search=" + quote(snippet, safe=""))
        params.append("phrase=true")
    return base + ("#" + "&".join(params) if params else "")


def squash(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def identifier_keys(text: str) -> set[str]:
    """Squashed forms of hyphen/slash/space-joined codes: KG-D6 -> kgd6."""
    keys: set[str] = set()
    for match in re.finditer(r"[A-Za-z0-9]+(?:(?:[-/]|\s(?=[A-Za-z]*\d))[A-Za-z0-9]+)+", text or ""):
        token = match.group(0)
        if any(ch.isdigit() for ch in token) and any(ch.isalpha() for ch in token):
            key = squash(token)
            if 3 <= len(key) <= 24:
                keys.add(key)
    for match in re.finditer(r"\b[A-Za-z]{1,6}\d{1,4}[A-Za-z]?\b", text or ""):
        keys.add(match.group(0).lower())
    return keys


def unique(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        item = (item or "").strip()
        key = item.lower()
        if item and key not in seen:
            seen.add(key)
            out.append(item)
    return out
