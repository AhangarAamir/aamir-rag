"""Map typed names (KGD6 PSC) to instrument_key / filename for query filters."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

from config import ALIASES_DB_FILE

_CLAUSE_LABELED = re.compile(
    r"\b(?:article|clause|section|art\.?)\s+(\d+(?:\.\d+)*)\b",
    re.IGNORECASE,
)
_CLAUSE_BARE = re.compile(r"\b(\d+\.\d+(?:\.\d+)*)\b")


def normalize_alias(text: str) -> str:
    cleaned = text.strip().lower()
    cleaned = cleaned.replace("_", " ").replace("-", " ")
    cleaned = re.sub(r"[^a-z0-9. ]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def slugify(text: str) -> str:
    cleaned = normalize_alias(text).replace(".", " ")
    return re.sub(r"\s+", "_", cleaned).strip("_")


def extract_clause(text: str) -> tuple[Optional[str], Optional[str]]:
    """Return (article, clause) parsed from a question, if any."""
    labeled = _CLAUSE_LABELED.search(text)
    if labeled:
        number = labeled.group(1)
        article = number.split(".", 1)[0]
        clause = number if "." in number else None
        return article, clause
    bare = _CLAUSE_BARE.search(text)
    if bare:
        number = bare.group(1)
        return number.split(".", 1)[0], number
    return None, None


def _connect() -> sqlite3.Connection:
    ALIASES_DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(ALIASES_DB_FILE))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS aliases (
            alias TEXT PRIMARY KEY,
            instrument_key TEXT NOT NULL,
            filename TEXT NOT NULL,
            document_title TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def reset_aliases() -> None:
    conn = _connect()
    try:
        conn.execute("DELETE FROM aliases")
        conn.commit()
    finally:
        conn.close()


GENERIC_ALIASES = {
    "joa",
    "psc",
    "pml",
    "jao",
    "sample",
    "sample document",
    "sample psc",
    "sample joa",
    "agreement",
    "contract",
    "document",
    "letter",
    "minutes",
    "notification",
    "playbook",
    "amendment",
}


def register_document(
    instrument_key: str,
    filename: str,
    document_title: str,
    aliases: Iterable[str],
) -> None:
    key = (instrument_key or slugify(document_title or filename)).strip()
    if not key:
        return
    names = {normalize_alias(item) for item in aliases if item}
    names.update(
        {
            normalize_alias(document_title),
            normalize_alias(filename),
            normalize_alias(Path(filename).stem),
            normalize_alias(key.replace("_", " ")),
            normalize_alias(key),
        }
    )
    names = {
        alias
        for alias in names
        if alias and alias not in GENERIC_ALIASES and len(alias) >= 4
    }
    names.discard("")
    conn = _connect()
    try:
        for alias in names:
            conn.execute(
                """
                INSERT INTO aliases (alias, instrument_key, filename, document_title)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(alias) DO UPDATE SET
                    instrument_key = excluded.instrument_key,
                    filename = excluded.filename,
                    document_title = excluded.document_title
                """,
                (alias, key, filename, document_title or filename),
            )
        conn.commit()
    finally:
        conn.close()


def lookup_alias(text: str) -> Optional[tuple[str, str, str]]:
    """Return (instrument_key, filename, title) for an exact normalized alias."""
    needle = normalize_alias(text)
    if not needle:
        return None
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT instrument_key, filename, document_title FROM aliases WHERE alias = ?",
            (needle,),
        ).fetchone()
        if row:
            return str(row[0]), str(row[1]), str(row[2])
        slug = slugify(text)
        row = conn.execute(
            "SELECT instrument_key, filename, document_title FROM aliases WHERE instrument_key = ?",
            (slug,),
        ).fetchone()
        if row:
            return str(row[0]), str(row[1]), str(row[2])
        return None
    finally:
        conn.close()


def find_alias_in_text(text: str) -> Optional[tuple[str, str, str, str]]:
    """Longest catalog alias that appears in text. Returns (alias, key, filename, title)."""
    blob = normalize_alias(text)
    if not blob:
        return None
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT alias, instrument_key, filename, document_title FROM aliases"
        ).fetchall()
    finally:
        conn.close()
    best = None
    best_len = 0
    for alias, key, filename, title in rows:
        token = str(alias)
        if token and token in blob and len(token) > best_len:
            best = (token, str(key), str(filename), str(title))
            best_len = len(token)
    return best


def aliases_for_key(instrument_key: str) -> List[str]:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT alias FROM aliases WHERE instrument_key = ?",
            (instrument_key,),
        ).fetchall()
        return [str(row[0]) for row in rows]
    finally:
        conn.close()


@dataclass
class SearchScope:
    document: Optional[str] = None
    instrument_key: Optional[str] = None
    filename: Optional[str] = None
    document_title: Optional[str] = None
    aliases: List[str] = field(default_factory=list)
    article: Optional[str] = None
    clause: Optional[str] = None
    named: bool = False
    resolved: bool = False


def resolve_search_scope(
    query: str,
    document: Optional[str] = None,
    article: Optional[str] = None,
    clause: Optional[str] = None,
) -> SearchScope:
    parsed_article, parsed_clause = extract_clause(query)
    article = (article or parsed_article or "").strip() or None
    clause = (clause or parsed_clause or "").strip() or None
    if clause and not article:
        article = clause.split(".", 1)[0]

    named_text = (document or "").strip() or None
    match = lookup_alias(named_text) if named_text else None
    if match is None and not named_text:
        found = find_alias_in_text(query)
        if found:
            named_text = found[0]
            match = (found[1], found[2], found[3])

    scope = SearchScope(
        document=named_text,
        article=article,
        clause=clause,
        named=bool(named_text),
    )
    if match:
        scope.instrument_key, scope.filename, scope.document_title = match
        scope.aliases = aliases_for_key(scope.instrument_key)
        scope.resolved = True
    elif named_text:
        scope.instrument_key = slugify(named_text)
        scope.aliases = [normalize_alias(named_text)]
    return scope


def chroma_filters(scope: SearchScope) -> dict:
    """Chroma where-clause. Multiple fields must be wrapped in $and."""
    clauses: list[dict] = []
    if scope.instrument_key:
        clauses.append({"instrument_key": {"$eq": scope.instrument_key}})
    if scope.clause:
        clauses.append({"clause": {"$eq": scope.clause}})
    elif scope.article:
        clauses.append({"article": {"$eq": scope.article}})
    if not clauses:
        return {}
    if len(clauses) == 1:
        field, spec = next(iter(clauses[0].items()))
        return {field: spec["$eq"]}
    return {"$and": clauses}
