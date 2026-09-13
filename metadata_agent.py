"""LLM uniqueness cards for chunks, using file identity plus chunk text."""

from __future__ import annotations

import re
from typing import List

from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from pydantic import BaseModel, Field

from config import DOCUMENT_CARD_CHARS, METADATA_BATCH_SIZE, OPENAI_MODEL
from folder_tags import chroma_safe
from prompts import METADATA_AGENT_INSTRUCTIONS

_document_agent: Agent | None = None
_chunk_agent: Agent | None = None


class DocumentCard(BaseModel):
    document_title: str = ""
    aliases: List[str] = Field(default_factory=list)
    instrument_key: str = ""
    instrument_type: str = "other"
    block_or_contract_area: str = ""
    parties: List[str] = Field(default_factory=list)
    document_status: str = "unknown"
    toc_outline: str = ""


class ChunkUniquenessCard(BaseModel):
    chunk_index: int
    article: str = "unknown"
    clause: str = "unknown"
    clause_aliases: List[str] = Field(default_factory=list)
    heading: str = ""
    locator_path: str = ""
    content_type: str = "other"
    defined_terms: List[str] = Field(default_factory=list)
    unique_entities: List[str] = Field(default_factory=list)
    distinguisher: str = ""
    ocr_uncertainty: bool = False


class ChunkCardBatch(BaseModel):
    cards: List[ChunkUniquenessCard] = Field(default_factory=list)


def _document_agent_instance() -> Agent:
    global _document_agent
    if _document_agent is None:
        _document_agent = Agent(
            id="document-card-agent",
            name="Document Card Agent",
            model=OpenAIResponses(id=OPENAI_MODEL),
            instructions=METADATA_AGENT_INSTRUCTIONS,
            output_schema=DocumentCard,
            markdown=False,
        )
    return _document_agent


def _chunk_agent_instance() -> Agent:
    global _chunk_agent
    if _chunk_agent is None:
        _chunk_agent = Agent(
            id="chunk-uniqueness-agent",
            name="Chunk Uniqueness Agent",
            model=OpenAIResponses(id=OPENAI_MODEL),
            instructions=METADATA_AGENT_INSTRUCTIONS,
            output_schema=ChunkCardBatch,
            markdown=False,
        )
    return _chunk_agent


def fallback_document_card(filename: str, folder_tags: dict[str, str]) -> DocumentCard:
    stem = filename.rsplit(".", 1)[0] if filename else "unknown"
    family = folder_tags.get("folder_family", "")
    instrument_type = family.lower() if family else "other"
    aliases = [stem, stem.replace("_", " ")]
    if family:
        aliases.append(f"{stem} {family}")
    return DocumentCard(
        document_title=stem.replace("_", " "),
        aliases=aliases,
        instrument_key=_slug(stem),
        instrument_type=instrument_type or "other",
        block_or_contract_area="",
        document_status="unknown",
    )


def _slug(text: str) -> str:
    from alias_catalog import slugify

    return slugify(text)


def extract_document_card(
    filename: str,
    folder_tags: dict[str, str],
    full_text: str,
) -> DocumentCard:
    sample = full_text[:DOCUMENT_CARD_CHARS] if full_text else ""
    prompt = (
        "Build the document card for this legal file. Use only visible text.\n\n"
        f"Filename: {filename}\n"
        f"Folder tags: {folder_tags}\n\n"
        f"File text (may be truncated):\n{sample}"
    )
    try:
        result = _document_agent_instance().run(prompt)
        card = result.content
        if isinstance(card, DocumentCard):
            if not card.instrument_key:
                card.instrument_key = _slug(card.document_title or filename)
            if filename and filename not in card.aliases:
                card.aliases.append(filename)
            return card
    except Exception:
        pass
    return fallback_document_card(filename, folder_tags)


def extract_chunk_cards(
    filename: str,
    folder_tags: dict[str, str],
    document_card: DocumentCard,
    chunks: List[str],
    start_index: int,
) -> List[ChunkUniquenessCard]:
    if not chunks:
        return []
    pieces = []
    for offset, text in enumerate(chunks):
        index = start_index + offset
        prev_text = chunks[offset - 1][:400] if offset > 0 else ""
        next_text = chunks[offset + 1][:400] if offset + 1 < len(chunks) else ""
        pieces.append(
            f"--- chunk_index={index} ---\n"
            f"PREV: {prev_text}\n"
            f"CHUNK:\n{text[:3500]}\n"
            f"NEXT: {next_text}\n"
        )
    prompt = (
        "Return one uniqueness card per chunk. chunk_index must match the labels.\n\n"
        f"Filename: {filename}\n"
        f"Folder tags: {folder_tags}\n"
        f"Document card: {document_card.model_dump()}\n\n"
        + "\n".join(pieces)
    )
    try:
        result = _chunk_agent_instance().run(prompt)
        batch = result.content
        if isinstance(batch, ChunkCardBatch) and batch.cards:
            by_index = {card.chunk_index: card for card in batch.cards}
            cards: List[ChunkUniquenessCard] = []
            for offset in range(len(chunks)):
                index = start_index + offset
                cards.append(by_index.get(index) or _fallback_chunk_card(index, document_card))
            return cards
    except Exception:
        pass
    return [
        _fallback_chunk_card(start_index + offset, document_card) for offset in range(len(chunks))
    ]


def _fallback_chunk_card(index: int, document_card: DocumentCard) -> ChunkUniquenessCard:
    title = document_card.document_title or document_card.instrument_key
    return ChunkUniquenessCard(
        chunk_index=index,
        locator_path=f"{title} > chunk {index}",
        distinguisher=f"From {title}",
        unique_entities=[title] if title else [],
    )


def enrich_chunks(
    filename: str,
    folder_tags: dict[str, str],
    full_text: str,
    chunk_texts: List[str],
) -> tuple[DocumentCard, List[ChunkUniquenessCard]]:
    document_card = extract_document_card(filename, folder_tags, full_text)
    cards: List[ChunkUniquenessCard] = []
    batch = max(1, METADATA_BATCH_SIZE)
    for start in range(0, len(chunk_texts), batch):
        window = chunk_texts[start : start + batch]
        cards.extend(
            extract_chunk_cards(filename, folder_tags, document_card, window, start)
        )
    return document_card, cards


def build_search_prefix(
    document_card: DocumentCard,
    chunk_card: ChunkUniquenessCard,
    folder_tags: dict[str, str],
) -> str:
    aliases = " | ".join(
        item for item in ([document_card.document_title, *document_card.aliases]) if item
    )
    family = folder_tags.get("folder_family") or document_card.instrument_type
    also = " | ".join(chunk_card.clause_aliases)
    lines = [
        f"[Document: {document_card.document_title or folder_tags.get('filename', '')} | aliases: {aliases}]",
        f"[Family: {family} | block: {document_card.block_or_contract_area}]",
        f"[Locator: {chunk_card.locator_path} | heading: {chunk_card.heading}]",
        f"[Type: {chunk_card.content_type}]",
    ]
    if also:
        lines.append(f"[Also known as: {also}]")
    if chunk_card.distinguisher:
        lines.append(f"[Distinct: {chunk_card.distinguisher}]")
    return "\n".join(lines) + "\n"


def _primary_number(text: str) -> str:
    if not text or text.strip().lower() == "unknown":
        return "unknown"
    match = re.search(r"\d+(?:\.\d+)*", text)
    return match.group(0) if match else "unknown"


def chunk_metadata(
    document_card: DocumentCard,
    chunk_card: ChunkUniquenessCard,
    folder_tags: dict[str, str],
) -> dict[str, str | int | float | bool]:
    article = _primary_number(chunk_card.article)
    clause = _primary_number(chunk_card.clause)
    aliases = list(chunk_card.clause_aliases)
    if chunk_card.clause and chunk_card.clause not in aliases:
        aliases.append(chunk_card.clause)
    payload = {
        **folder_tags,
        "document_title": document_card.document_title,
        "aliases": document_card.aliases,
        "instrument_key": document_card.instrument_key,
        "instrument_type": document_card.instrument_type,
        "block_or_contract_area": document_card.block_or_contract_area,
        "parties": document_card.parties,
        "document_status": document_card.document_status,
        "article": article,
        "clause": clause,
        "clause_aliases": aliases,
        "heading": chunk_card.heading,
        "locator_path": chunk_card.locator_path,
        "content_type": chunk_card.content_type,
        "defined_terms": chunk_card.defined_terms,
        "unique_entities": chunk_card.unique_entities,
        "distinguisher": chunk_card.distinguisher,
        "ocr_uncertainty": chunk_card.ocr_uncertainty,
    }
    return chroma_safe(payload)
