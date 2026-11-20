"""OpenAI models: the agents' chat model, embeddings, and one-shot calls.

Chat runs on the Responses API (agno OpenAIResponses). Reasoning models
(gpt-5 / gpt-6 class) reject function tools with reasoning on
/v1/chat/completions, and Responses keeps their reasoning between tool calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from agno.models.openai.responses import OpenAIResponses
from openai import OpenAI

from legal_rag.config import (
    EMBED_BATCH_SIZE,
    EMBED_ENABLED,
    EMBEDDING_MODEL,
    INGEST_MODEL,
    OPENAI_API_BASE,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_REASONING_EFFORT,
)


@dataclass(frozen=True)
class EmbeddingSettings:
    enabled: bool = EMBED_ENABLED
    model: str = EMBEDDING_MODEL
    batch_size: int = EMBED_BATCH_SIZE


def embedding_settings() -> EmbeddingSettings:
    return EmbeddingSettings()


class LegalResponses(OpenAIResponses):
    """Also treats Luna-named and o-series ids as reasoning models, so their
    reasoning items are carried across tool calls."""

    def _using_reasoning_model(self) -> bool:
        name = (self.id or "").lower()
        return super()._using_reasoning_model() or "luna" in name or name.startswith(("o1", "o3", "o4"))


def build_chat_model(model_id: Optional[str] = None) -> OpenAIResponses:
    kwargs: dict[str, Any] = {
        "id": model_id or OPENAI_MODEL,
        "base_url": OPENAI_API_BASE,
        "api_key": OPENAI_API_KEY or None,
        "timeout": 300.0,
        "max_retries": 2,
    }
    if OPENAI_REASONING_EFFORT:
        kwargs["reasoning_effort"] = OPENAI_REASONING_EFFORT
    return LegalResponses(**kwargs)


def client(timeout: float = 30.0) -> OpenAI:
    return OpenAI(base_url=OPENAI_API_BASE, api_key=OPENAI_API_KEY or None, timeout=timeout, max_retries=2)


def complete(
    prompt: str,
    *,
    model: Optional[str] = None,
    json_mode: bool = False,
    max_tokens: Optional[int] = None,
    timeout: float = 120.0,
) -> tuple[str, str]:
    """One direct Responses call (no agent). Returns (text, model id)."""
    model_id = model or INGEST_MODEL
    kwargs: dict[str, Any] = {"model": model_id, "input": [{"role": "user", "content": prompt}]}
    if OPENAI_REASONING_EFFORT:
        kwargs["reasoning"] = {"effort": OPENAI_REASONING_EFFORT}
    if max_tokens:
        kwargs["max_output_tokens"] = max_tokens
    if json_mode:
        kwargs["text"] = {"format": {"type": "json_object"}}
    response = client(timeout=timeout).responses.create(**kwargs)
    return (response.output_text or "").strip(), response.model or model_id
