"""Chunk labels run in batches, and embeddings are batched the same way."""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch

from agno.knowledge.document.base import Document

from legal_rag.knowledge.chunking import CHUNKER_CONCURRENCY, LegalAgenticChunking, LegalChunkDecision
from legal_rag.knowledge.store import build_embedder


class ChunkBatchTests(unittest.TestCase):
    def test_more_than_one_batch_does_not_overlap(self):
        chunker = LegalAgenticChunking(max_chunk_size=250)
        active: set[int] = set()
        max_active = 0
        crossed: list[set[int]] = []

        async def _alabel(**kwargs):
            nonlocal max_active
            role = kwargs.get("role")
            index = int(kwargs.get("chunk_index") or 0)
            if role == "label":
                active.add(index)
                max_active = max(max_active, len(active))
                if any(item <= CHUNKER_CONCURRENCY for item in active) and any(
                    item > CHUNKER_CONCURRENCY for item in active
                ):
                    crossed.append(set(active))
                await asyncio.sleep(0)
                active.discard(index)
            return LegalChunkDecision(instrument_name="Doc", split_at=1)

        chunker._alabel = _alabel  # type: ignore[method-assign]
        document = Document(name="budget.pdf", content="A" * 8000)
        with patch("legal_rag.knowledge.chunking.get_chunker_agent", return_value=object()):
            chunks = asyncio.run(chunker.achunk(document))

        self.assertGreater(len(chunks), CHUNKER_CONCURRENCY)
        self.assertLessEqual(max_active, CHUNKER_CONCURRENCY)
        self.assertEqual(crossed, [])

    def test_embedder_batches_requests(self):
        embedder = build_embedder()
        self.assertTrue(embedder.enable_batch)
        self.assertEqual(embedder.batch_size, CHUNKER_CONCURRENCY)
