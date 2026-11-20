"""OpenAI embeddings, plus an in-memory cosine index loaded from SQLite.

A corpus of tens of thousands of clause chunks fits in one numpy matrix, so
exact search is fast and needs no extra service.
"""

from __future__ import annotations

import threading
from typing import Optional, Sequence

import numpy as np
from agno.utils.log import log_warning

from legal_rag.index import store
from legal_rag.models import EmbeddingSettings, client, embedding_settings

MAX_EMBED_CHARS = 6000


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit]


def embed_texts(texts: Sequence[str], cfg: Optional[EmbeddingSettings] = None) -> np.ndarray:
    cfg = cfg or embedding_settings()
    api = client(timeout=300.0)
    out: list[list[float]] = []
    batch = max(1, cfg.batch_size)
    for start in range(0, len(texts), batch):
        chunk = [_clip(t, MAX_EMBED_CHARS) or " " for t in texts[start:start + batch]]
        try:
            response = api.embeddings.create(model=cfg.model, input=chunk)
        except Exception as exc:
            # Small embedding models have short context windows. Retry shorter.
            if "context" not in str(exc).lower() and "length" not in str(exc).lower():
                raise
            response = api.embeddings.create(model=cfg.model, input=[_clip(t, 1500) for t in chunk])
        out.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))
    return np.asarray(out, dtype=np.float32)


def to_blob(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


class VectorIndex:
    """Normalized embedding matrix for the configured model, reloaded when the index changes."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._version = -1
        self._model = ""
        self.ids = np.zeros(0, dtype=np.int64)
        self.doc_ids = np.zeros(0, dtype=np.int64)
        self.matrix = np.zeros((0, 0), dtype=np.float32)

    def _load(self, model: str) -> None:
        rows = store.query(
            "SELECT e.chunk_id, e.vec, e.dim, c.doc_id FROM embeddings e JOIN chunks c ON c.id = e.chunk_id"
            " WHERE e.model = ?",
            (model,),
        )
        if not rows:
            self.ids = np.zeros(0, dtype=np.int64)
            self.doc_ids = np.zeros(0, dtype=np.int64)
            self.matrix = np.zeros((0, 0), dtype=np.float32)
            return
        dim = rows[0]["dim"]
        rows = [r for r in rows if r["dim"] == dim]
        matrix = np.frombuffer(b"".join(r["vec"] for r in rows), dtype=np.float32).reshape(len(rows), dim)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.matrix = matrix / norms
        self.ids = np.asarray([r["chunk_id"] for r in rows], dtype=np.int64)
        self.doc_ids = np.asarray([r["doc_id"] for r in rows], dtype=np.int64)

    def ensure(self, model: str) -> None:
        version = store.index_version()
        with self._lock:
            if version != self._version or model != self._model:
                self._load(model)
                self._version = version
                self._model = model

    def search(self, query: str, top_n: int = 50, doc_ids: Optional[Sequence[int]] = None) -> list[tuple[int, float]]:
        cfg = embedding_settings()
        if not cfg.enabled:
            return []
        self.ensure(cfg.model)
        if self.matrix.size == 0:
            return []
        try:
            q = embed_texts([query], cfg)[0]
        except Exception as exc:
            log_warning(f"Query embedding failed, using keyword search only: {exc}")
            return []
        if q.shape[0] != self.matrix.shape[1]:
            log_warning("Embedding size differs from the index. Re-embed the library.")
            return []
        q = q / (np.linalg.norm(q) or 1.0)
        scores = self.matrix @ q
        if doc_ids:
            mask = np.isin(self.doc_ids, np.asarray(list(doc_ids), dtype=np.int64))
            scores = np.where(mask, scores, -np.inf)
        n = min(top_n, int(np.isfinite(scores).sum()))
        if n <= 0:
            return []
        top = np.argpartition(-scores, n - 1)[:n]
        top = top[np.argsort(-scores[top])]
        return [(int(self.ids[i]), float(scores[i])) for i in top]


VECTORS = VectorIndex()
