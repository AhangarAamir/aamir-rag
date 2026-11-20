"""Offline test setup: isolated data dir, no embeddings, no LLM calls."""

import os
import sys
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="legal-rag-test-"))
os.environ.update(
    DATA_DIR=str(_TMP / "data"),
    CORPUS_DIR=str(_TMP / "corpus"),
    EMBED_ENABLED="false",
    INGEST_LLM_PROFILE="false",
)
sys.path.insert(0, str(Path(__file__).parent / "fixtures"))

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def indexed_corpus():
    from build_corpus import build

    from legal_rag.config import CORPUS_DIR
    from legal_rag.ingest.pipeline import ingest_path

    build(CORPUS_DIR)
    report = ingest_path(None, use_llm=False)
    assert not report.failed, report.failed
    return report
