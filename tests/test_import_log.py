"""One .log file per imported document, named from that document."""

from __future__ import annotations

from datetime import datetime

from legal_rag.knowledge.import_log import document_log_name, write_document_log


def test_log_name_is_document_name_plus_log():
    assert document_log_name("budget.0.1.pdf") == "budget.0.1.pdf.log"
    assert document_log_name("JOA/New JOA/a:b.pdf") == "a_b.pdf.log"
    assert document_log_name("") == "document.log"


def test_second_log_for_same_name_keeps_both(tmp_path):
    when = datetime(2026, 11, 20, 10, 0, 0)
    first = write_document_log(file_name="a.pdf", source_path="JOA/a.pdf", status="added", finished_at=when, directory=tmp_path)
    second = write_document_log(file_name="a.pdf", source_path="JOA/a.pdf", status="failed", message="boom",
                                finished_at=when, directory=tmp_path)
    assert first.name == "a.pdf.log"
    assert second.name == "a.pdf_2026-11-20_100000.log"
    assert "status: failed" in second.read_text() and "message: boom" in second.read_text()


def test_ingest_writes_one_log_per_document(indexed_corpus):
    from legal_rag.knowledge.import_log import IMPORT_LOG_DIR

    logged = {p.name for p in IMPORT_LOG_DIR.glob("*.log")}
    for rel in indexed_corpus.added:
        assert document_log_name(rel) in logged
