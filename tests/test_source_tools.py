from types import SimpleNamespace

import pytest

from legal_rag.agents.source_tools import SourceFileTools
from legal_rag.agents.verify import verify_answer
from legal_rag.config import CORPUS_DIR
from legal_rag.index import search as engine
from legal_rag.ingest.pipeline import ingest_path
from legal_rag.ingest.structure import parse_markdown

MD = """# Constitution notes

Intro text that explains what this document covers in some detail here.

# PART III — FUNDAMENTAL RIGHTS

## 14. Equality before the law

The State shall not deny to any person equality before the law or the equal protection of the laws.

### 14.1 Equal protection

Reasonable classification is permitted where it rests on an intelligible differentia.

## 21A. Right to education

The State shall provide free and compulsory education to all children of the age of six to fourteen years.
"""


@pytest.fixture(scope="module")
def md_doc(indexed_corpus):
    path = CORPUS_DIR / "notes" / "rights.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(MD, encoding="utf-8")
    ingest_path("notes", use_llm=False)
    return path


def test_markdown_headings_become_articles_and_clauses():
    units = parse_markdown([MD]).units
    got = {(u.article, u.clause_id) for u in units}
    assert ("14", "") in got and ("14", "14.1") in got and ("21A", "") in got
    sub = next(u for u in units if u.clause_id == "14.1")
    assert sub.breadcrumb.endswith("14. Equality before the law › 14.1 Equal protection")


def test_get_clause_on_markdown(md_doc):
    chunks = engine.get_clause("article 21a", "rights")["chunks"]
    assert chunks and "free and compulsory education" in chunks[0]["text"]


def test_read_markdown_section_live(md_doc):
    out = SourceFileTools().read_markdown_section("rights", "14")
    assert "equality before the law" in out and "14.1 Equal protection" in out
    assert "[C" in out  # citable


def test_read_pdf_pages_live_with_layout(indexed_corpus):
    out = SourceFileTools().read_source_pages("KGD6 PSC", 10, keep_layout=True)
    assert "Government 40%" in out and "--- page 10 ---" in out
    assert "Cite with the indexed passages" in out


def test_grep_source_finds_phrase_across_files(indexed_corpus):
    out = SourceFileTools().grep_source("deed of adherence")
    assert "DGH approval assignment 2018.pdf" in out and "KG-DWN-98_3 PSC.pdf" in out


def test_grep_reports_unknown_document(indexed_corpus):
    assert "No file in the corpus matches" in SourceFileTools().grep_source("royalty", document="Mahanadi 2009")


def test_quotes_from_live_reads_count_as_verified(indexed_corpus):
    live = SourceFileTools().read_source_pages("KGD6 PSC", 10, keep_layout=True)
    tag = engine.get_clause("10.7", "KGD6 PSC")["chunks"][0]["id"]
    run = SimpleNamespace(tools=[SimpleNamespace(result=live), SimpleNamespace(result=f"[C{tag}] x")])
    _, report = verify_answer(f'At that band the split is "IM 2.5 or more but less than 3.5" [C{tag}].', run)
    assert report["unverified_quotes"] == []
