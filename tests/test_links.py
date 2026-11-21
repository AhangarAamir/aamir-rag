"""documents.link follows the old query prompt's References rules (its examples are the expected values)."""

import pytest

import sqlite3

from legal_rag.domain import citation_path, citation_url, document_link, page_link, search_snippet

DC = "https://enp1.ril.com/sites/Regulatory/Document%20Center/"


@pytest.mark.parametrize(
    "rel, label, url",
    [
        # Document Center families, nested folders kept, spaces -> %20
        ("JOA/NEW JOA/file.pdf", "JOA/NEW JOA/file.pdf", DC + "JOA/NEW%20JOA/file.pdf"),
        ("PEL/file.pdf", "PEL/file.pdf", DC + "PEL/file.pdf"),
        ("PML/OLD PML/file.pdf", "PML/OLD PML/file.pdf", DC + "PML/OLD%20PML/file.pdf"),
        ("RSC/file.pdf", "RSC/file.pdf", DC + "RSC/file.pdf"),
        # sites/ files: from sites down, no Document Center inserted
        (
            "sites/Regulatory/KGD6/MINUTESnRESOLUTIONS/OCR/Document Center/file.pdf",
            "sites/Regulatory/KGD6/MINUTESnRESOLUTIONS/OCR/Document Center/file.pdf",
            "https://enp1.ril.com/sites/Regulatory/KGD6/MINUTESnRESOLUTIONS/OCR/Document%20Center/file.pdf",
        ),
        (
            "sites/Regulatory/KGUDWHP/Document Center/KGUDWHP-2018-1/MCR-MCM/file.pdf",
            "sites/Regulatory/KGUDWHP/Document Center/KGUDWHP-2018-1/MCR-MCM/file.pdf",
            "https://enp1.ril.com/sites/Regulatory/KGUDWHP/Document%20Center/KGUDWHP-2018-1/MCR-MCM/file.pdf",
        ),
        # workspace dump segment removed
        ("RSC/aamirWS/data/ril_data/RSC/file.pdf", "RSC/file.pdf", DC + "RSC/file.pdf"),
        ("RSC/aamirWS/data/ril_data/RSC/Sub/file.pdf", "RSC/Sub/file.pdf", DC + "RSC/Sub/file.pdf"),
        # the file name is kept as it is: base URL + folder hierarchy + "/" + filename
        ("01_fundamental_rights.md", "01_fundamental_rights.md", DC + "01_fundamental_rights.md"),
        ("PSC/Notes on KG-D6.txt", "PSC/Notes on KG-D6.txt", DC + "PSC/Notes%20on%20KG-D6.txt"),
    ],
)
def test_aamir_rag_link_rules(rel, label, url):
    assert citation_path(rel) == label
    assert document_link(rel) == url
    assert citation_url(rel) == url
    assert "aamirWS" not in url and "data/ril_data" not in url


def test_documents_table_has_link_for_every_file(indexed_corpus):
    from legal_rag.index import store

    docs = store.list_documents()
    assert docs
    for doc in docs:
        assert doc["link"] == document_link(doc["rel_path"])
        assert doc["link"].startswith("https://enp1.ril.com/sites/")


def test_search_hits_and_references_use_the_stored_link(indexed_corpus):
    from legal_rag.index import search as engine
    from legal_rag.index import store

    hit = engine.search("participating interest", top_k=1)["hits"][0]
    doc = store.get_document(hit["doc_id"])
    assert hit["link"] == doc["link"]
    assert hit["url"] == f"{doc['link']}#page={hit['page_start']}"


def test_old_index_gets_link_column(tmp_path):
    from legal_rag.index import store

    old = tmp_path / "old.db"
    conn = sqlite3.connect(old)
    conn.execute("CREATE TABLE documents (id INTEGER PRIMARY KEY, rel_path TEXT UNIQUE NOT NULL, filename TEXT NOT NULL,"
                 " folder TEXT NOT NULL DEFAULT '', aliases TEXT NOT NULL DEFAULT '[]', parties TEXT NOT NULL DEFAULT '[]',"
                 " empty_pages TEXT NOT NULL DEFAULT '[]')")
    conn.execute("INSERT INTO documents(rel_path, filename) VALUES('JOA/NEW JOA/a b.pdf', 'a b.pdf')")
    conn.commit()
    conn.close()
    previous = store._db_path
    try:
        store.configure(old)
        assert store.query("SELECT link FROM documents")[0]["link"] == DC + "JOA/NEW%20JOA/a%20b.pdf"
    finally:
        store.configure(previous)


def test_page_link_opens_the_page_and_searches_the_text():
    link = DC + "JOA/NEW%20JOA/a.pdf"
    assert page_link(link, 12) == link + "#page=12"
    assert page_link(link + "#page=3", 7, "two hundred (200) days from the date of such declaration") == (
        link + "#page=7&search=two%20hundred%20%28200%29%20days%20from%20the%20date%20of&phrase=true"
    )
    assert page_link("/files/a.pdf", None) == "/files/a.pdf"


def test_search_snippet_keeps_printed_words():
    assert search_snippet("“10.7 Development Plan.” The Contractor shall submit [C12]") == (
        "10.7 Development Plan. The Contractor shall submit"
    )
    assert search_snippet("Block KG-DWN-98_3 per cent. of the devel-") == "Block KG-DWN-98_3 per cent. of the"
