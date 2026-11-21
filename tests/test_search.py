from legal_rag.index import search as engine


def test_named_clause_stays_in_named_document(indexed_corpus):
    psc = engine.get_clause("10.7", "KGD6 PSC")["chunks"]
    joa = engine.get_clause("10.7", "NEC-OSN-97/2 JOA")["chunks"]
    assert psc and all("KG-DWN-98_3" in c["filename"] for c in psc)
    assert "two hundred (200) days" in " ".join(c["text"] for c in psc)
    assert joa and "Sole Risk" in joa[0]["text"]


def test_unknown_document_is_reported_not_substituted(indexed_corpus):
    result = engine.search("Article 10.7", document="Mahanadi PSC 2009")
    assert result["hits"] == [] and "No indexed document" in result["note"]


def test_definition_lookup_tolerates_typo(indexed_corpus):
    result = engine.find_definition("Reserviour", "KG-D6 PSC")
    assert result["chunks"][0]["defined_term"] == "Reservoir"


def test_glossary_expansion_finds_mopng(indexed_corpus):
    hits = engine.search("moping approval assignment")["hits"]
    assert "DGH" in hits[0]["filename"]


def test_clause_number_ranks_first(indexed_corpus):
    hits = engine.search("what does 21.5 say")["hits"]
    assert hits[0]["clause_id"] == "21.5"


def test_citation_links(indexed_corpus):
    hit = engine.get_clause("10.7", "KGD6 PSC")["chunks"][0]
    link = "https://enp1.ril.com/sites/Regulatory/Document%20Center/PSC/Domestic/KG-DWN-98_3%20PSC.pdf"
    assert hit["link"] == link
    assert hit["url"] == f"{link}#page={hit['page_start']}"
    assert hit["cite_path"] == "PSC/Domestic/KG-DWN-98_3 PSC.pdf"
    assert hit["local_url"].startswith("/files/PSC/Domestic/KG-DWN-98_3%20PSC.pdf#page=")
