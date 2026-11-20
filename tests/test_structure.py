from legal_rag.ingest.extract import clean_pages
from legal_rag.ingest.structure import parse


def units(text: str):
    return parse([text]).units


def test_articles_clauses_and_definitions():
    text = (
        "ARTICLE 1\nDEFINITIONS\n"
        '1.1 "Discovery" means a find of Petroleum.\n'
        '1.2 "Discovery Area" means part of the Contract Area.\n'
        "ARTICLE 10 - DEVELOPMENT\n"
        "10.1 The Contractor shall notify.\n"
        "10.7 Development Plan. The Contractor shall submit a plan.\n"
    )
    got = [(u.kind, u.article, u.clause_id, u.defined_term or u.heading) for u in units(text)]
    assert ("definition", "1", "1.1", "Discovery") in got
    assert ("definition", "1", "1.2", "Discovery Area") in got
    assert ("clause", "10", "10.7", "Development Plan") in got


def test_wrapped_references_are_not_clauses():
    text = (
        "ARTICLE 10 - DEVELOPMENT\n"
        "10.2 The Contractor shall submit particulars. The obligations in Article\n"
        "10.2 above shall apply to every Discovery.\n"
        "10.5 The royalty shall be at the rate of\n"
        "12.5 per cent of the well-head value.\n"
        "10.6 Any notice under\n"
        "Article 21.\n"
        "10.7 Development Plan. Submit within 200 days.\n"
    )
    clauses = [u.clause_id for u in units(text) if u.kind == "clause"]
    assert clauses == ["10.2", "10.5", "10.6", "10.7"]


def test_ocr_article_numbers_and_roman():
    got = {u.article for u in units("ARTICLE IO\nLEASE\n10.1 The lease.\nARTICLE XII\nTERM\n12.1 The term.\n")}
    assert {"10", "12"} <= got


def test_article_headings_survive_running_head_removal():
    pages = [f"Running Head\nARTICLE {n}\nTITLE {n}\n{n}.1 The text {n}.\nPage {n} of 5" for n in range(1, 6)]
    cleaned = clean_pages(pages)
    assert all(p.startswith("ARTICLE") for p in cleaned)
    assert all("Page" not in p and "Running Head" not in p for p in cleaned)


def test_long_clause_is_split_into_parts():
    body = " ".join(["The Contractor shall do things."] * 200)
    us = parse([f"ARTICLE 5 - WORK\n5.1 {body}\n"], max_chars=1000).units
    parts = [u for u in us if u.clause_id == "5.1"]
    assert len(parts) > 1 and all(len(u.text) <= 1100 for u in parts)
    assert parts[0].parts == len(parts)
