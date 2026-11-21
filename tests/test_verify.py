import re
from types import SimpleNamespace

from legal_rag.agents.verify import verify_answer
from legal_rag.index import search as engine


def run_with(ids):
    result = "\n".join(f"[C{i}] passage" for i in ids)
    return SimpleNamespace(tools=[SimpleNamespace(result=result)])


def test_references_built_and_invented_tags_flagged(indexed_corpus):
    hit = engine.get_clause("10.7", "KGD6 PSC")["chunks"][0]
    answer = f'The plan is due within "two hundred (200) days from the date of such declaration" [C{hit["id"]}][C99999][B3].'
    text, report = verify_answer(answer, run_with([hit["id"]]))
    refs = text.split("## References")[1]
    assert hit["link"] + "#page=" in refs and "&search=two%20hundred%20%28200%29%20days" in refs
    assert report["unknown"] == [99999]
    assert report["stray_tags"] == ["[B3]"] and "[B3]" not in text.split("## References")[0]
    assert report["unverified_quotes"] == []


def test_made_up_quote_is_flagged(indexed_corpus):
    hit = engine.get_clause("10.7", "KGD6 PSC")["chunks"][0]
    answer = f'It says "the Contractor must submit the plan within ninety days of discovery" [C{hit["id"]}].'
    text, report = verify_answer(answer, run_with([hit["id"]]))
    assert report["unverified_quotes"] and "Verification" in text


def test_social_reply_untouched():
    text, _ = verify_answer("Hello! Ask me about your contracts.", SimpleNamespace(tools=[]))
    assert text == "Hello! Ask me about your contracts."


def test_researched_answer_without_citations_is_flagged(indexed_corpus):
    text, _ = verify_answer("The royalty is 10%.", run_with([1]))
    assert "cites no passage" in text


def test_ellipsis_and_bracket_quotes_are_checked_piecewise(indexed_corpus):
    hit = engine.get_clause("21.5", "KGD6 PSC")["chunks"][0]
    answer = (
        'It says "The Contractor shall have the right to retain a Development Area... **subject to Article 10.7**" '
        f'and "the Contractor shall [then] relinquish the Discovery Area" [C{hit["id"]}].'
    )
    _, report = verify_answer(answer, run_with([hit["id"]]))
    assert report["unverified_quotes"] == []


def test_each_page_of_a_multi_page_passage_gets_its_own_reference(indexed_corpus):
    from legal_rag.index import store

    row = store.query("SELECT id, page_start, page_end FROM chunks WHERE page_end > page_start ORDER BY id LIMIT 1")[0]
    hit = engine.get_chunks([row["id"]])[0]
    text, _ = verify_answer(f"See this passage [C{hit['id']}].", run_with([hit["id"]]))
    refs = [line for line in text.split("## References")[1].splitlines() if line.startswith("- [")]
    pages = [int(re.search(r"#page=(\d+)", line).group(1)) for line in refs]
    assert pages == list(range(row["page_start"], row["page_end"] + 1))
    assert "(continued)" in refs[-1]


def test_same_document_cited_twice_gives_two_references_on_their_pages(indexed_corpus):
    a = engine.get_clause("10.7", "KGD6 PSC")["chunks"][0]
    b = engine.get_clause("21.5", "KGD6 PSC")["chunks"][0]
    text, _ = verify_answer(f"Due dates [C{a['id']}]. Retention [C{b['id']}].", run_with([a["id"], b["id"]]))
    refs = [line for line in text.split("## References")[1].splitlines() if line.startswith("- [")]
    assert len(refs) == 2
    assert f"#page={a['page_start']}&search=" in refs[0] and f"[C{a['id']}]" in refs[0]
    assert f"#page={b['page_start']}&search=" in refs[1] and f"[C{b['id']}]" in refs[1]


def test_quote_lands_on_the_page_that_contains_it(indexed_corpus):
    from legal_rag.agents.verify import reference_targets
    from legal_rag.index import store

    row = store.query("SELECT id FROM chunks WHERE page_end > page_start ORDER BY id LIMIT 1")[0]
    hit = engine.get_chunks([row["id"]])[0]
    last = store.query("SELECT text FROM pages WHERE doc_id=? AND page_no=?", (hit["doc_id"], hit["page_end"]))[0]["text"]
    line = next(l.strip() for l in last.splitlines() if len(l.strip()) > 30 and l.strip() in hit["text"])
    targets = reference_targets(hit, [line])
    assert [t["page"] for t in targets] == [hit["page_end"]]
