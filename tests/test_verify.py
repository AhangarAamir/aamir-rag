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
    assert "## References" in text and hit["url"] in text
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
