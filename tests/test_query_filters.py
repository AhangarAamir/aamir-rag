"""Query-time Chroma filters and per-run scratchpad."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from legal_chunking import build_chroma_filters
from query_agent import SharedCorpusKnowledgeTools, scratch_for_run


class BuildChromaFiltersTests(unittest.TestCase):
    def test_empty_is_none(self):
        self.assertIsNone(build_chroma_filters())

    def test_single_key_stays_plain(self):
        self.assertEqual(build_chroma_filters(doc_family="PSC"), {"doc_family": "PSC"})
        self.assertEqual(build_chroma_filters(article="21"), {"article": "21"})

    def test_two_keys_use_and(self):
        self.assertEqual(
            build_chroma_filters(doc_family="PSC", article="21"),
            {
                "$and": [
                    {"doc_family": {"$eq": "PSC"}},
                    {"article": {"$eq": "21"}},
                ]
            },
        )

    def test_clause_drops_article(self):
        filters = build_chroma_filters(
            doc_family="PSC",
            clause_id="21.5.5",
            article="21",
        )
        self.assertEqual(
            filters,
            {
                "$and": [
                    {"doc_family": {"$eq": "PSC"}},
                    {"clause_id": {"$eq": "21.5.5"}},
                ]
            },
        )
        self.assertNotIn("article", str(filters))

    def test_top_level_has_one_operator(self):
        filters = build_chroma_filters(doc_family="PSC", article="21")
        assert filters is not None
        self.assertEqual(len(filters), 1)
        self.assertIn("$and", filters)


class ScratchForRunTests(unittest.TestCase):
    def test_new_run_clears_prior_thoughts(self):
        first = SimpleNamespace(run_id="run-1", session_state={"thoughts": ["10.7"], "analysis": ["old"]})
        state = scratch_for_run(first)
        self.assertEqual(state["thoughts"], [])
        self.assertEqual(state["analysis"], [])
        state["thoughts"].append("21.5 this turn")

        second = SimpleNamespace(run_id="run-2", session_state=state)
        next_state = scratch_for_run(second)
        self.assertEqual(next_state["thoughts"], [])
        self.assertEqual(next_state["_scratch_run_id"], "run-2")

    def test_same_run_keeps_notes(self):
        ctx = SimpleNamespace(run_id="run-1", session_state=None)
        state = scratch_for_run(ctx)
        state["thoughts"].append("first")
        again = scratch_for_run(ctx)
        self.assertEqual(again["thoughts"], ["first"])


class FakeKnowledge:
    def __init__(self):
        self.calls = []

    def search(self, query, max_results=None, filters=None, user_id=None):
        self.calls.append({"query": query, "filters": filters})
        return []


class SearchQueryTests(unittest.TestCase):
    def test_does_not_prepend_clause_or_instrument(self):
        knowledge = FakeKnowledge()
        tools = SharedCorpusKnowledgeTools(
            knowledge=knowledge,
            enable_think=False,
            enable_search=True,
            enable_analyze=False,
            add_instructions=False,
        )
        ctx = SimpleNamespace(run_id="r1", session_state={})
        tools.search_knowledge(
            ctx,
            query="relinquishment of right of development",
            instrument_name="KGD6 PSC",
            doc_family="PSC",
            clause_id="21.5",
            article="21",
        )
        self.assertTrue(knowledge.calls)
        self.assertEqual(
            knowledge.calls[0]["query"],
            "relinquishment of right of development",
        )
        self.assertIn("$and", knowledge.calls[0]["filters"])


if __name__ == "__main__":
    unittest.main()
