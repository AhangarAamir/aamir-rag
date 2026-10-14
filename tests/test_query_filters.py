"""Query-time Chroma filters and per-run scratchpad."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from legal_rag.knowledge.chunking import build_chroma_filters, strip_workspace_path
from agno.knowledge.document import Document

from legal_rag.agents.query import SharedCorpusKnowledgeTools, scratch_for_run, slim_hit


class WorkspacePathTests(unittest.TestCase):
    def test_strips_rsc_workspace_dump(self):
        raw = "RSC/aamirWS/data/ril_data/RSC/RSC GS_OSHP_2022_2_ONGC_RIL_BP.pdf"
        self.assertEqual(strip_workspace_path(raw), "RSC/RSC GS_OSHP_2022_2_ONGC_RIL_BP.pdf")

    def test_strips_workspace_from_document_center_url(self):
        raw = (
            "https://enp1.ril.com/sites/Regulatory/Document%20Center/"
            "RSC/aamirWS/data/ril_data/RSC/file.pdf"
        )
        self.assertEqual(
            strip_workspace_path(raw),
            "https://enp1.ril.com/sites/Regulatory/Document%20Center/RSC/file.pdf",
        )

    def test_leaves_real_psc_folders(self):
        raw = "PSC/ril_data/Domestic/NELP 1/KGD6-PSC_2000.04.12.pdf"
        self.assertEqual(strip_workspace_path(raw), raw)


class BuildChromaFiltersTests(unittest.TestCase):
    def test_empty_is_none(self):
        self.assertIsNone(build_chroma_filters())

    def test_single_key_stays_plain(self):
        self.assertEqual(build_chroma_filters(doc_family="PSC"), {"doc_family": "PSC"})
        self.assertEqual(build_chroma_filters(article="21"), {"article": "21"})
        self.assertEqual(build_chroma_filters(block="KG-D6"), {"block": "KG-D6"})
        self.assertEqual(
            build_chroma_filters(content_type="definition"),
            {"content_type": "definition"},
        )

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

    def test_instrument_name_ors_instrument_key(self):
        filters = build_chroma_filters(instrument_name="NEC-OSN-97-2 PSC")
        self.assertEqual(
            filters,
            {
                "$or": [
                    {"instrument_name": {"$eq": "NEC-OSN-97-2 PSC"}},
                    {"instrument_key": {"$eq": "nec-osn-97-2"}},
                ]
            },
        )

    def test_agent_fields_and_with_family(self):
        filters = build_chroma_filters(
            doc_family="PSC",
            block="NEC-OSN-97-2 (NEC25)",
            defined_terms="Petroleum",
            document_status="amendment",
            content_type="definition",
        )
        assert filters is not None
        self.assertIn("$and", filters)
        joined = str(filters)
        self.assertIn("block", joined)
        self.assertIn("defined_terms", joined)
        self.assertIn("document_status", joined)
        self.assertIn("content_type", joined)
        self.assertNotIn("running_summary", joined)
        self.assertNotIn("filename", joined)
        self.assertNotIn("_agno", joined)

    def test_ocr_uncertainty_true(self):
        self.assertEqual(
            build_chroma_filters(ocr_uncertainty="true"),
            {"ocr_uncertainty": True},
        )

    def test_ignores_blank_agent_fields(self):
        self.assertEqual(
            build_chroma_filters(heading="", parties="", cross_references=""),
            None,
        )


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
            block="KG-D6",
            defined_terms="Petroleum",
        )
        self.assertTrue(knowledge.calls)
        self.assertEqual(
            knowledge.calls[0]["query"],
            "relinquishment of right of development",
        )
        filters = knowledge.calls[0]["filters"]
        self.assertIn("$and", filters)
        blob = str(filters)
        self.assertIn("instrument_key", blob)
        self.assertIn("block", blob)
        self.assertIn("defined_terms", blob)


class SlimHitTests(unittest.TestCase):
    def test_truncates_chunk_text_and_drops_extra_metadata(self):
        doc = Document(
            content="A" * 2000,
            name="KGD6.pdf",
            meta_data={
                "instrument_name": "KGD6 PSC",
                "clause_id": "10.7",
                "raw_dump": "B" * 5000,
            },
        )
        hit = slim_hit(doc)
        self.assertLess(len(hit["content"]), 1300)
        self.assertTrue(hit["content"].endswith("…"))
        self.assertEqual(hit["meta_data"]["clause_id"], "10.7")
        self.assertNotIn("raw_dump", hit["meta_data"])


if __name__ == "__main__":
    unittest.main()
