"""Query-time Chroma filters and per-run scratchpad."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agno.knowledge.document.base import Document
from agno.knowledge.reader.excel_reader import ExcelReader

from legal_rag.knowledge.catalog import identify_collection
from legal_rag.knowledge.chunking import (
    LegalAgenticChunking,
    LegalChunkDecision,
    PathAwareExcelReader,
    build_chroma_filters,
    origin_from_source,
    strip_workspace_path,
)
from legal_rag.agents.query import SharedCorpusKnowledgeTools, scratch_for_run
from legal_rag.prompts import QUERY_INSTRUCTIONS


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

    def test_sites_path_keeps_from_sites_and_minutes_theme(self):
        raw = (
            r"D:\aamirWS\data\ril_data\sites\Regulatory\KGD6"
            r"\MINUTESnRESOLUTIONS\OCR\Document Center\minutes.pdf"
        )
        origin = origin_from_source(raw)
        self.assertEqual(
            origin.folder_path,
            "sites/Regulatory/KGD6/MINUTESnRESOLUTIONS/OCR/Document Center",
        )
        self.assertEqual(origin.filename, "minutes.pdf")
        self.assertEqual(origin.guessed_family, "MINUTES_RESOLUTIONS")
        self.assertEqual(origin.block, "KGD6")
        self.assertEqual(origin.category, "OCR")
        self.assertEqual(
            identify_collection(origin.folder_path, origin.filename).code,
            "MINUTES_RESOLUTIONS",
        )

    def test_audited_accounts_under_kgd6_stays_minutes(self):
        raw = (
            "sites/Regulatory/KGD6/MINUTESnRESOLUTIONS/AUDITEDACCOUNTS/"
            "Document Center/accounts.pdf"
        )
        origin = origin_from_source(raw)
        self.assertEqual(origin.guessed_family, "MINUTES_RESOLUTIONS")
        self.assertEqual(origin.category, "AUDITEDACCOUNTS")
        self.assertNotEqual(origin.guessed_family, "AUDITED_ACCOUNTS")

    def test_kgudwhp_2022_mcm_mcr_is_mcr_mcm(self):
        raw = (
            "sites/Regulatory/KGUDWHP/Document Center/KGUDWHP-2022-1/"
            "MCM-MCR/status.xlsx"
        )
        origin = origin_from_source(raw)
        self.assertEqual(
            origin.folder_path,
            "sites/Regulatory/KGUDWHP/Document Center/KGUDWHP-2022-1/MCM-MCR",
        )
        self.assertEqual(origin.guessed_family, "MCR_MCM")
        self.assertEqual(origin.block, "KG-UDWHP-2022/1")
        self.assertEqual(origin.category, "MCM-MCR")

    def test_nec25_sites_is_minutes_not_nec25_family(self):
        raw = (
            "sites/Regulatory/NEC25/MINUTESnRESOLUTIONS/REVIEWOFDOC/"
            "Document Center/review.pdf"
        )
        origin = origin_from_source(raw)
        self.assertEqual(origin.guessed_family, "MINUTES_RESOLUTIONS")
        self.assertEqual(origin.block, "NEC25")
        self.assertEqual(origin.category, "REVIEWOFDOC")

    def test_joa_path_is_not_a_sites_theme(self):
        origin = origin_from_source(r"D:\aamirWS\data\ril_data\JOA\New JOA\file.pdf")
        self.assertEqual(origin.guessed_family, "JOA")
        self.assertFalse(origin.folder_path.lower().startswith("sites/"))
        self.assertEqual(origin.block, "")

    def test_empty_chunker_block_keeps_sites_block(self):
        document = Document(
            name="minutes",
            content="text",
            meta_data={
                "block": "KGD6",
                "doc_family": "MINUTES_RESOLUTIONS",
                "folder_path": "sites/Regulatory/KGD6/MINUTESnRESOLUTIONS/OCR",
            },
        )
        decision = LegalChunkDecision(doc_family="OCR", block="")
        meta = LegalAgenticChunking._chunk_metadata(
            None,
            document=document,
            decision=decision,
            chunk_number=1,
            chunk_text="text",
        )
        self.assertEqual(meta["block"], "KGD6")
        self.assertEqual(meta["doc_family"], "MINUTES_RESOLUTIONS")

    def test_excel_reader_stamps_sites_path(self):
        reader = PathAwareExcelReader(chunk=False)
        source = (
            r"D:\aamirWS\data\ril_data\sites\Regulatory\KGUDWHP\Document Center"
            r"\KGUDWHP-2018-1\MCR-MCM\status.xlsx"
        )
        with patch.object(
            ExcelReader,
            "read",
            return_value=[Document(content="a,b", name="status.xlsx")],
        ):
            docs = reader.read(source)
        meta = docs[0].meta_data
        self.assertEqual(
            meta["folder_path"],
            "sites/Regulatory/KGUDWHP/Document Center/KGUDWHP-2018-1/MCR-MCM",
        )
        self.assertEqual(meta["doc_family"], "MCR_MCM")
        self.assertEqual(meta["block"], "KG-UDWHP-2018/1")
        self.assertEqual(meta["sites_category"], "MCR-MCM")

    def test_sites_citation_rule_is_in_the_query_prompt(self):
        self.assertIn("folder_path starts with sites/", QUERY_INSTRUCTIONS)
        self.assertIn(
            "https://enp1.ril.com/sites/Regulatory/KGD6/MINUTESnRESOLUTIONS/OCR/Document%20Center/file.pdf",
            QUERY_INSTRUCTIONS,
        )
        self.assertIn(
            "https://enp1.ril.com/sites/Regulatory/Document%20Center/",
            QUERY_INSTRUCTIONS,
        )


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


if __name__ == "__main__":
    unittest.main()
