"""Observability store, classifier, and summary payload."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory

import legal_rag.observability.adapter as obs_adapter
from legal_rag.observability.adapter import classify_query_class, estimate_cost, resolve_model_id, status_from_raw
from legal_rag.observability.api import _build_summary
from legal_rag.config import OPENAI_MODEL
from legal_rag.observability.store import add_comment, configure_db, iso, list_runs, upsert_run


class ClassifyTests(unittest.TestCase):
    def test_ingest_path_uses_folder_type(self):
        self.assertEqual(classify_query_class("ingest data/incoming/PSC/foo.pdf"), "PSC")

    def test_unclassified_text_is_empty(self):
        self.assertEqual(classify_query_class("any", "ingest-agent"), "")

    def test_joa_from_text(self):
        self.assertEqual(classify_query_class("What does the JOA say about operators?"), "JOA")

    def test_pi_without_family_is_empty(self):
        self.assertEqual(classify_query_class("What is the participating interest of RIL?"), "")

    def test_default_is_not_legal_query(self):
        self.assertEqual(classify_query_class("Hello"), "")


class CostTests(unittest.TestCase):
    def setUp(self):
        from legal_rag.observability.pricing import reset_for_tests

        reset_for_tests()

    def test_uses_reported_cost(self):
        self.assertEqual(estimate_cost("gpt-4o", 100, 10, reported=1.5), 1.5)

    def test_estimates_from_rates(self):
        cost = estimate_cost("gpt-4o", 1_000_000, 1_000_000, 0)
        self.assertAlmostEqual(cost, 12.5)

    def test_luna_official_rates(self):
        cost = estimate_cost("gpt-5.6-luna", 1_000_000, 1_000_000, 0)
        self.assertAlmostEqual(cost, 1.4)

    def test_unknown_model_is_zero_without_catalog(self):
        cost = estimate_cost("not-a-real-model", 1_000_000, 0, 0)
        self.assertAlmostEqual(cost, 0.0)


class CatalogPricingTests(unittest.TestCase):
    def setUp(self):
        from legal_rag.observability.pricing import reset_for_tests

        reset_for_tests()

    def tearDown(self):
        from legal_rag.observability.pricing import reset_for_tests

        reset_for_tests()

    def test_catalog_alias_lookup(self):
        from legal_rag.observability.pricing import load_catalog, model_rates, pricing_source

        load_catalog(
            {
                "azure/my-custom": {
                    "input_cost_per_token": 1e-6,
                    "output_cost_per_token": 3e-6,
                    "cache_read_input_token_cost": 1e-7,
                    "litellm_provider": "azure",
                }
            }
        )
        self.assertEqual(model_rates("my-custom"), (1.0, 0.1, 3.0))
        self.assertEqual(pricing_source("my-custom"), "catalog")

    def test_env_override_selected_model_only(self):
        import os
        from unittest.mock import patch

        from legal_rag.observability.pricing import model_rates, pricing_source

        with patch.dict(
            os.environ,
            {
                "OBS_INPUT_USD_PER_M": "9",
                "OBS_CACHED_USD_PER_M": "1",
                "OBS_OUTPUT_USD_PER_M": "11",
            },
        ):
            self.assertEqual(model_rates(OPENAI_MODEL), (9.0, 1.0, 11.0))
            self.assertEqual(pricing_source(OPENAI_MODEL), "env")
            self.assertEqual(model_rates("gpt-4o"), (2.5, 1.25, 10.0))


class ModelResolveTests(unittest.TestCase):
    def test_uses_env_model_when_missing(self):
        self.assertEqual(resolve_model_id(SimpleNamespace(model=None)), OPENAI_MODEL)
        self.assertEqual(OPENAI_MODEL, "gpt-5.6-luna")

    def test_uses_agent_model_id(self):
        agent = SimpleNamespace(model=SimpleNamespace(id="gpt-5.6-luna"))
        self.assertEqual(
            resolve_model_id(SimpleNamespace(model=""), agent),
            "gpt-5.6-luna",
        )


class StatusTests(unittest.TestCase):
    def test_post_hook_running_is_ok(self):
        self.assertEqual(status_from_raw("RUNNING", post_hook=True), "ok")
        self.assertEqual(status_from_raw("COMPLETED", post_hook=True), "ok")

    def test_session_running_stays_run(self):
        self.assertEqual(status_from_raw("RUNNING"), "run")
        self.assertEqual(status_from_raw("COMPLETED"), "ok")
        self.assertEqual(status_from_raw("ERROR"), "fail")


class SessionImportTests(unittest.TestCase):
    def setUp(self):
        import legal_rag.observability.adapter

        self.tmp = TemporaryDirectory()
        configure_db(Path(self.tmp.name) / "obs.db")
        self.sessions = Path(self.tmp.name) / "sessions.db"
        conn = __import__("sqlite3").connect(self.sessions)
        conn.execute(
            """
            CREATE TABLE query_sessions_runs (
                run_id TEXT PRIMARY KEY,
                session_id TEXT,
                agent_id TEXT,
                user_id TEXT,
                status TEXT,
                run_data TEXT NOT NULL,
                created_at INTEGER,
                updated_at INTEGER
            )
            """
        )
        payload = {
            "run_id": "hist-1",
            "agent_id": "query-agent",
            "agent_name": "Query Agent",
            "session_id": "s-hist",
            "model": "gpt-5.6-luna",
            "status": "COMPLETED",
            "created_at": 1790000000,
            "content": "The operator is RIL.",
            "input": {"input_content": "What does the JOA say about the operator?"},
            "metrics": {
                "input_tokens": 100,
                "output_tokens": 20,
                "cache_read_tokens": 0,
                "total_tokens": 120,
                "duration": 2.5,
            },
            "tools": [
                {
                    "tool_call_id": "t1",
                    "tool_name": "think",
                    "tool_args": {"thought": "plan"},
                    "tool_call_error": False,
                    "created_at": 1790000001,
                    "metrics": {"duration": 0.01},
                }
            ],
        }
        conn.execute(
            """
            INSERT INTO query_sessions_runs (
                run_id, session_id, agent_id, user_id, status, run_data, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "hist-1",
                "s-hist",
                "query-agent",
                "u1",
                "COMPLETED",
                __import__("json").dumps(payload),
                1790000000,
                1790000003,
            ),
        )
        conn.commit()
        conn.close()
        obs_adapter._SESSION_WATERMARK.clear()
        self._prev_import = obs_adapter.SESSION_IMPORT_ENABLED
        obs_adapter.SESSION_IMPORT_ENABLED = True

    def tearDown(self):
        import legal_rag.observability.adapter

        obs_adapter.SESSION_IMPORT_ENABLED = self._prev_import
        self.tmp.cleanup()

    def test_imports_completed_session_run(self):
        from legal_rag.observability.adapter import import_session_runs

        count = import_session_runs([self.sessions])
        self.assertEqual(count, 1)
        rows = list_runs(datetime(2020, 1, 1, tzinfo=timezone.utc))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["run_id"], "hist-1")
        self.assertEqual(rows[0]["status"], "ok")
        self.assertEqual(rows[0]["query_class"], "JOA")
        self.assertEqual(rows[0]["input_tokens"], 100)


class StoreSummaryTests(unittest.TestCase):
    def setUp(self):
        import legal_rag.observability.adapter

        obs_adapter.SESSION_IMPORT_ENABLED = False
        self.tmp = TemporaryDirectory()
        configure_db(Path(self.tmp.name) / "obs.db")
        start = datetime(2026, 9, 8, 12, 0, tzinfo=timezone.utc)
        upsert_run(
            {
                "run_id": "r1",
                "app": "legal-rag",
                "agent_id": "query-agent",
                "agent_name": "Query Agent",
                "route": "query-agent",
                "query_class": "JOA",
                "status": "ok",
                "model": "gpt-5.6-luna",
                "started_at": iso(start),
                "ended_at": iso(start + timedelta(seconds=21)),
                "duration_ms": 21000,
                "input_tokens": 1000,
                "output_tokens": 200,
                "cached_tokens": 0,
                "total_tokens": 1200,
                "cost_usd": 0.01,
                "session_id": "s1",
                "user_id": "u1",
                "input_text": "JOA operator?",
                "output_text": "The operator is...",
            }
        )
        upsert_run(
            {
                "run_id": "r2",
                "app": "legal-rag",
                "agent_id": "ingest-agent",
                "agent_name": "Ingest Agent",
                "route": "ingest-agent",
                "query_class": "ingest",
                "status": "fail",
                "model": "gpt-5.6-luna",
                "started_at": iso(start + timedelta(days=1)),
                "ended_at": iso(start + timedelta(days=1, seconds=40)),
                "duration_ms": 40000,
                "input_tokens": 500,
                "output_tokens": 50,
                "cached_tokens": 0,
                "total_tokens": 550,
                "cost_usd": 0.002,
                "session_id": "s2",
                "user_id": "u1",
                "input_text": "ingest all",
                "output_text": "error",
            }
        )
        add_comment("check this", run_id="r1", query="JOA operator?")

    def tearDown(self):
        self.tmp.cleanup()

    def test_lists_runs(self):
        rows = list_runs(datetime(2026, 9, 1, tzinfo=timezone.utc), app="legal-rag")
        self.assertEqual(len(rows), 2)

    def test_summary_kpis(self):
        since = datetime(2026, 9, 1, tzinfo=timezone.utc)
        runs = list_runs(since, app="legal-rag")
        summary = _build_summary(runs, [{"comment": "x"}], days=14)
        self.assertEqual(summary["runs"], 2)
        self.assertEqual(summary["ok"], 1)
        self.assertEqual(summary["fail"], 1)
        self.assertAlmostEqual(summary["success_rate"], 50.0)
        self.assertEqual(summary["tokens"], 1750)
        names = {row["name"] for row in summary["routes"]}
        self.assertEqual(names, {"query-agent", "ingest-agent"})
        classes = {row["name"] for row in summary["query_mix"]}
        self.assertEqual(classes, {"JOA", "PEL", "PML", "PSC", "RSC"})
        self.assertNotIn("ingest", classes)
        self.assertNotIn("legal query", classes)
        joa = next(row for row in summary["query_mix"] if row["name"] == "JOA")
        self.assertEqual(joa["count"], 1)
        self.assertEqual(sum(row["count"] for row in summary["query_mix"]), 1)
        self.assertAlmostEqual(summary["scatter"][0]["latency_s"], 21.0)
        self.assertEqual(summary["slowest_class"], "JOA")
        self.assertEqual(summary["slowest_route"], "ingest-agent")
        self.assertGreater(summary["p95_ms"], summary["avg_ms"])
        self.assertIn("cost_input", summary["days_series"][0])
        dates = [row["date"] for row in summary["days_series"]]
        parsed = [datetime.fromisoformat(day).date() for day in dates]
        self.assertEqual(
            parsed,
            [parsed[0] + timedelta(days=offset) for offset in range(len(parsed))],
        )
        self.assertGreater(len(dates), 2)
        empty = next(
            row
            for row in summary["days_series"]
            if row["ok"] + row["fail"] + row["run"] == 0
        )
        self.assertEqual(empty["cost"], 0.0)
        self.assertEqual(empty["input_tokens"], 0)

    def test_unrealistic_latency_is_omitted(self):
        start = datetime(2026, 9, 8, 12, 0, tzinfo=timezone.utc)
        upsert_run(
            {
                "run_id": "outlier",
                "app": "legal-rag",
                "agent_id": "query-agent",
                "agent_name": "Query Agent",
                "route": "query-agent",
                "query_class": "JOA",
                "status": "ok",
                "model": "gpt-5.6-luna",
                "started_at": iso(start),
                "ended_at": iso(start + timedelta(seconds=11000)),
                "duration_ms": 11_000_000,
                "input_tokens": 10,
                "output_tokens": 1,
                "cached_tokens": 0,
                "total_tokens": 11,
                "cost_usd": 0.0,
                "session_id": "s-out",
                "user_id": "u1",
                "input_text": "JOA outlier",
                "output_text": "",
            }
        )
        since = datetime(2026, 9, 1, tzinfo=timezone.utc)
        runs = list_runs(since, app="legal-rag")
        summary = _build_summary(runs, [], days=14)
        self.assertEqual(summary["runs"], 3)
        self.assertEqual(len(summary["scatter"]), 2)
        self.assertTrue(all(row["latency_s"] <= 200 for row in summary["scatter"]))
        self.assertLess(summary["p95_ms"], 200_000)


class ApiTests(StoreSummaryTests):
    def setUp(self):
        super().setUp()
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from legal_rag.observability.api import router

        app = FastAPI()
        app.include_router(router)
        self.client = TestClient(app)

    def test_summary_http(self):
        res = self.client.get("/observability/summary", params={"days": 365})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["runs"], 2)
        self.assertEqual(data["ok"], 1)
        self.assertEqual(data["fail"], 1)
        self.assertEqual({row["name"] for row in data["query_mix"]}, {"JOA", "PEL", "PML", "PSC", "RSC"})
        self.assertEqual(data["pricing"]["selected_model"], OPENAI_MODEL)

    def test_pricing_http(self):
        res = self.client.get("/observability/pricing")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["selected_model"], OPENAI_MODEL)
        self.assertIn(data["source"], {"fallback", "catalog", "env"})

    def test_comment_http(self):
        res = self.client.post("/observability/comments", json={"comment": "looks good"})
        self.assertEqual(res.status_code, 200)
        listed = self.client.get("/observability/summary", params={"days": 365})
        self.assertGreaterEqual(listed.json()["feedback"], 2)


if __name__ == "__main__":
    unittest.main()
