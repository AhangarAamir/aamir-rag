"""Observability store, classifier, and summary payload."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory

from obs_adapter import classify_query_class, estimate_cost, resolve_model_id
from obs_api import _build_summary
from config import OPENAI_MODEL
from obs_store import add_comment, configure_db, iso, list_runs, upsert_run


class ClassifyTests(unittest.TestCase):
    def test_ingest_route(self):
        self.assertEqual(classify_query_class("any", "ingest-agent"), "ingest")

    def test_joa_from_text(self):
        self.assertEqual(classify_query_class("What does the JOA say about operators?"), "JOA")

    def test_pi_economics(self):
        self.assertEqual(
            classify_query_class("What is the participating interest of RIL?"),
            "PI / economics",
        )

    def test_default_legal_query(self):
        self.assertEqual(classify_query_class("Hello"), "legal query")


class CostTests(unittest.TestCase):
    def setUp(self):
        from obs_pricing import reset_for_tests

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
        from obs_pricing import reset_for_tests

        reset_for_tests()

    def tearDown(self):
        from obs_pricing import reset_for_tests

        reset_for_tests()

    def test_catalog_alias_lookup(self):
        from obs_pricing import load_catalog, model_rates, pricing_source

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

        from obs_pricing import model_rates, pricing_source

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


class StoreSummaryTests(unittest.TestCase):
    def setUp(self):
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
        self.assertEqual(classes, {"JOA", "ingest"})
        self.assertEqual(summary["slowest_route"], "ingest-agent")
        self.assertGreater(summary["p95_ms"], summary["avg_ms"])
        self.assertIn("cost_input", summary["days_series"][0])


class ApiTests(StoreSummaryTests):
    def setUp(self):
        super().setUp()
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from obs_api import router

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
        self.assertGreaterEqual(len(data["query_mix"]), 2)
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
