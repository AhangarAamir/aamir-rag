"""Candidate lessons stay out of learned knowledge until a reviewer approves."""

from __future__ import annotations

import json
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from agno.learn import LearningMachine

import legal_rag.learning.extract as learning_extract
import legal_rag.learning.runtime as learning_runtime
import legal_rag.learning.store as learning_store
import legal_rag.observability.store as obs_store
from legal_rag.learning.runtime import configure_learning_machine, learning_config
from legal_rag.learning.service import (
    analyze_logs,
    approve_lesson,
    preview_lessons,
    reject_lesson,
    submit_feedback,
    summary,
)
from legal_rag.observability.store import iso, replace_spans, upsert_run


class FakeKnowledge:
    def __init__(self):
        self.rows: list[dict] = []

    def insert(self, name, text_content, reader=None, skip_if_exists=True, metadata=None):
        self.rows.append(
            {"name": name, "text": text_content, "metadata": metadata or {}}
        )

    def search(self, query, max_results=5, filters=None):
        hits = []
        namespace = (filters or {}).get("namespace")
        for row in self.rows:
            if namespace and row["metadata"].get("namespace") != namespace:
                continue
            hits.append(SimpleNamespace(content=row["text"]))
        return hits[:max_results]


def _draft(prompt: str) -> str:
    return json.dumps(
        {
            "title": "Recheck the named instrument",
            "lesson": "For similar legal assistant questions, re-check the named instrument before answering.",
            "severity": "medium",
            "tags": ["user_feedback"],
            "confidence": 0.8,
            "files": ["legal_rag/agents/query.py", "legal_rag/prompts.py"],
            "area": "contract",
            "context": "When the user corrects a citation",
        }
    )


class LearningReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        root = Path(self.tmp.name)
        self._learning_db = learning_store.LEARNING_DB_FILE
        self._obs_db = obs_store.OBS_DB_FILE
        learning_store.configure_db(root / "learning.db")
        obs_store.configure_db(root / "observability.db")
        self.knowledge = FakeKnowledge()
        configure_learning_machine(
            LearningMachine(knowledge=self.knowledge, learned_knowledge=learning_config())
        )
        learning_extract.configure_extractor(_draft)

    def tearDown(self):
        learning_extract.configure_extractor(None)
        configure_learning_machine(None)
        learning_store.configure_db(self._learning_db)
        obs_store.configure_db(self._obs_db)
        self.tmp.cleanup()

    def test_feedback_is_a_candidate_until_approved(self):
        created = submit_feedback(
            feedback="That clause is from the wrong PSC.",
            query="What does article 10.7 say?",
            answer="Article 10.7 sets the participating interest.",
            run_id="run-1",
            session_id="session-1",
            user_id="user-1",
        )
        lesson = created["lesson"]
        self.assertEqual(lesson["status"], "candidate")
        self.assertEqual(self.knowledge.rows, [])
        self.assertEqual(preview_lessons("article 10.7"), [])

        approved = approve_lesson(lesson["lesson_id"])
        self.assertEqual(approved["status"], "approved")
        self.assertEqual(len(self.knowledge.rows), 1)
        self.assertEqual(learning_store.count_eval_cases(), 1)
        recalled = preview_lessons("article 10.7")
        self.assertEqual([item["title"] for item in recalled], ["Recheck the named instrument"])
        self.assertEqual(recalled[0]["lesson_id"], lesson["lesson_id"])
        counts = summary(datetime.now(timezone.utc) - timedelta(days=14))
        self.assertEqual(counts["total_feedbacks"], 1)
        self.assertEqual(counts["approved_feedback"], 1)
        self.assertEqual(counts["incidents"], 1)

    def test_reject_does_not_enter_learned_knowledge(self):
        created = submit_feedback(
            feedback="Irrelevant answer",
            query="Who is the operator?",
            answer="The operator is not stated.",
            run_id="run-2",
        )
        rejected = reject_lesson(created["lesson"]["lesson_id"])
        self.assertEqual(rejected["status"], "rejected")
        self.assertEqual(self.knowledge.rows, [])
        self.assertEqual(preview_lessons("operator"), [])
        self.assertEqual(learning_store.count_eval_cases(), 0)

    def test_preview_ignores_unapproved_vector_rows(self):
        created = submit_feedback(
            feedback="Wrong document",
            query="Show the assignment clause",
            answer="See the JOA.",
            run_id="run-3",
        )
        approve_lesson(created["lesson"]["lesson_id"])
        self.knowledge.rows.append(
            {
                "name": "not approved",
                "text": json.dumps(
                    {
                        "title": "Should stay hidden",
                        "learning": "Do not use this.",
                        "namespace": "global",
                    }
                ),
                "metadata": {"namespace": "global"},
            }
        )
        titles = [item["title"] for item in preview_lessons("assignment")]
        self.assertEqual(titles, ["Recheck the named instrument"])

    def test_analyze_logs_creates_candidates_only(self):
        started = datetime.now(timezone.utc) - timedelta(hours=1)
        upsert_run(
            {
                "run_id": "run-fail",
                "app": "legal-rag",
                "agent_id": "query-agent",
                "agent_name": "Query Agent",
                "route": "query-agent",
                "query_class": "PSC",
                "status": "fail",
                "model": "test",
                "started_at": iso(started),
                "ended_at": iso(started),
                "duration_ms": 10,
                "input_tokens": 1,
                "output_tokens": 1,
                "cached_tokens": 0,
                "total_tokens": 2,
                "cost_usd": 0,
                "session_id": "s",
                "user_id": "u",
                "input_text": "What is the PSC term?",
                "output_text": "",
            }
        )
        replace_spans(
            "run-fail",
            [
                {
                    "span_id": "span-1",
                    "name": "search_knowledge",
                    "kind": "tool",
                    "status": "error",
                    "started_at": iso(started),
                    "duration_ms": 5,
                    "detail": "search failed",
                }
            ],
        )
        first = analyze_logs(datetime.now(timezone.utc) - timedelta(days=1))
        self.assertEqual(first["created"], 1)
        lessons = learning_store.list_lessons(status="candidate")
        self.assertEqual(len(lessons), 1)
        self.assertEqual(lessons[0]["status"], "candidate")
        self.assertEqual(self.knowledge.rows, [])
        second = analyze_logs(datetime.now(timezone.utc) - timedelta(days=1))
        self.assertEqual(second["created"], 0)

    def test_agentic_store_can_search_but_not_save(self):
        store = learning_runtime.get_learning_machine().learned_knowledge_store
        names = [tool.__name__ for tool in store.get_agent_tools()]
        self.assertEqual(store.config.mode.value, "agentic")
        self.assertEqual(store.config.namespace, "global")
        self.assertIn("search_learnings", names)
        self.assertNotIn("save_learning", names)
