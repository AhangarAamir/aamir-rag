"""Standalone-query isolation from prior session chat."""

from __future__ import annotations

import unittest
from agno.models.message import Message
from agno.run import RunContext
from agno.run.agent import RunInput, RunOutput
from agno.run.base import RunStatus
from agno.session.agent import AgentSession
from agno.session.summary import SessionSummary

from legal_rag.agents.topic_shift import (
    hide_prior_context,
    isolate_unrelated_query,
    looks_like_follow_up,
    prior_user_texts,
    restore_prior_context,
    restore_query_history,
    should_isolate_from_history,
)


def _session_with_prior(question: str, summary: str = "Prior topic: Article 10.7") -> AgentSession:
    return AgentSession(
        session_id="s1",
        summary=SessionSummary(summary=summary, topics=["10.7"]),
        runs=[
            RunOutput(
                run_id="r1",
                status=RunStatus.completed,
                input=RunInput(input_content=question),
                messages=[
                    Message(role="user", content=question),
                    Message(role="assistant", content="Article 10.7 requires a development plan."),
                ],
            )
        ],
    )


class FollowUpHeuristicTests(unittest.TestCase):
    def test_anaphora_is_follow_up(self):
        self.assertTrue(looks_like_follow_up("Does that clause still apply?"))
        self.assertTrue(looks_like_follow_up("What about royalty?"))
        self.assertTrue(looks_like_follow_up("and the same contract?"))

    def test_self_contained_question_is_not_follow_up(self):
        self.assertFalse(
            looks_like_follow_up("What is the participating interest of RIL?")
        )
        self.assertFalse(looks_like_follow_up("hello"))
        self.assertFalse(
            looks_like_follow_up(
                "What is the definition of Petroleum in this Agreement as per KGD6 PSC?"
            )
        )


class ShouldIsolateTests(unittest.TestCase):
    def test_no_prior_keeps_history(self):
        self.assertFalse(
            should_isolate_from_history(
                "What is the participating interest of RIL?",
                [],
                classify=lambda current, prior: True,
            )
        )

    def test_follow_up_keeps_history_without_llm(self):
        called = {"n": 0}

        def classify(current, prior):
            called["n"] += 1
            return False

        self.assertFalse(
            should_isolate_from_history(
                "Does that still apply to assignment?",
                ["What does article 10.7 state in KGD6 PSC?"],
                classify=classify,
            )
        )
        self.assertEqual(called["n"], 0)

    def test_classifier_standalone_isolates(self):
        self.assertTrue(
            should_isolate_from_history(
                "What is the participating interest of RIL?",
                ["What does article 10.7 state in KGD6 PSC?"],
                classify=lambda current, prior: False,
            )
        )

    def test_classifier_related_keeps_history(self):
        self.assertFalse(
            should_isolate_from_history(
                "What is the royalty rate?",
                ["What does article 10.7 state in KGD6 PSC?"],
                classify=lambda current, prior: True,
            )
        )

    def test_classifier_error_defaults_to_standalone(self):
        def boom(current, prior):
            raise RuntimeError("no model")

        self.assertTrue(
            should_isolate_from_history(
                "What is the participating interest of RIL?",
                ["What does article 10.7 state in KGD6 PSC?"],
                classify=boom,
            )
        )


class HideRestoreTests(unittest.TestCase):
    def test_hide_clears_messages_and_summary(self):
        session = _session_with_prior("What does article 10.7 state in KGD6 PSC?")
        self.assertTrue(session.get_messages())
        self.assertIsNotNone(session.summary)
        hide_prior_context(session)
        self.assertEqual(session.get_messages(), [])
        self.assertIsNone(session.summary)
        restore_prior_context(session)
        restored = session.get_messages()
        self.assertTrue(restored)
        self.assertEqual(session.summary.summary, "Prior topic: Article 10.7")

    def test_restore_is_idempotent(self):
        session = _session_with_prior("What does article 10.7 state in KGD6 PSC?")
        restore_prior_context(session)
        restore_query_history(session)
        self.assertTrue(session.get_messages())


class HookTests(unittest.TestCase):
    def test_pre_hook_hides_unrelated_question(self):
        session = _session_with_prior("What does article 10.7 state in KGD6 PSC?")
        run_input = RunInput(input_content="What is the participating interest of RIL?")
        ctx = RunContext(run_id="r2", session_id="s1", session_state={})
        isolate_unrelated_query(
            run_input,
            session,
            ctx,
            classify=lambda current, prior: False,
        )
        self.assertTrue(ctx.session_state["standalone_query"])
        self.assertEqual(session.get_messages(), [])
        self.assertEqual(prior_user_texts(session), ["What does article 10.7 state in KGD6 PSC?"])
        restore_query_history(session)
        self.assertTrue(session.get_messages())

    def test_pre_hook_keeps_follow_up(self):
        session = _session_with_prior("What does article 10.7 state in KGD6 PSC?")
        run_input = RunInput(input_content="Does that still apply?")
        ctx = RunContext(run_id="r2", session_id="s1", session_state={})
        isolate_unrelated_query(run_input, session, ctx, classify=lambda current, prior: False)
        self.assertFalse(ctx.session_state["standalone_query"])
        self.assertTrue(session.get_messages())


class PriorUserTextsTests(unittest.TestCase):
    def test_reads_run_input(self):
        session = _session_with_prior("What does article 10.7 state in KGD6 PSC?")
        self.assertEqual(
            prior_user_texts(session),
            ["What does article 10.7 state in KGD6 PSC?"],
        )

    def test_empty_session(self):
        session = AgentSession(session_id="empty")
        self.assertEqual(prior_user_texts(session), [])


if __name__ == "__main__":
    unittest.main()
