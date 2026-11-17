"""gpt-6-luna tool calls must keep their reasoning item."""

from __future__ import annotations

import unittest

from agno.models.message import Message

from legal_rag.models import LunaResponses


def _tool_message() -> Message:
    return Message(
        role="assistant",
        content="",
        tool_calls=[
            {
                "id": "fc_abc",
                "call_id": "call_abc",
                "type": "function",
                "function": {"name": "search_pdfs", "arguments": "{}"},
            }
        ],
        provider_data={
            "response_id": "resp_123",
            "reasoning_output": {
                "id": "rs_abc",
                "type": "reasoning",
                "summary": [],
            },
        },
    )


class ReasoningModelTests(unittest.TestCase):
    def test_gpt6_luna_is_a_reasoning_model(self):
        self.assertTrue(LunaResponses(id="gpt-6-luna")._using_reasoning_model())
        self.assertTrue(LunaResponses(id="gpt-5.6-luna")._using_reasoning_model())
        self.assertFalse(LunaResponses(id="gpt-4o")._using_reasoning_model())

    def test_follow_up_uses_the_stored_response_instead_of_a_bare_function_call(self):
        model = LunaResponses(id="gpt-6-luna")
        messages = [
            Message(role="user", content="summarize the D34 discovery"),
            _tool_message(),
            Message(role="tool", content="well sequence", tool_call_id="fc_abc"),
        ]
        formatted = model._format_messages(messages)
        types = [item.get("type") for item in formatted if isinstance(item, dict)]
        self.assertNotIn("function_call", types)
        self.assertIn("function_call_output", types)
        params = model.get_request_params(messages=messages)
        self.assertEqual(params["previous_response_id"], "resp_123")

    def test_replayed_function_call_is_preceded_by_its_reasoning_item(self):
        model = LunaResponses(id="gpt-6-luna")
        message = _tool_message()
        message.provider_data = {"reasoning_output": message.provider_data["reasoning_output"]}
        formatted = model._format_messages([Message(role="user", content="D34"), message])
        kinds = []
        for item in formatted:
            if isinstance(item, dict):
                kinds.append(item.get("type") or item.get("role"))
            else:
                kinds.append(getattr(item, "type", None))
        self.assertLess(kinds.index("reasoning"), kinds.index("function_call"))
