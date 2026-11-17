"""Tool calls are written to the console log."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from legal_rag.observability.console import _log_tool_call, install_tool_call_logging


class ToolConsoleTests(unittest.TestCase):
    def test_log_includes_agent_and_call(self):
        call = SimpleNamespace(
            function=SimpleNamespace(_agent=SimpleNamespace(name="Query Agent"), name="search_pdfs"),
            get_call_str=lambda: "search_pdfs(query='Article 10.7', folder='PSC')",
        )
        with patch("legal_rag.observability.console.log_info") as info:
            _log_tool_call(call)
        message = info.call_args.args[0]
        self.assertIn("Query Agent", message)
        self.assertIn("search_pdfs(query='Article 10.7', folder='PSC')", message)

    def test_execute_logs_the_call_before_running_the_tool(self):
        from agno.tools.function import FunctionCall

        install_tool_call_logging()
        call = SimpleNamespace(
            function=SimpleNamespace(_agent=SimpleNamespace(name="Ingest Agent"), name="ingest_path"),
            get_call_str=lambda: "ingest_path(path='data/ril_data')",
        )
        with patch("legal_rag.observability.console.log_info") as info:
            with self.assertRaises(AttributeError):
                FunctionCall.execute(call)
        message = info.call_args.args[0]
        self.assertIn("Ingest Agent tool call:", message)
        self.assertIn("ingest_path(path='data/ril_data')", message)
