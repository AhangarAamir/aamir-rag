"""One .log file per imported document, named from that document."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import json

from legal_rag.knowledge.import_log import (
    document_log_name,
    failed_files_from_folder,
    install_per_file_import_logs,
    record_import,
    write_document_log,
)


class _Row:
    def __init__(self, status: str, message: str, metadata: dict | None = None, name: str = ""):
        self.status = status
        self.status_message = message
        self.metadata = metadata or {}
        self.name = name


class _Knowledge:
    def __init__(self, rows: dict):
        self.rows = rows

    def get_content_by_id(self, content_id: str, user_id: str | None = None):
        return self.rows.get(content_id)


class ImportLogTests(unittest.TestCase):
    def test_log_name_is_the_file_name_plus_log(self):
        original = "2022-23(RE)_2023-24(BE)_20230412_MCR_Safety Management Review & Work Prog budget.0.1.pdf"
        self.assertEqual(document_log_name(original), f"{original}.log")

    def test_second_import_of_the_same_file_keeps_both_logs(self):
        when = datetime(2026, 10, 22, 16, 18, 0)
        name = "budget.0.1.pdf"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = write_document_log(
                file_name=name,
                source_path=rf"D:\data\{name}",
                status="failed",
                message="PDF reader extracted no text",
                finished_at=when,
                directory=root,
            )
            second = write_document_log(
                file_name=name,
                source_path=rf"D:\data\{name}",
                status="completed",
                finished_at=when,
                directory=root,
            )
            names = sorted(path.name for path in root.glob("*.log"))
            self.assertEqual(first.name, "budget.0.1.pdf.log")
            self.assertTrue(second.name.startswith("budget.0.1.pdf_"))
            self.assertTrue(second.name.endswith(".log"))
            self.assertEqual(len(names), 2)
            text = first.read_text(encoding="utf-8")
            self.assertIn("status: failed", text)
            self.assertIn("PDF reader extracted no text", text)

    def test_log_is_written_when_each_document_finishes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "sites" / "one.pdf"
            second = root / "sites" / "two.pdf"
            first.parent.mkdir()
            first.write_bytes(b"%PDF")
            second.write_bytes(b"%PDF")
            knowledge = _Loader()
            knowledge._import_log_dir = root / "logs"
            install_per_file_import_logs(knowledge)

            async def run():
                await knowledge._aload_from_path(_content(first, "completed", ""), True, False)
                await knowledge._aload_from_path(_content(second, "failed", "no text"), True, False)

            asyncio.run(run())
            names = sorted(path.name for path in (root / "logs").glob("*.log"))
            self.assertEqual(names, ["one.pdf.log", "two.pdf.log"])
            failed = (root / "logs" / "two.pdf.log").read_text(encoding="utf-8")
        self.assertIn("status: failed", failed)
        self.assertIn("no text", failed)

    def test_failed_files_use_child_rows_not_the_truncated_summary(self):
        knowledge = _Knowledge(
            {
                "parent": _Row(
                    "completed",
                    "1 of 2 files loaded",
                    {
                        "_agno": {
                            "children": ["ok", "bad"],
                            "failed": [{"path": "truncated-only.pdf", "error": "old"}],
                        }
                    },
                ),
                "ok": _Row("completed", "", {"_agno": {"source_path": "/ok.pdf"}}, "ok.pdf"),
                "bad": _Row(
                    "failed",
                    "ValueError: unreadable workbook",
                    {"_agno": {"source_path": "/bad.xlsx"}},
                    "bad.xlsx",
                ),
            }
        )
        failures = failed_files_from_folder(knowledge, "parent", None)
        self.assertEqual(
            failures,
            [{"path": "/bad.xlsx", "error": "ValueError: unreadable workbook"}],
        )

    def test_record_import_logs_a_failed_run_with_no_row(self):
        when = datetime(2026, 10, 23, 10, 1, 0)
        with tempfile.TemporaryDirectory() as tmp:
            logged = record_import(
                _Knowledge({}),
                source_path=r"D:\data\budget.pdf",
                started_at=when,
                result=json.dumps({"ok": False, "error": "PDF reader failed"}),
                directory=Path(tmp),
            )
            payload = json.loads(logged)
            text = Path(payload["import_log"]).read_text(encoding="utf-8")
        self.assertEqual(Path(payload["import_log"]).name, "budget.pdf.log")
        self.assertIn("budget.pdf", text)
        self.assertIn("PDF reader failed", text)


def _content(path: Path, status: str, message: str) -> SimpleNamespace:
    return SimpleNamespace(path=str(path), status=status, status_message=message, id="row", user_id=None)


class _Loader:
    _import_log_dir: Path

    async def _aload_from_path(self, content, upsert, skip_if_exists, include=None, exclude=None):
        return None

    def _load_from_path(self, content, upsert, skip_if_exists, include=None, exclude=None):
        return None
