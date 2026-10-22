"""Per-run import failure logs."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from legal_rag.knowledge.import_log import failed_files_from_folder, write_import_log


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
    def test_two_writes_in_the_same_second_make_two_files(self):
        when = datetime(2026, 10, 22, 16, 18, 0)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = write_import_log(
                source_path=r"D:\aamirWS\data\ril_data\sites",
                started_at=when,
                finished_at=when,
                status="completed",
                loaded=2,
                failures=[{"path": r"D:\aamirWS\data\ril_data\sites\bad.xlsx", "error": "ExcelReader: boom"}],
                directory=root,
            )
            second = write_import_log(
                source_path=r"D:\aamirWS\data\ril_data\sites",
                started_at=when,
                finished_at=when,
                status="completed",
                loaded=4,
                failures=[],
                directory=root,
            )
            names = sorted(path.name for path in root.glob("import_*.log"))
            self.assertEqual(
                names,
                ["import_2026-10-22_161800.log", "import_2026-10-22_161800_2.log"],
            )
            self.assertEqual(first.name, names[0])
            self.assertEqual(second.name, names[1])
            text = first.read_text(encoding="utf-8")
            self.assertIn("failed: 1", text)
            self.assertIn("bad.xlsx", text)
            self.assertIn("ExcelReader: boom", text)
            self.assertIn("No file failures.", second.read_text(encoding="utf-8"))

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
