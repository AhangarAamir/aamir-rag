"""UTF-8 decode failures are retried as Windows-1252, then Latin-1."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from legal_rag.knowledge.chunking import PathAwareTextReader
from legal_rag.knowledge.text_encoding import decode_text


class TextEncodingTests(unittest.TestCase):
    def test_byte_e2_is_read_as_windows_1252(self):
        text, encoding = decode_text("Work programme \u00e2 budget".encode("cp1252"))
        self.assertEqual(encoding, "cp1252")
        self.assertIn("budget", text)
        self.assertIn("\u00e2", text)

    def test_valid_utf8_stays_utf8(self):
        text, encoding = decode_text("Work programme \u2013 budget".encode("utf-8"))
        self.assertEqual(encoding, "utf-8")
        self.assertIn("\u2013", text)

    def test_binary_pdf_is_not_decoded_as_text(self):
        with self.assertRaises(UnicodeDecodeError):
            decode_text(b"%PDF-1.4\n(Safety \xe2 budget) endobj\n")

    def test_text_reader_loads_windows_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sites" / "Regulatory" / "KGD6" / "note.txt"
            path.parent.mkdir(parents=True)
            path.write_bytes("Safety \u00e2 budget".encode("cp1252"))
            docs = PathAwareTextReader(chunk=False).read(path)
        self.assertIn("budget", docs[0].content)
        self.assertEqual(docs[0].meta_data["source_encoding"], "cp1252")
        self.assertEqual(docs[0].meta_data["folder_path"], "sites/Regulatory/KGD6")
