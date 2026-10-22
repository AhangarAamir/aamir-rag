"""Specialized readers fall back to the text reader."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from agno.knowledge.document.base import Document
from agno.knowledge.reader.base import Reader
from agno.knowledge.reader.reader_factory import ReaderFactory

from legal_rag.knowledge import reader_fallback as reader_fallback_module
from legal_rag.knowledge.chunking import PathAwareTextReader
from legal_rag.knowledge.reader_fallback import (
    TextFallbackReader,
    async_read_with_text_fallback,
    install_text_fallback,
    read_with_text_fallback,
)


class _Boom(Reader):
    def __init__(self) -> None:
        super().__init__(chunk=False, name="Boom reader")

    def read(self, file, name=None, password=None):
        raise ValueError("pdf boom")

    async def async_read(self, file, name=None, password=None):
        raise ValueError("pdf boom")


class _Blank(Reader):
    def __init__(self) -> None:
        super().__init__(chunk=False, name="Blank reader")

    def read(self, file, name=None, password=None):
        return [Document(name=name or "blank", content="   ")]

    async def async_read(self, file, name=None, password=None):
        return [Document(name=name or "blank", content="   ")]


class _Ok(Reader):
    def __init__(self) -> None:
        super().__init__(chunk=False, name="Ok reader")

    def read(self, file, name=None, password=None):
        return [Document(name=name or "ok", content="from the pdf reader")]


class _EmptyText(Reader):
    def __init__(self) -> None:
        super().__init__(chunk=False, name="Empty text")

    def read(self, file, name=None):
        return []

    async def async_read(self, file, name=None):
        return []


class ReaderFallbackTests(unittest.TestCase):
    def test_reader_exception_uses_text_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sites" / "Regulatory" / "KGD6" / "note.pdf"
            path.parent.mkdir(parents=True)
            path.write_text("appointment minutes", encoding="utf-8")
            docs = read_with_text_fallback(_Boom(), PathAwareTextReader(chunk=False), path)
        self.assertEqual(docs[0].content, "appointment minutes")
        self.assertEqual(docs[0].meta_data["reader_fallback"], "text")
        self.assertIn("pdf boom", docs[0].meta_data["reader_error"])
        self.assertEqual(docs[0].meta_data["folder_path"], "sites/Regulatory/KGD6")

    def test_empty_reader_result_uses_text_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "status.xlsx"
            path.write_text("column text", encoding="utf-8")
            docs = read_with_text_fallback(_Blank(), PathAwareTextReader(chunk=False), path)
        self.assertEqual(docs[0].content, "column text")
        self.assertEqual(docs[0].meta_data["reader_replaced"], "Blank reader")

    def test_successful_reader_is_kept(self):
        text_reader = _EmptyText()
        docs = read_with_text_fallback(_Ok(), text_reader, Path("ignored.pdf"), name="ignored.pdf")
        self.assertEqual(docs[0].content, "from the pdf reader")
        self.assertNotIn("reader_fallback", docs[0].meta_data or {})

    def test_raises_original_error_when_text_reader_is_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.pdf"
            path.write_bytes(b"\x00\x00")
            with self.assertRaises(ValueError) as caught:
                read_with_text_fallback(_Boom(), _EmptyText(), path)
        self.assertIn("pdf boom", str(caught.exception))

    def test_async_reader_exception_uses_text_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "minutes.pdf"
            path.write_text("async minutes", encoding="utf-8")
            docs = asyncio.run(
                async_read_with_text_fallback(_Boom(), PathAwareTextReader(chunk=False), path)
            )
        self.assertEqual(docs[0].content, "async minutes")
        self.assertEqual(docs[0].meta_data["reader_fallback"], "text")

    def test_factory_pdf_reader_falls_back_and_txt_stays_plain(self):
        text_reader = PathAwareTextReader(chunk=False)
        previous_get = ReaderFactory.get_reader_for_extension
        previous_pdf = ReaderFactory._reader_cache.get("pdf")
        try:
            install_text_fallback(text_reader)
            ReaderFactory._reader_cache["pdf"] = _Boom()
            reader = ReaderFactory.get_reader_for_extension(".pdf")
            self.assertIsInstance(reader, TextFallbackReader)
            plain = ReaderFactory.get_reader_for_extension(".txt")
            self.assertIs(plain, text_reader)
            with tempfile.TemporaryDirectory() as tmp:
                path = (
                    Path(tmp)
                    / "sites"
                    / "Regulatory"
                    / "KGUDWHP"
                    / "Document Center"
                    / "KGUDWHP-2018-1"
                    / "MCR-MCM"
                    / "status.pdf"
                )
                path.parent.mkdir(parents=True)
                path.write_text("mcr text", encoding="utf-8")
                docs = reader.read(path)
            self.assertEqual(docs[0].content, "mcr text")
            self.assertEqual(
                docs[0].meta_data["folder_path"],
                "sites/Regulatory/KGUDWHP/Document Center/KGUDWHP-2018-1/MCR-MCM",
            )
            self.assertEqual(docs[0].meta_data["doc_family"], "MCR_MCM")
        finally:
            ReaderFactory.get_reader_for_extension = reader_fallback_module._ORIGINAL_GET or previous_get
            if previous_pdf is None:
                ReaderFactory._reader_cache.pop("pdf", None)
            else:
                ReaderFactory._reader_cache["pdf"] = previous_pdf
