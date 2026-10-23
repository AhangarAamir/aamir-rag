"""An empty PDF extract must not be stored as zero documents."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from agno.knowledge.document.base import Document
from agno.knowledge.reader.pdf_reader import PDFReader

from legal_rag.knowledge.chunking import PathAwarePDFReader


class EmptyPdfExtractTests(unittest.TestCase):
    def test_pdftotext_is_used_when_the_pdf_reader_returns_no_text(self):
        reader = PathAwarePDFReader(chunk=False)
        with tempfile_pdf() as path:
            with patch.object(PDFReader, "read", return_value=[Document(name="budget", content="  ")]):
                with patch(
                    "legal_rag.knowledge.chunking._pdftotext",
                    return_value="Safety Management Review budget",
                ) as pdftotext:
                    docs = reader.read(path, name=path.name)
        pdftotext.assert_called_once()
        self.assertIn("Safety Management Review", docs[0].content)
        self.assertIn("OTHERAPPROVAL", docs[0].meta_data["folder_path"])

    def test_file_fails_when_no_extractor_finds_text(self):
        reader = PathAwarePDFReader(chunk=False)
        with tempfile_pdf() as path:
            with patch.object(PDFReader, "read", return_value=[Document(name="budget", content="")]):
                with patch("legal_rag.knowledge.chunking._pdftotext", return_value=""):
                    with patch("legal_rag.knowledge.chunking._pypdf_layout_text", return_value=""):
                        with self.assertRaises(RuntimeError) as caught:
                            reader.read(path, name=path.name)
        self.assertIn("not indexed", str(caught.exception))

    def test_existing_text_skips_pdftotext(self):
        reader = PathAwarePDFReader(chunk=False)
        with tempfile_pdf() as path:
            with patch.object(
                PDFReader,
                "read",
                return_value=[Document(name="budget", content="already extracted")],
            ):
                with patch("legal_rag.knowledge.chunking._pdftotext") as pdftotext:
                    docs = reader.read(path, name=path.name)
        pdftotext.assert_not_called()
        self.assertEqual(docs[0].content, "already extracted")


def tempfile_pdf():
    import tempfile
    from contextlib import contextmanager

    @contextmanager
    def _open():
        with tempfile.TemporaryDirectory() as tmp:
            path = (
                Path(tmp)
                / "sites"
                / "Regulatory"
                / "KGD6"
                / "MINUTESnRESOLUTIONS"
                / "OTHERAPPROVAL"
                / "Document Center"
                / "budget.pdf"
            )
            path.parent.mkdir(parents=True)
            path.write_bytes(b"%PDF-1.4\n")
            yield path

    return _open()
