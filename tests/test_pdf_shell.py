"""PDF shell search over data/ril_data, including Windows-style paths."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from legal_rag.knowledge.chunking import origin_from_source
from legal_rag.tools.pdf_shell import (
    PdfCorpusShell,
    citation_for,
    main,
    read_corpus,
    resolve_under,
    search_corpus,
)


def write_text_pdf(path: Path, text: str) -> None:
    safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 12 Tf 72 720 Td ({safe}) Tj ET\n".encode("latin-1", errors="replace")
    parts = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Count 1 /Kids [3 0 R] >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"endstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    header = b"%PDF-1.4\n"
    chunks = []
    offsets = [0]
    cursor = len(header)
    for index, part in enumerate(parts, start=1):
        offsets.append(cursor)
        blob = f"{index} 0 obj\n".encode() + part + b"\nendobj\n"
        chunks.append(blob)
        cursor += len(blob)
    xref = [b"xref\n", f"0 {len(parts) + 1}\n".encode(), b"0000000000 65535 f \n"]
    for offset in offsets[1:]:
        xref.append(f"{offset:010d} 00000 n \n".encode())
    trailer = (
        f"trailer << /Size {len(parts) + 1} /Root 1 0 R >>\nstartxref\n{cursor}\n%%EOF\n".encode()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + b"".join(chunks) + b"".join(xref) + trailer)


class CorpusPathTests(unittest.TestCase):
    def test_windows_folder_stays_inside_the_corpus(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = resolve_under(root, r"JOA\New JOA")
            self.assertEqual(target, (root / "JOA" / "New JOA").resolve())
            self.assertIsNone(resolve_under(root, r"..\pyproject.toml"))
            self.assertIsNone(resolve_under(root, r"C:\Windows\file.pdf"))

    def test_citation_matches_document_center_and_sites_rules(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            joa = root / "JOA" / "New JOA" / "file.pdf"
            sites = (
                root
                / "sites"
                / "Regulatory"
                / "KGD6"
                / "MINUTESnRESOLUTIONS"
                / "OCR"
                / "minutes.pdf"
            )
            dumped = root / "RSC" / "aamirWS" / "data" / "ril_data" / "RSC" / "file.pdf"
            for path in (joa, sites, dumped):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"%PDF")
            self.assertEqual(
                citation_for(joa, root),
                {"filename": "file.pdf", "folder_path": "JOA/New JOA", "doc_family": "JOA"},
            )
            sites_hit = citation_for(sites, root)
            self.assertEqual(sites_hit["filename"], "minutes.pdf")
            self.assertTrue(sites_hit["folder_path"].startswith("sites/"))
            self.assertNotIn("ril_data", sites_hit["folder_path"])
            dumped_hit = citation_for(dumped, root)
            self.assertEqual(dumped_hit["folder_path"], "RSC")
            self.assertNotIn("aamirWS", dumped_hit["folder_path"])

    def test_ingest_origin_is_relative_to_ril_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf = root / "PEL" / "licence.pdf"
            pdf.parent.mkdir(parents=True)
            pdf.write_bytes(b"%PDF")
            with patch("legal_rag.knowledge.chunking.RIL_DATA_DIR", root):
                origin = origin_from_source(pdf)
            self.assertEqual(origin.folder_path, "PEL")
            self.assertEqual(origin.filename, "licence.pdf")
            self.assertNotIn("ril_data", origin.folder_path)


class SearchTests(unittest.TestCase):
    def test_search_and_read_use_pypdf_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf = root / "PSC" / "KGD6-PSC.pdf"
            write_text_pdf(pdf, "Article 10.7 Discovery Area participating interest")
            found = search_corpus(
                "participating interest",
                root=root,
                folder=r"PSC",
                filename="KGD6",
            )
            self.assertEqual(len(found["hits"]), 1)
            hit = found["hits"][0]
            self.assertEqual(hit["filename"], "KGD6-PSC.pdf")
            self.assertEqual(hit["folder_path"], "PSC")
            self.assertEqual(hit["page"], 1)
            self.assertIn("participating interest", hit["snippet"])
            self.assertNotIn(str(root), json.dumps(found))

            opened = read_corpus(r"PSC\KGD6-PSC.pdf", root=root, pages="1")
            self.assertEqual(opened["folder_path"], "PSC")
            self.assertIn("Article 10.7", opened["pages"][0]["text"])

    def test_escape_and_missing_folder_do_not_leave_the_corpus(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outside = Path(tmp).parent / "secret.pdf"
            write_text_pdf(outside, "participating interest secret")
            escaped = search_corpus("participating interest", root=root, folder="../")
            self.assertEqual(escaped["hits"], [])
            self.assertIn("data/ril_data", escaped["note"])
            missing = read_corpus("../secret.pdf", root=root)
            self.assertEqual(missing["pages"], [])

    def test_cli_prints_hits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_text_pdf(root / "JOA" / "New JOA" / "file.pdf", "Assignment of Interest")
            buffer = StringIO()
            with redirect_stdout(buffer):
                code = main(
                    ["--root", str(root), "search", "--query", "Assignment of Interest"]
                )
            self.assertEqual(code, 0)
            payload = json.loads(buffer.getvalue())
            self.assertEqual(payload["hits"][0]["folder_path"], "JOA/New JOA")


class ShellCommandTests(unittest.TestCase):
    def test_linux_and_windows_share_one_argv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_text_pdf(root / "PSC" / "file.pdf", "royalty well-head")
            tool = PdfCorpusShell(corpus_dir=root, timeout=60)
            self.assertEqual(set(tool.functions), {"search_pdfs", "read_pdf"})
            self.assertNotIn("run_shell_command", tool.functions)
            with patch("legal_rag.tools.pdf_shell.subprocess.run", wraps=__import__("subprocess").run) as run:
                raw = tool.search_pdfs("royalty", folder="PSC")
            argv = run.call_args.args[0]
            self.assertEqual(argv[0], sys.executable)
            self.assertEqual(argv[1:3], ["-m", "legal_rag.tools.pdf_shell"])
            self.assertNotIn("pdftotext", argv)
            self.assertNotIn("grep", argv)
            self.assertFalse(run.call_args.kwargs.get("shell", False))
            self.assertEqual(run.call_args.kwargs["cwd"], str(Path(tool.base_dir)))
            payload = json.loads(raw)
            self.assertEqual(payload["hits"][0]["folder_path"], "PSC")
            self.assertIn("royalty", payload["hits"][0]["snippet"])

            refused = tool._run(["pdftotext", "file.pdf"])
            self.assertTrue(refused.startswith("Error:"))
