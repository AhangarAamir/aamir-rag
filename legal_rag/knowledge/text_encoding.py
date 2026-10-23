"""Retry a text file in another encoding when UTF-8 cannot decode it.

Byte 0xE2 is valid in Windows-1252 (â) and is also a UTF-8 lead byte.
A Windows-saved note then raises ``UnicodeDecodeError: 'utf-8' codec can't
decode byte 0xe2``. The same bytes are read again as cp1252, then Latin-1.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Optional

from agno.utils.log import log_warning

_BINARY_MAGICS = (
    b"%PDF",
    b"PK\x03\x04",
    b"PK\x05\x06",
    b"PK\x07\x08",
    b"\xd0\xcf\x11\xe0\xa1\xb1",
    b"\x89PNG\r\n\x1a\n",
    b"\xff\xd8\xff",
)
# cp1252 first: that is the Windows encoding that contains byte 0xE2.
_FALLBACK_ENCODINGS = ("cp1252", "iso-8859-1")


def open_decoded(file: Any, name: Optional[str] = None) -> tuple[io.BytesIO, str]:
    """Return UTF-8 bytes the existing readers can open, plus the encoding used."""
    raw = _read_bytes(file)
    text, encoding = decode_text(raw)
    label = name or _label(file)
    if encoding != "utf-8":
        log_warning(f"Decoded {label} as {encoding} after UTF-8 failed.")
    buffer = io.BytesIO(text.encode("utf-8"))
    buffer.name = _buffer_name(file, name)
    return buffer, encoding


def decode_text(raw: bytes) -> tuple[str, str]:
    """Decode ``raw``. UTF-8 first, then Windows-1252, then Latin-1."""
    if not raw:
        return "", "utf-8"
    if raw.startswith(_BINARY_MAGICS):
        raise UnicodeDecodeError("utf-8", raw, 0, 1, "file is binary, not encoded text")
    try:
        return raw.decode("utf-8-sig"), "utf-8"
    except UnicodeDecodeError:
        pass
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            return raw.decode("utf-16"), "utf-16"
        except UnicodeDecodeError:
            pass
    for encoding in _FALLBACK_ENCODINGS:
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        if _readable(text):
            return text, encoding
    raise UnicodeDecodeError(
        "utf-8",
        raw,
        0,
        min(1, len(raw)),
        "could not decode as utf-8, cp1252, or latin-1",
    )


def build_encoding_csv_reader():
    """CSV reader that retries Windows-1252 when UTF-8 cannot decode the file.

    Agno's CSV reader imports aiofiles. When that package is missing, path
    ingest keeps the default reader.
    """
    try:
        from agno.knowledge.reader.csv_reader import CSVReader
    except ImportError:
        return None

    class EncodingCSVReader(CSVReader):
        def read(self, file, delimiter: str = ",", quotechar: str = '"', name: Optional[str] = None):
            stream, _encoding = open_decoded(file, name)
            return super().read(stream, delimiter=delimiter, quotechar=quotechar, name=name)

        async def async_read(
            self,
            file,
            delimiter: str = ",",
            quotechar: str = '"',
            page_size: int = 1000,
            name: Optional[str] = None,
        ):
            stream, _encoding = open_decoded(file, name)
            return await super().async_read(
                stream,
                delimiter=delimiter,
                quotechar=quotechar,
                page_size=page_size,
                name=name,
            )

    return EncodingCSVReader(name="Legal CSV Reader", chunk=False)


def build_encoding_json_reader():
    from agno.knowledge.reader.json_reader import JSONReader

    class EncodingJSONReader(JSONReader):
        def read(self, path, name: Optional[str] = None):
            stream, _encoding = open_decoded(path, name)
            return super().read(stream, name=name)

    return EncodingJSONReader(name="Legal JSON Reader", chunk=False)


def tag_encoding(documents: list, encoding: str) -> None:
    for document in documents:
        meta = dict(getattr(document, "meta_data", None) or {})
        meta["source_encoding"] = encoding
        document.meta_data = meta


def _readable(text: str) -> bool:
    sample = text[:200_000]
    if not sample.strip():
        return True
    length = len(sample)
    controls = sum(1 for char in sample if ord(char) < 32 and char not in "\n\r\t\f")
    if controls / length > 0.05:
        return False
    letters = sum(1 for char in sample if char.isalnum() or char.isspace())
    return letters / length >= 0.5


def _read_bytes(file: Any) -> bytes:
    if isinstance(file, Path):
        return file.read_bytes()
    if isinstance(file, str) and file.strip():
        return Path(file).read_bytes()
    read = getattr(file, "read", None)
    if read is None:
        return b""
    seek = getattr(file, "seek", None)
    if seek is not None:
        try:
            seek(0)
        except Exception:
            pass
    payload = read()
    if isinstance(payload, str):
        return payload.encode("utf-8")
    return payload or b""


def _buffer_name(file: Any, name: Optional[str]) -> str:
    if isinstance(file, Path):
        return str(file)
    if isinstance(file, str) and file.strip():
        return file
    return str(getattr(file, "name", "") or name or "text")


def _label(file: Any) -> str:
    if isinstance(file, Path):
        return file.name
    if isinstance(file, str) and file.strip():
        return Path(file).name
    return str(getattr(file, "name", "") or "file")
