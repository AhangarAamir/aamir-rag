"""Retry a failed specialized reader with the text reader.

Import uses ReaderFactory.get_reader_for_extension. When that reader raises,
or returns no usable text for a non-empty file, the file is read as text
before the import is recorded as a failure.
"""

from __future__ import annotations

import inspect
import io
import re
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

from agno.knowledge.document.base import Document
from agno.knowledge.reader.base import Reader
from agno.knowledge.reader.reader_factory import ReaderFactory
from agno.knowledge.reader.text_reader import TextReader
from agno.utils.log import log_warning

_ORIGINAL_GET: Optional[Callable[..., Reader]] = None
_READABLE = re.compile(r"[A-Za-z0-9]")


class _UnavailableReader(Reader):
    """Stand-in when no specialized reader can be constructed."""

    def __init__(self, exc: BaseException):
        super().__init__(chunk=False, name="Unavailable reader")
        self.exc = exc

    def read(self, file: Any, name: Optional[str] = None, password: Optional[str] = None) -> List[Document]:
        raise self.exc

    async def async_read(
        self,
        file: Any,
        name: Optional[str] = None,
        password: Optional[str] = None,
    ) -> List[Document]:
        raise self.exc


class TextFallbackReader(Reader):
    """Try ``primary`` and, on a reader failure, read the file as text."""

    def __init__(self, primary: Reader, text_reader: Reader):
        super().__init__(chunk=False, name="Text fallback reader")
        self.primary = primary
        self.text_reader = text_reader

    def read(
        self,
        file: Any,
        name: Optional[str] = None,
        password: Optional[str] = None,
    ) -> List[Document]:
        return read_with_text_fallback(self.primary, self.text_reader, file, name=name, password=password)

    async def async_read(
        self,
        file: Any,
        name: Optional[str] = None,
        password: Optional[str] = None,
    ) -> List[Document]:
        return await async_read_with_text_fallback(
            self.primary,
            self.text_reader,
            file,
            name=name,
            password=password,
        )


def install_text_fallback(text_reader: Reader) -> None:
    """Wrap factory readers so a reader-side failure retries as text."""
    global _ORIGINAL_GET
    if _ORIGINAL_GET is None:
        _ORIGINAL_GET = ReaderFactory.get_reader_for_extension

    def _get(extension: str) -> Reader:
        normalized = (extension or "").lower()
        if normalized in {".txt", ".text"}:
            return text_reader
        assert _ORIGINAL_GET is not None
        try:
            reader = _ORIGINAL_GET(extension)
        except Exception as exc:
            log_warning(f"No reader for {extension or '(none)'} ({exc}). Using the text reader.")
            return TextFallbackReader(_UnavailableReader(exc), text_reader)
        if isinstance(reader, (TextReader, TextFallbackReader)):
            return reader
        return TextFallbackReader(reader, text_reader)

    ReaderFactory.get_reader_for_extension = _get  # type: ignore[method-assign]


def read_with_text_fallback(
    primary: Reader,
    text_reader: Reader,
    file: Any,
    name: Optional[str] = None,
    password: Optional[str] = None,
) -> List[Document]:
    documents, error = _attempt_primary(primary.read, file, name, password)
    return _finish_fallback(primary, text_reader, file, name, documents, error)


async def async_read_with_text_fallback(
    primary: Reader,
    text_reader: Reader,
    file: Any,
    name: Optional[str] = None,
    password: Optional[str] = None,
) -> List[Document]:
    documents, error = await _attempt_primary_async(primary.async_read, file, name, password)
    return await _finish_fallback_async(primary, text_reader, file, name, documents, error)


def _attempt_primary(
    method: Callable[..., List[Document]],
    file: Any,
    name: Optional[str],
    password: Optional[str],
) -> tuple[List[Document], Optional[BaseException]]:
    _rewind(file)
    try:
        return _invoke(method, file, name, password), None
    except FileNotFoundError:
        raise
    except Exception as exc:
        log_warning(f"{_reader_name(method)} failed for {_label(file, name)}: {exc}. Retrying with the text reader.")
        return [], exc


async def _attempt_primary_async(
    method: Callable[..., Any],
    file: Any,
    name: Optional[str],
    password: Optional[str],
) -> tuple[List[Document], Optional[BaseException]]:
    _rewind(file)
    try:
        documents = await _invoke(method, file, name, password)
        return list(documents or []), None
    except FileNotFoundError:
        raise
    except Exception as exc:
        log_warning(f"{_reader_name(method)} failed for {_label(file, name)}: {exc}. Retrying with the text reader.")
        return [], exc


def _finish_fallback(
    primary: Reader,
    text_reader: Reader,
    file: Any,
    name: Optional[str],
    documents: List[Document],
    error: Optional[BaseException],
) -> List[Document]:
    if _usable(documents):
        return documents
    if error is None and _source_size(file) == 0:
        return documents
    _rewind(file)
    fallback = _usable(text_reader.read(file, name=name))
    if not fallback:
        _rewind(file)
        fallback = _usable(_recover_replaced_text(file, name, text_reader))
    if fallback:
        _tag(fallback, error, primary)
        log_warning(f"Loaded {_label(file, name)} with the text reader after {_reader_name(primary)} failed.")
        return fallback
    if error is not None:
        raise error
    raise RuntimeError(f"{_reader_name(primary)} produced no text and the text reader fallback was empty")


async def _finish_fallback_async(
    primary: Reader,
    text_reader: Reader,
    file: Any,
    name: Optional[str],
    documents: List[Document],
    error: Optional[BaseException],
) -> List[Document]:
    if _usable(documents):
        return documents
    if error is None and _source_size(file) == 0:
        return documents
    _rewind(file)
    fallback = _usable(await text_reader.async_read(file, name=name))
    if not fallback:
        _rewind(file)
        fallback = _usable(_recover_replaced_text(file, name, text_reader))
    if fallback:
        _tag(fallback, error, primary)
        log_warning(f"Loaded {_label(file, name)} with the text reader after {_reader_name(primary)} failed.")
        return fallback
    if error is not None:
        raise error
    raise RuntimeError(f"{_reader_name(primary)} produced no text and the text reader fallback was empty")


def _recover_replaced_text(file: Any, name: Optional[str], text_reader: Reader) -> List[Document]:
    """Decode bytes with replacement characters when strict UTF-8 fails."""
    raw = _read_bytes(file)
    if not raw:
        return []
    text = raw.decode("utf-8", errors="replace")
    if not _READABLE.search(text):
        return []
    buffer = io.BytesIO(text.encode("utf-8"))
    if isinstance(file, Path):
        buffer.name = str(file)
        fallback_name = name or file.name
    else:
        buffer.name = name or getattr(file, "name", "") or "text"
        fallback_name = name or Path(str(buffer.name)).name
    return text_reader.read(buffer, name=fallback_name)


def _tag(documents: List[Document], error: Optional[BaseException], primary: Reader) -> None:
    detail = ""
    if error is not None:
        detail = f"{type(error).__name__}: {error}"[:300]
    for document in documents:
        meta = dict(document.meta_data or {})
        meta["reader_fallback"] = "text"
        meta["reader_replaced"] = _reader_name(primary)
        if detail:
            meta["reader_error"] = detail
        document.meta_data = meta


def _usable(documents: Optional[List[Document]]) -> List[Document]:
    kept = [document for document in documents or [] if str(getattr(document, "content", "") or "").strip()]
    return kept


def _invoke(method: Callable[..., Any], file: Any, name: Optional[str], password: Optional[str]) -> Any:
    kwargs: dict[str, Any] = {}
    parameters = inspect.signature(method).parameters
    if "name" in parameters:
        kwargs["name"] = name
    if password is not None and "password" in parameters:
        kwargs["password"] = password
    return method(file, **kwargs)


def _rewind(file: Any) -> None:
    seek = getattr(file, "seek", None)
    if seek is None or isinstance(file, Path):
        return
    try:
        seek(0)
    except Exception:
        return


def _source_size(file: Any) -> int:
    if isinstance(file, Path):
        try:
            return file.stat().st_size
        except OSError:
            return 0
    seek = getattr(file, "seek", None)
    tell = getattr(file, "tell", None)
    if seek is None or tell is None:
        return 1
    try:
        position = tell()
        seek(0, io.SEEK_END)
        size = tell()
        seek(position)
        return int(size)
    except Exception:
        return 1


def _read_bytes(file: Any) -> bytes:
    if isinstance(file, Path):
        try:
            return file.read_bytes()
        except OSError:
            return b""
    read = getattr(file, "read", None)
    if read is None:
        return b""
    _rewind(file)
    try:
        payload = read()
    except Exception:
        return b""
    if isinstance(payload, str):
        return payload.encode("utf-8", errors="replace")
    return payload or b""


def _label(file: Any, name: Optional[str]) -> str:
    if name:
        return name
    if isinstance(file, Path):
        return file.name
    return str(getattr(file, "name", "") or "file")


def _reader_name(reader: Union[Reader, Callable[..., Any]]) -> str:
    if isinstance(reader, Reader):
        return reader.name or type(reader).__name__
    owner = getattr(reader, "__self__", None)
    if isinstance(owner, Reader):
        return owner.name or type(owner).__name__
    return getattr(reader, "__name__", "reader")
