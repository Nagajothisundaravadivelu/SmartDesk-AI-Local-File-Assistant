from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pypdf import PdfReader
from pypdf.errors import PdfReadError


class DocumentError(RuntimeError):
    """Raised when a document cannot be loaded or read safely."""


@dataclass
class DocumentContent:
    file_path: str
    file_name: str
    extension: str
    text: str
    char_count: int
    page_count: Optional[int] = None


def _normalize_text(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")).strip()


def _read_txt_file(path: Path) -> DocumentContent:
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
    last_error: Optional[Exception] = None

    for encoding in encodings:
        try:
            raw = path.read_bytes()
            decoded = raw.decode(encoding)
            text = _normalize_text(decoded)
            if not text:
                raise DocumentError("The selected text document is empty.")
            return DocumentContent(
                file_path=str(path),
                file_name=path.name,
                extension=path.suffix.lower(),
                text=text,
                char_count=len(text),
                page_count=None,
            )
        except UnicodeDecodeError as exc:
            last_error = exc
        except OSError as exc:
            raise DocumentError(f"The selected text file could not be read: {exc}") from exc

    raise DocumentError(
        "The selected text document could not be decoded. Please use UTF-8, UTF-16, or another supported plain text format."
    ) from last_error


def _read_pdf_file(path: Path) -> DocumentContent:
    try:
        reader = PdfReader(str(path))
    except FileNotFoundError as exc:
        raise DocumentError("The selected PDF does not exist.") from exc
    except PermissionError as exc:
        raise DocumentError("The selected PDF cannot be opened because of a permissions issue.") from exc
    except PdfReadError as exc:
        raise DocumentError("The selected PDF is corrupted or unreadable.") from exc
    except Exception as exc:
        raise DocumentError("The selected PDF could not be read.") from exc

    try:
        if reader.is_encrypted:
            raise DocumentError("The selected PDF is password-protected and cannot be read without the password.")
    except Exception as exc:
        if isinstance(exc, DocumentError):
            raise
        raise DocumentError("The selected PDF is encrypted or unreadable.") from exc

    pages: list[str] = []
    for page in reader.pages:
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        pages.append(page_text)

    text = _normalize_text("\n\n".join(pages))
    if not text:
        raise DocumentError(
            "The PDF contains no extractable text. For scanned images, OCR has not been implemented in this version."
        )

    return DocumentContent(
        file_path=str(path),
        file_name=path.name,
        extension=path.suffix.lower(),
        text=text,
        char_count=len(text),
        page_count=len(reader.pages),
    )


def load_document(file_path: str) -> DocumentContent:
    path = Path(file_path)
    if not path.exists():
        raise DocumentError("The selected document does not exist.")
    if not path.is_file():
        raise DocumentError("The selected path is not a readable file.")

    suffix = path.suffix.lower()
    if suffix == ".txt":
        return _read_txt_file(path)
    if suffix == ".pdf":
        return _read_pdf_file(path)
    raise DocumentError("Unsupported file type. Please select a PDF or TXT document.")
