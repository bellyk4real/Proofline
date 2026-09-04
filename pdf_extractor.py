"""Utilities for extracting text and metadata from PDF documents."""

import re
from os import PathLike
from pathlib import Path

import fitz


def clean_text(text: str) -> str:
    """Normalize extracted PDF text for downstream processing.

    Args:
        text: Raw text extracted from a PDF page.

    Returns:
        Text with non-printable characters removed, line-wraps repaired, and
        whitespace collapsed to single spaces.
    """
    normalized_text = re.sub(r"\s+", " ", text)
    printable_text = "".join(
        character for character in normalized_text if character.isprintable()
    )
    repaired_text = re.sub(r"(?<=\w)-\s+(?=\w)", "", printable_text)
    return re.sub(r"\s+", " ", repaired_text).strip()


def extract_pdf_pages(pdf_path: str | PathLike[str]) -> list[dict[str, object]]:
    """Extract PDF text as page records with page-level metadata.

    Args:
        pdf_path: Path to the PDF document to extract.

    Returns:
        A list of page records containing extracted text and metadata.
    """
    path = Path(pdf_path)

    with fitz.open(path) as document:
        document_metadata = dict(document.metadata)
        total_pages = len(document)
        pages = []

        for page_index, page in enumerate(document):
            pages.append(
                {
                    "text": clean_text(page.get_text()),
                    "metadata": {
                        "source": str(path),
                        "page": page_index,
                        "page_number": page_index + 1,
                        "total_pages": total_pages,
                        **document_metadata,
                    },
                }
            )

    return pages
