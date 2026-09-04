from pathlib import Path

import fitz

from pdf_extractor import clean_text, extract_pdf_pages


def _create_pdf(directory: Path) -> Path:
    """Create a two-page PDF fixture with document metadata."""
    path = directory / "sample.pdf"
    with fitz.open() as document:
        document.set_metadata(
            {"title": "Sample document", "author": "Proofline"}
        )

        first_page = document.new_page()
        first_page.insert_text((72, 72), "First page text")
        second_page = document.new_page()
        second_page.insert_text((72, 72), "Second page text")

        document.save(path)
    return path


def test_extracts_text_and_page_metadata(tmp_path: Path) -> None:
    path = _create_pdf(tmp_path)

    pages = extract_pdf_pages(path)

    assert len(pages) == 2
    assert pages[0]["text"].strip() == "First page text"
    assert pages[1]["text"].strip() == "Second page text"
    assert pages[0]["metadata"]["source"] == str(path)
    assert pages[0]["metadata"]["page"] == 0
    assert pages[1]["metadata"]["page"] == 1
    assert pages[0]["metadata"]["page_number"] == 1
    assert pages[1]["metadata"]["page_number"] == 2
    assert pages[1]["metadata"]["total_pages"] == 2


def test_includes_document_metadata_on_each_page(tmp_path: Path) -> None:
    path = _create_pdf(tmp_path)

    pages = extract_pdf_pages(path)

    for page in pages:
        assert page["metadata"]["title"] == "Sample document"
        assert page["metadata"]["author"] == "Proofline"


def test_extracts_blank_page_with_empty_text(tmp_path: Path) -> None:
    path = tmp_path / "blank.pdf"
    with fitz.open() as document:
        document.new_page()
        document.save(path)

    pages = extract_pdf_pages(path)

    assert len(pages) == 1
    assert pages[0]["text"] == ""


def test_clean_text_normalizes_whitespace_and_formatting() -> None:
    raw_text = "  First\n\tline-\r\nwrap.\x00  Second\u00a0line  "

    assert clean_text(raw_text) == "First linewrap. Second line"
