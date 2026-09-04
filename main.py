"""Command-line entry point for PDF text extraction."""

import argparse
import json
from pathlib import Path

from pdf_extractor import extract_pdf_pages


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Extract text and page metadata from a PDF."
    )
    parser.add_argument("pdf_path", type=Path, help="Path to the PDF document.")
    return parser.parse_args()


def main() -> None:
    """Extract a PDF and print the page records as JSON."""
    args = parse_args()
    pages = extract_pdf_pages(args.pdf_path)
    print(json.dumps(pages, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
