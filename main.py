"""Command-line entry point for indexing PDF chunks in Qdrant."""

import argparse
import json
from pathlib import Path

from embedding_generator import generate_embeddings
from pdf_extractor import extract_pdf_pages
from quadrant_client import (
    QDRANT_BATCH_SIZE,
    QDRANT_COLLECTION_NAME,
    ensure_collection,
    get_qdrant_client,
    upsert_embedded_chunks,
    verify_collection,
)
from semantic_chunker import semantic_chunk_pages


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate PDF embeddings and upload them to Qdrant."
    )
    parser.add_argument(
        "pdf_paths",
        nargs="*",
        type=Path,
        help="PDF paths. Defaults to every PDF in data/docs.",
    )
    parser.add_argument(
        "--collection",
        default=QDRANT_COLLECTION_NAME,
        help="Qdrant collection name.",
    )
    parser.add_argument(
        "--batch-size",
        default=QDRANT_BATCH_SIZE,
        type=int,
        help="Maximum points per Qdrant upload request.",
    )
    return parser.parse_args()


def discover_pdfs(pdf_paths: list[Path]) -> list[Path]:
    """Resolve explicit PDF paths or discover PDFs in the default directory.

    Args:
        pdf_paths: Explicit PDF paths supplied by the caller.

    Returns:
        Sorted PDF paths to process.

    Raises:
        FileNotFoundError: If an explicit path does not exist.
    """
    if pdf_paths:
        missing_paths = [path for path in pdf_paths if not path.is_file()]
        if missing_paths:
            raise FileNotFoundError(f"PDF not found: {missing_paths[0]}")
        return pdf_paths
    return sorted(Path("data/docs").glob("*.pdf"))


def build_chunks(pdf_paths: list[Path]) -> list[dict[str, object]]:
    """Extract and semantically chunk all supplied PDF documents.

    Args:
        pdf_paths: PDF documents to process.

    Returns:
        Structured semantic chunks ready for embedding.
    """
    pages = [page for path in pdf_paths for page in extract_pdf_pages(path)]
    return semantic_chunk_pages(pages)


def main() -> None:
    """Generate, upload, and validate embeddings for PDF documents."""
    args = parse_args()
    pdf_paths = discover_pdfs(args.pdf_paths)
    if not pdf_paths:
        raise FileNotFoundError("No PDF documents found in data/docs.")

    chunks = build_chunks(pdf_paths)
    embedded_chunks = generate_embeddings(chunks)
    if len(embedded_chunks) != len(chunks):
        raise ValueError("Embedding count does not match chunk count.")

    client = get_qdrant_client()
    ensure_collection(client, args.collection)
    uploaded_count = upsert_embedded_chunks(
        client,
        embedded_chunks,
        collection_name=args.collection,
        batch_size=args.batch_size,
    )
    report = verify_collection(
        client,
        collection_name=args.collection,
        expected_point_count=uploaded_count,
    )
    print(
        json.dumps(
            {
                "documents": len(pdf_paths),
                "chunks": len(chunks),
                "embeddings": len(embedded_chunks),
                "uploaded_points": uploaded_count,
                "collection": report,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
