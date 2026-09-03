"""Utilities for splitting extracted PDF pages by semantic boundaries."""

import json
from pathlib import Path
from typing import Any, Literal

from langchain_core.embeddings import Embeddings
from langchain_experimental.text_splitter import SemanticChunker

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BreakpointThresholdType = Literal[
    "percentile", "standard_deviation", "gradient"
]


def create_embeddings(model_name: str = DEFAULT_EMBEDDING_MODEL) -> Embeddings:
    """Create the Hugging Face embedding model used for semantic chunking.

    Args:
        model_name: Sentence Transformers model name.

    Returns:
        A LangChain-compatible embedding model.
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=model_name)


def semantic_chunk_pages(
    pages: list[dict[str, Any]],
    embeddings: Embeddings | None = None,
    breakpoint_threshold_type: BreakpointThresholdType = "percentile",
) -> list[dict[str, Any]]:
    """Split extracted pages into semantically coherent chunks.

    Args:
        pages: Page records returned by ``extract_pdf_pages``.
        embeddings: Embedding model used to compare sentence meaning. When
            omitted, ``all-MiniLM-L6-v2`` is loaded automatically.
        breakpoint_threshold_type: Semantic boundary method. Supported values
            are ``percentile``, ``standard_deviation``, and ``gradient``.

    Returns:
        Chunk records containing content and structured chunk metadata.

    Raises:
        TypeError: If a page's metadata is not a dictionary.
        ValueError: If the breakpoint threshold type is unsupported.
    """
    supported_thresholds = {
        "percentile",
        "standard_deviation",
        "gradient",
    }
    if breakpoint_threshold_type not in supported_thresholds:
        raise ValueError(
            "breakpoint_threshold_type must be one of: "
            "percentile, standard_deviation, gradient"
        )

    chunker = SemanticChunker(
        embeddings or create_embeddings(),
        breakpoint_threshold_type=breakpoint_threshold_type,
    )
    chunks = []
    chunk_index = 0

    for page in pages:
        text = page["text"]
        metadata = page["metadata"]
        if not isinstance(text, str) or not text.strip():
            continue
        if not isinstance(metadata, dict):
            raise TypeError("Page metadata must be a dictionary.")

        source_filename = Path(str(metadata.get("source", "unknown"))).name
        source_filename = source_filename or "unknown"

        for document in chunker.create_documents([text], metadatas=[metadata]):
            content = document.page_content
            chunk_metadata = {
                **document.metadata,
                "chunking_method": "semantic",
                "source_filename": source_filename,
                "source": document.metadata.get("source", source_filename),
                "page_number": document.metadata.get("page_number"),
            }
            chunks.append(
                {
                    "id": f"{source_filename}:{chunk_index}",
                    "source": source_filename,
                    "chunk_index": chunk_index,
                    "content": content,
                    "length": len(content),
                    "metadata": chunk_metadata,
                }
            )
            chunk_index += 1

    return chunks


def chunks_to_json(chunks: list[dict[str, Any]]) -> str:
    """Serialize structured chunks as indented JSON.

    Args:
        chunks: Chunk records returned by :func:`semantic_chunk_pages`.

    Returns:
        A JSON string suitable for persistence or transmission.
    """
    return json.dumps(chunks, indent=2, ensure_ascii=False)


def save_chunks_json(
    chunks: list[dict[str, Any]], output_path: str | Path
) -> Path:
    """Persist structured chunks to a JSON file.

    Args:
        chunks: Chunk records returned by :func:`semantic_chunk_pages`.
        output_path: Destination path for the JSON file.

    Returns:
        The path written to disk.
    """
    path = Path(output_path)
    path.write_text(chunks_to_json(chunks) + "\n", encoding="utf-8")
    return path
