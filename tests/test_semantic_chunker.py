import json
from typing import Any

import pytest

from semantic_chunker import (
    DEFAULT_EMBEDDING_MODEL,
    chunks_to_json,
    create_embeddings,
    save_chunks_json,
    semantic_chunk_pages,
)


class _FakeEmbeddings:
    """Embedding fixture with deterministic vectors for semantic chunking."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(index), 1.0] for index, _ in enumerate(texts)]

    def embed_query(self, text: str) -> list[float]:
        return [0.0, 1.0]


def test_semantic_chunking_preserves_page_metadata() -> None:
    pages: list[dict[str, Any]] = [
        {
            "text": (
                "First topic has supporting context. "
                "It remains related to the first topic. "
                "Second topic introduces a different subject."
            ),
            "metadata": {"source": "sample.pdf", "page_number": 2},
        }
    ]

    chunks = semantic_chunk_pages(
        pages,
        _FakeEmbeddings(),
        breakpoint_threshold_type="gradient",
    )

    assert chunks
    assert all(chunk["source"] == "sample.pdf" for chunk in chunks)
    assert all(chunk["metadata"]["source"] == "sample.pdf" for chunk in chunks)
    assert all(chunk["metadata"]["page_number"] == 2 for chunk in chunks)
    assert all(chunk["content"].strip() for chunk in chunks)


def test_semantic_chunks_include_structured_metadata() -> None:
    pages: list[dict[str, Any]] = [
        {
            "text": "A document page with meaningful content.",
            "metadata": {"source": "/docs/sample.pdf", "page_number": 1},
        }
    ]

    chunks = semantic_chunk_pages(pages, _FakeEmbeddings())
    chunk = chunks[0]
    metadata = chunk["metadata"]

    assert chunk["id"] == "sample.pdf:0"
    assert chunk["source"] == "sample.pdf"
    assert chunk["chunk_index"] == 0
    assert chunk["length"] == len(chunk["content"])
    assert metadata["source_filename"] == "sample.pdf"
    assert metadata["chunking_method"] == "semantic"


@pytest.mark.parametrize(
    "threshold_type", ["percentile", "standard_deviation", "gradient"]
)
def test_supported_breakpoint_threshold_types(threshold_type: str) -> None:
    pages = [
        {
            "text": (
                "One sentence. Another sentence. "
                "A third sentence. A fourth sentence."
            ),
            "metadata": {},
        }
    ]

    chunks = semantic_chunk_pages(
        pages,
        _FakeEmbeddings(),
        breakpoint_threshold_type=threshold_type,
    )

    assert chunks


def test_rejects_unsupported_breakpoint_threshold_type() -> None:
    with pytest.raises(ValueError, match="breakpoint_threshold_type"):
        semantic_chunk_pages(
            [],
            _FakeEmbeddings(),
            breakpoint_threshold_type="fixed",  # type: ignore[arg-type]
        )


def test_default_embedding_model_name() -> None:
    assert DEFAULT_EMBEDDING_MODEL == "all-MiniLM-L6-v2"
    assert create_embeddings.__defaults__ == (DEFAULT_EMBEDDING_MODEL,)


def test_chunks_can_be_serialized_and_persisted(tmp_path) -> None:
    chunks = [
        {
            "id": "sample.pdf:0",
            "source": "sample.pdf",
            "chunk_index": 0,
            "content": "A semantic chunk.",
            "length": 17,
            "metadata": {
                "source_filename": "sample.pdf",
                "chunking_method": "semantic",
            },
        }
    ]

    output_path = save_chunks_json(chunks, tmp_path / "chunks.json")

    assert output_path.exists()
    assert json.loads(output_path.read_text(encoding="utf-8")) == chunks
    assert json.loads(chunks_to_json(chunks)) == chunks