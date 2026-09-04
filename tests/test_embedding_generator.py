from typing import Any

import numpy as np
import pytest

from embedding_generator import (
    EMBEDDING_DIMENSION,
    MODEL_NAME,
    create_embedding_model,
    generate_embeddings,
)


class _FakeModel:
    """Embedding model fixture that records encode options."""

    def __init__(self, dimension: int = EMBEDDING_DIMENSION) -> None:
        self.dimension = dimension
        self.encode_kwargs: dict[str, Any] = {}

    def encode(self, texts: list[str], **kwargs: Any) -> np.ndarray:
        self.encode_kwargs = kwargs
        return np.ones((len(texts), self.dimension), dtype=np.float32)


def test_default_model_configuration() -> None:
    assert MODEL_NAME == "all-MiniLM-L6-v2"
    assert EMBEDDING_DIMENSION == 384
    assert create_embedding_model.__defaults__ == (MODEL_NAME,)


def test_generates_normalized_384_dimension_vectors() -> None:
    model = _FakeModel()
    chunks = [{"id": "sample.pdf:0", "content": "A semantic chunk."}]

    embedded_chunks = generate_embeddings(chunks, model)

    assert len(embedded_chunks[0]["embedding"]) == EMBEDDING_DIMENSION
    assert embedded_chunks[0]["id"] == chunks[0]["id"]
    assert embedded_chunks[0]["content"] == chunks[0]["content"]
    assert model.encode_kwargs["normalize_embeddings"] is True
    assert model.encode_kwargs["convert_to_numpy"] is True


def test_generates_one_vector_per_chunk() -> None:
    model = _FakeModel()
    chunks = [
        {"id": "sample.pdf:0", "content": "First chunk."},
        {"id": "sample.pdf:1", "content": "Second chunk."},
    ]

    embedded_chunks = generate_embeddings(chunks, model)

    assert len(embedded_chunks) == len(chunks)
    assert all(
        len(chunk["embedding"]) == EMBEDDING_DIMENSION
        for chunk in embedded_chunks
    )


def test_rejects_unexpected_embedding_dimension() -> None:
    model = _FakeModel(dimension=3)
    chunks = [{"content": "A semantic chunk."}]

    with pytest.raises(ValueError, match="384-dimensional"):
        generate_embeddings(chunks, model)
