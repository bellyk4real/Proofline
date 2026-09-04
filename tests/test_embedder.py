import math
from typing import Any

import pytest

from embedder import (
    DEFAULT_BATCH_SIZE,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_NAME,
    embed_chunks,
    embed_texts,
)
from semantic_chunker import DEFAULT_EMBEDDING_MODEL


class _FakeSentenceTransformer:
    """Sentence Transformers stand-in that records encode arguments."""

    def __init__(self, dimension: int = EMBEDDING_DIMENSION) -> None:
        self.dimension = dimension
        self.calls: list[dict[str, Any]] = []

    def encode(
        self, sentences: list[str], **kwargs: Any
    ) -> list[list[float]]:
        self.calls.append({"sentences": list(sentences), **kwargs})
        vectors = []

        for index, _ in enumerate(sentences):
            vector = [float(index + 1)] + [1.0] * (self.dimension - 1)
            if kwargs.get("normalize_embeddings"):
                norm = math.sqrt(sum(value**2 for value in vector))
                vector = [value / norm for value in vector]
            vectors.append(vector)

        return vectors


def _chunk(chunk_index: int = 0, content: str = "A semantic chunk.") -> dict:
    return {
        "id": f"sample.pdf:{chunk_index}",
        "source": "sample.pdf",
        "chunk_index": chunk_index,
        "content": content,
        "length": len(content),
        "metadata": {
            "source_filename": "sample.pdf",
            "page_number": 1,
            "chunking_method": "semantic",
        },
    }


def test_embedding_model_matches_the_chunking_model() -> None:
    assert EMBEDDING_MODEL_NAME == "all-MiniLM-L6-v2"
    assert EMBEDDING_MODEL_NAME == DEFAULT_EMBEDDING_MODEL
    assert EMBEDDING_DIMENSION == 384


def test_embeds_each_chunk_with_a_384_dimension_vector() -> None:
    chunks = [_chunk(0), _chunk(1, "Another semantic chunk.")]

    embedded = embed_chunks(chunks, _FakeSentenceTransformer())

    assert len(embedded) == 2
    for chunk in embedded:
        assert len(chunk["embedding"]) == EMBEDDING_DIMENSION
        assert chunk["embedding_dimension"] == EMBEDDING_DIMENSION
        assert chunk["embedding_model"] == EMBEDDING_MODEL_NAME


def test_embeddings_are_normalized_for_cosine_similarity() -> None:
    model = _FakeSentenceTransformer()

    embedded = embed_chunks([_chunk(0), _chunk(1, "Other content.")], model)

    assert model.calls[0]["normalize_embeddings"] is True
    assert model.calls[0]["batch_size"] == DEFAULT_BATCH_SIZE
    for chunk in embedded:
        norm = math.sqrt(sum(value**2 for value in chunk["embedding"]))
        assert norm == pytest.approx(1.0, abs=1e-6)


def test_preserves_chunk_fields_and_metadata() -> None:
    chunks = [_chunk(3)]

    embedded = embed_chunks(chunks, _FakeSentenceTransformer())
    chunk = embedded[0]
    metadata = chunk["metadata"]

    assert chunk["id"] == "sample.pdf:3"
    assert chunk["source"] == "sample.pdf"
    assert chunk["chunk_index"] == 3
    assert chunk["content"] == "A semantic chunk."
    assert chunk["length"] == len(chunk["content"])
    assert metadata["source_filename"] == "sample.pdf"
    assert metadata["page_number"] == 1
    assert metadata["chunking_method"] == "semantic"
    assert metadata["embedding_model"] == EMBEDDING_MODEL_NAME
    assert metadata["embedding_dimension"] == EMBEDDING_DIMENSION


def test_does_not_mutate_the_input_chunks() -> None:
    chunks = [_chunk(0)]

    embed_chunks(chunks, _FakeSentenceTransformer())

    assert "embedding" not in chunks[0]
    assert "embedding_model" not in chunks[0]["metadata"]


def test_embeds_chunks_in_order() -> None:
    model = _FakeSentenceTransformer()
    chunks = [_chunk(0, "First chunk."), _chunk(1, "Second chunk.")]

    embedded = embed_chunks(chunks, model)

    assert model.calls[0]["sentences"] == ["First chunk.", "Second chunk."]
    assert embedded[0]["embedding"] != embedded[1]["embedding"]


def test_batch_size_is_passed_to_the_model() -> None:
    model = _FakeSentenceTransformer()

    embed_chunks([_chunk(0)], model, batch_size=8)

    assert model.calls[0]["batch_size"] == 8


def test_embed_texts_returns_no_vectors_for_no_texts() -> None:
    assert embed_texts([]) == []


def test_rejects_chunks_without_content() -> None:
    chunk = _chunk(0)
    chunk["content"] = "   "

    with pytest.raises(ValueError, match="sample.pdf:0"):
        embed_chunks([chunk], _FakeSentenceTransformer())


def test_rejects_non_dict_chunk_metadata() -> None:
    chunk = _chunk(0)
    chunk["metadata"] = ["not", "a", "dict"]

    with pytest.raises(TypeError, match="metadata"):
        embed_chunks([chunk], _FakeSentenceTransformer())


def test_rejects_vectors_of_mixed_dimensions() -> None:
    class _RaggedModel:
        def encode(
            self, sentences: list[str], **kwargs: Any
        ) -> list[list[float]]:
            return [[1.0, 0.0], [1.0, 0.0, 0.0]]

    with pytest.raises(ValueError, match="mixed dimensions"):
        embed_texts(["first", "second"], _RaggedModel())
