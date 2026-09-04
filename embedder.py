"""Utilities for embedding semantic chunks with Sentence Transformers."""

from collections.abc import Sequence
from typing import Any, Protocol

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
DEFAULT_BATCH_SIZE = 32


class SentenceEmbedder(Protocol):
    """Protocol for Sentence Transformers-compatible embedding models."""

    def encode(self, sentences: list[str], **kwargs: Any) -> Any:
        """Encode sentences into embedding vectors."""


def load_embedding_model(
    model_name: str = EMBEDDING_MODEL_NAME,
) -> SentenceEmbedder:
    """Load the Sentence Transformers model used for chunk embeddings.

    Args:
        model_name: Sentence Transformers model name. The default model emits
            384-dimensional vectors and ends with a normalization layer, so
            its outputs are already unit length for cosine similarity.

    Returns:
        A loaded Sentence Transformers model.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed_texts(
    texts: Sequence[str],
    model: SentenceEmbedder | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[list[float]]:
    """Embed texts as normalized vectors for cosine-similarity search.

    Args:
        texts: Texts to embed, in order.
        model: Embedding model. When omitted, ``all-MiniLM-L6-v2`` is loaded
            automatically.
        batch_size: Number of texts encoded per forward pass.

    Returns:
        One vector per input text, in input order.

    Raises:
        ValueError: If the model returns vectors of differing dimensions.
    """
    if not texts:
        return []

    encoder = model if model is not None else load_embedding_model()
    vectors = encoder.encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=True,
    )
    embeddings = [[float(value) for value in vector] for vector in vectors]

    if len({len(embedding) for embedding in embeddings}) > 1:
        raise ValueError(
            "Embedding model returned vectors of mixed dimensions."
        )

    return embeddings


def embed_chunks(
    chunks: list[dict[str, Any]],
    model: SentenceEmbedder | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    model_name: str = EMBEDDING_MODEL_NAME,
) -> list[dict[str, Any]]:
    """Attach a normalized embedding vector to each semantic chunk.

    Args:
        chunks: Chunk records returned by ``semantic_chunk_pages``.
        model: Embedding model. When omitted, ``all-MiniLM-L6-v2`` is loaded
            automatically.
        batch_size: Number of chunks encoded per forward pass.
        model_name: Model name recorded on each chunk for provenance.

    Returns:
        New chunk records carrying ``embedding``, ``embedding_model``, and
        ``embedding_dimension`` alongside the original chunk fields.

    Raises:
        TypeError: If a chunk's metadata is not a dictionary.
        ValueError: If a chunk has no text content to embed.
    """
    contents = []

    for chunk in chunks:
        content = chunk.get("content")
        if not isinstance(content, str) or not content.strip():
            chunk_id = chunk.get("id", "<unknown>")
            raise ValueError(
                f"Chunk {chunk_id!r} has no text content to embed."
            )
        if not isinstance(chunk.get("metadata", {}), dict):
            raise TypeError("Chunk metadata must be a dictionary.")
        contents.append(content)

    embeddings = embed_texts(contents, model, batch_size)
    embedded_chunks = []

    for chunk, embedding in zip(chunks, embeddings, strict=True):
        metadata = {
            **chunk.get("metadata", {}),
            "embedding_model": model_name,
            "embedding_dimension": len(embedding),
        }
        embedded_chunks.append(
            {
                **chunk,
                "metadata": metadata,
                "embedding": embedding,
                "embedding_model": model_name,
                "embedding_dimension": len(embedding),
            }
        )

    return embedded_chunks
