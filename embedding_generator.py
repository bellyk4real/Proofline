"""Generate normalized embeddings for structured semantic chunks."""

import os
from typing import Any

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

MODEL_NAME = os.environ["EMBEDDING_MODEL_NAME"]
EMBEDDING_DIMENSION = int(os.environ["EMBEDDING_DIMENSION"])
NORMALIZE_EMBEDDINGS = os.environ["NORMALIZE_EMBEDDINGS"].lower() == "true"


def create_embedding_model(
    model_name: str = MODEL_NAME,
) -> SentenceTransformer:
    """Load a Sentence Transformers embedding model.

    Args:
        model_name: Sentence Transformers model name.

    Returns:
        A loaded Sentence Transformer model.
    """
    return SentenceTransformer(model_name)


def generate_embeddings(
    chunks: list[dict[str, Any]],
    model: SentenceTransformer | None = None,
) -> list[dict[str, Any]]:
    """Generate normalized 384-dimensional vectors for each chunk.

    Args:
        chunks: Structured semantic chunks containing a ``content`` field.
        model: Optional loaded model. The default model is loaded when omitted.

    Returns:
        Copies of the chunks with an ``embedding`` vector added to each one.

    Raises:
        KeyError: If a chunk does not contain ``content``.
        ValueError: If the model returns vectors with an unexpected dimension.
    """
    embedding_model = model or create_embedding_model()
    texts = [chunk["content"] for chunk in chunks]
    vectors = embedding_model.encode(
        texts,
        normalize_embeddings=NORMALIZE_EMBEDDINGS,
        convert_to_numpy=True,
    )
    if len(vectors) != len(chunks):
        raise ValueError(
            "Embedding count does not match chunk count: "
            f"received {len(vectors)} embeddings for {len(chunks)} chunks."
        )

    embedded_chunks = []
    for chunk, vector in zip(chunks, vectors):
        if len(vector) != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Expected {EMBEDDING_DIMENSION}-dimensional embedding, "
                f"received {len(vector)} dimensions."
            )
        embedded_chunk = dict(chunk)
        embedded_chunk["embedding"] = vector.tolist()
        embedded_chunks.append(embedded_chunk)

    return embedded_chunks
