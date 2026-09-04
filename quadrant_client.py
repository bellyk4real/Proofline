"""Qdrant connection and collection setup utilities."""

import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

load_dotenv()

QDRANT_COLLECTION_NAME = os.environ["QDRANT_COLLECTION_NAME"]
EMBEDDING_DIMENSION = int(os.environ["EMBEDDING_DIMENSION"])
QDRANT_DISTANCE = os.environ["QDRANT_DISTANCE"].upper()
QDRANT_BATCH_SIZE = int(os.environ["QDRANT_BATCH_SIZE"])


def classify_chunk(content: str, source: str) -> tuple[str, list[str]]:
    """Assign a topic label and tags using document and content keywords.

    Args:
        content: Full chunk text.
        source: Original document filename.

    Returns:
        A topic label and a list of topic/category tags.
    """
    searchable_text = f"{source} {content}".lower()
    tags = ["Legal_Document"]

    if any(
        keyword in searchable_text
        for keyword in ("artificial intelligence", "ai act")
    ):
        tags.append("EU_AI_Act")
        return "AI_Definition", tags
    if any(
        keyword in searchable_text
        for keyword in ("regulation", "commission", "official journal")
    ):
        tags.append("EU_Regulation")
        return "EU_Regulation", tags
    if any(keyword in searchable_text for keyword in ("guideline", "notice")):
        tags.append("EU_Guidance")
        return "EU_Guidance", tags

    tags.append("General_Legal")
    return "General_Legal", tags


def get_qdrant_client() -> QdrantClient:
    """Create a Qdrant client using environment configuration.

    Returns:
        A configured Qdrant client.
    """
    return QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ["QDRANT_API_KEY"],
    )


def ensure_collection(
    client: QdrantClient,
    collection_name: str = QDRANT_COLLECTION_NAME,
) -> bool:
    """Create the embedding collection when it does not already exist.

    Args:
        client: Qdrant client used to inspect and create the collection.
        collection_name: Collection to create or verify.

    Returns:
        ``True`` when a collection was created, otherwise ``False``.

    Raises:
        ValueError: If ``QDRANT_DISTANCE`` is not a supported distance metric.
    """
    if client.collection_exists(collection_name):
        return False

    try:
        distance = Distance[QDRANT_DISTANCE]
    except KeyError as error:
        raise ValueError(
            "QDRANT_DISTANCE must be a supported Qdrant distance metric."
        ) from error

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=EMBEDDING_DIMENSION,
            distance=distance,
        ),
    )
    return True


def verify_collection(
    client: QdrantClient,
    collection_name: str = QDRANT_COLLECTION_NAME,
    expected_point_count: int | None = None,
) -> dict[str, object]:
    """Verify collection vector configuration and point count.

    Args:
        client: Qdrant client used to inspect the collection.
        collection_name: Collection to verify.
        expected_point_count: Optional expected number of stored points.

    Returns:
        A report containing observed values and verification status.

    Raises:
        ValueError: If the vector configuration or point count is invalid.
    """
    collection = client.get_collection(collection_name)
    vectors = collection.config.params.vectors
    vector_size = getattr(vectors, "size", None)
    distance = getattr(vectors, "distance", None)
    observed_distance = getattr(distance, "value", distance)
    expected_distance = Distance[QDRANT_DISTANCE].value
    point_count = client.count(collection_name, exact=True).count

    report = {
        "collection_name": collection_name,
        "vector_size": vector_size,
        "expected_vector_size": EMBEDDING_DIMENSION,
        "distance": observed_distance,
        "expected_distance": expected_distance,
        "point_count": point_count,
        "expected_point_count": expected_point_count,
        "vector_size_valid": vector_size == EMBEDDING_DIMENSION,
        "distance_valid": observed_distance == expected_distance,
        "point_count_valid": (
            expected_point_count is None
            or point_count == expected_point_count
        ),
    }
    report["valid"] = all(
        report[key]
        for key in ("vector_size_valid", "distance_valid", "point_count_valid")
    )

    if not report["valid"]:
        raise ValueError(f"Qdrant collection verification failed: {report}")
    return report


def upsert_embedded_chunks(
    client: QdrantClient,
    chunks: list[dict[str, object]],
    collection_name: str = QDRANT_COLLECTION_NAME,
    batch_size: int = QDRANT_BATCH_SIZE,
) -> int:
    """Store embedded chunks and rich retrieval metadata in Qdrant.

    Args:
        client: Qdrant client used for the upsert operation.
        chunks: Embedded chunks with ``embedding``, ``content``, and metadata.
        collection_name: Target Qdrant collection.
        batch_size: Maximum number of points submitted per request.

    Returns:
        The number of points submitted to Qdrant.

    Raises:
        TypeError: If content, embedding, or metadata has the wrong type.
        ValueError: If an embedding has the configured dimension mismatch or
            ``batch_size`` is not positive.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    timestamp = datetime.now(timezone.utc).isoformat()
    points = []

    for chunk in chunks:
        content = chunk["content"]
        source = str(chunk["source"])
        embedding = chunk["embedding"]
        if not isinstance(content, str) or not isinstance(embedding, list):
            raise TypeError("Chunk content must be text and embedding a list.")
        if len(embedding) != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Expected {EMBEDDING_DIMENSION}-dimensional embedding, "
                f"received {len(embedding)} dimensions."
            )

        topic_label, tags = classify_chunk(content, source)
        chunk_id = str(chunk["id"])
        metadata = chunk.get("metadata", {})
        if not isinstance(metadata, dict):
            raise TypeError("Chunk metadata must be a dictionary.")
        payload = {
            "id": chunk_id,
            "source_filename": Path(source).name,
            "original_text": content,
            "timestamp": timestamp,
            "tags": tags,
            "topic_label": topic_label,
            "chunk_index": chunk["chunk_index"],
            "chunk_length": len(content),
            "metadata": metadata,
        }
        points.append(
            PointStruct(
                id=str(uuid5(NAMESPACE_URL, chunk_id)),
                vector=embedding,
                payload=payload,
            )
        )

    for batch_start in range(0, len(points), batch_size):
        client.upsert(
            collection_name=collection_name,
            points=points[batch_start : batch_start + batch_size],
            wait=True,
        )
    return len(points)


if __name__ == "__main__":
    qdrant_client = get_qdrant_client()
    created = ensure_collection(qdrant_client)
    status = "created" if created else "already exists"
    print(f"Collection {QDRANT_COLLECTION_NAME} {status}.")
