"""Qdrant connection, storage, and retrieval utilities."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, Filter, PointStruct, VectorParams

load_dotenv()

QDRANT_COLLECTION_NAME = os.environ["QDRANT_COLLECTION_NAME"]
EMBEDDING_DIMENSION = int(os.environ["EMBEDDING_DIMENSION"])
QDRANT_DISTANCE = os.environ["QDRANT_DISTANCE"].upper()
QDRANT_BATCH_SIZE = int(os.environ["QDRANT_BATCH_SIZE"])
QDRANT_TOP_K = int(os.environ["QDRANT_TOP_K"])


def classify_chunk(content: str, source: str) -> tuple[str, list[str]]:
    """Assign a topic label and tags using document and content keywords."""
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
    """Create a Qdrant client using environment configuration."""
    return QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ["QDRANT_API_KEY"],
    )


def ensure_collection(
    client: QdrantClient,
    collection_name: str = QDRANT_COLLECTION_NAME,
) -> bool:
    """Create the configured vector collection when it does not exist."""
    if client.collection_exists(collection_name):
        return False
    try:
        distance = Distance[QDRANT_DISTANCE]
    except KeyError as error:
        raise ValueError("QDRANT_DISTANCE is not supported.") from error
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
    """Verify vector size, distance metric, and point count."""
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
    """Store embedded chunks with rich metadata in Qdrant."""
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
            raise ValueError("Embedding dimension does not match configuration.")
        topic_label, tags = classify_chunk(content, source)
        metadata = chunk.get("metadata", {})
        if not isinstance(metadata, dict):
            raise TypeError("Chunk metadata must be a dictionary.")
        payload = {
            "id": str(chunk["id"]),
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
                id=str(uuid5(NAMESPACE_URL, str(chunk["id"]))),
                vector=embedding,
                payload=payload,
            )
        )
    for start in range(0, len(points), batch_size):
        client.upsert(
            collection_name=collection_name,
            points=points[start : start + batch_size],
            wait=True,
        )
    return len(points)


def query_points(
    client: QdrantClient,
    query_vector: list[float],
    collection_name: str = QDRANT_COLLECTION_NAME,
    limit: int = QDRANT_TOP_K,
    query_filter: Filter | None = None,
) -> list[dict[str, object]]:
    """Retrieve ranked points, optionally narrowed by payload metadata."""
    if not query_vector:
        raise ValueError("query_vector must not be empty.")
    if limit <= 0:
        raise ValueError("limit must be greater than zero.")
    query_kwargs = {
        "collection_name": collection_name,
        "query": query_vector,
        "limit": limit,
        "with_payload": True,
    }
    if query_filter is not None:
        query_kwargs["query_filter"] = query_filter
    response = client.query_points(**query_kwargs)
    results = []
    for result in response.points:
        payload = result.payload or {}
        results.append(
            {
                "id": result.id,
                "score": result.score,
                "text": payload.get("original_text", ""),
                "payload": payload,
            }
        )
    return results


def format_query_results(
    results: list[dict[str, object]],
    preview_length: int = 200,
) -> list[dict[str, object]]:
    """Format retrieval results with scores, labels, and text previews.

    Args:
        results: Results returned by :func:`query_points`.
        preview_length: Maximum number of characters in each text preview.

    Returns:
        Display-ready result records with rank and selected metadata.

    Raises:
        ValueError: If ``preview_length`` is not positive.
    """
    if preview_length <= 0:
        raise ValueError("preview_length must be greater than zero.")

    formatted_results = []
    for rank, result in enumerate(results, start=1):
        payload = result.get("payload") or {}
        if not isinstance(payload, dict):
            payload = {}
        text = str(result.get("text", ""))
        preview = text[:preview_length]
        if len(text) > preview_length:
            preview = f"{preview}..."
        formatted_results.append(
            {
                "rank": rank,
                "id": result.get("id"),
                "similarity_score": result.get("score"),
                "topic_label": payload.get("topic_label", "Unknown"),
                "source_filename": payload.get("source_filename", "Unknown"),
                "text_preview": preview,
                "payload": payload,
            }
        )
    return formatted_results


def display_query_results(
    results: list[dict[str, object]],
    preview_length: int = 200,
) -> None:
    """Print formatted retrieval results as indented JSON."""
    print(json.dumps(format_query_results(results, preview_length), indent=2))


if __name__ == "__main__":
    qdrant_client = get_qdrant_client()
    created = ensure_collection(qdrant_client)
    status = "created" if created else "already exists"
    print(f"Collection {QDRANT_COLLECTION_NAME} {status}.")
