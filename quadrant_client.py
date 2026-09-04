"""Qdrant connection and collection setup utilities."""

import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

load_dotenv()

QDRANT_COLLECTION_NAME = os.environ["QDRANT_COLLECTION_NAME"]
EMBEDDING_DIMENSION = int(os.environ["EMBEDDING_DIMENSION"])
QDRANT_DISTANCE = os.environ["QDRANT_DISTANCE"].upper()

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


if __name__ == "__main__":
    qdrant_client = get_qdrant_client()
    created = ensure_collection(qdrant_client)
    status = "created" if created else "already exists"
    print(f"Collection {QDRANT_COLLECTION_NAME} {status}.")
