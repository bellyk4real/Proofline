from unittest.mock import Mock

import pytest
from qdrant_client.http.models import Distance

from quadrant_client import EMBEDDING_DIMENSION, ensure_collection


def test_creates_collection_with_embedding_configuration() -> None:
    client = Mock()
    client.collection_exists.return_value = False

    created = ensure_collection(client, "test_chunks")

    assert created is True
    client.create_collection.assert_called_once()
    request = client.create_collection.call_args.kwargs
    assert request["collection_name"] == "test_chunks"
    assert request["vectors_config"].size == EMBEDDING_DIMENSION
    assert request["vectors_config"].distance == Distance.COSINE


def test_does_not_recreate_existing_collection() -> None:
    client = Mock()
    client.collection_exists.return_value = True

    created = ensure_collection(client, "test_chunks")

    assert created is False
    client.create_collection.assert_not_called()


def test_rejects_unsupported_distance(monkeypatch: pytest.MonkeyPatch) -> None:
    client = Mock()
    client.collection_exists.return_value = False
    monkeypatch.setattr("quadrant_client.QDRANT_DISTANCE", "INVALID")

    with pytest.raises(ValueError, match="QDRANT_DISTANCE"):
        ensure_collection(client, "test_chunks")
