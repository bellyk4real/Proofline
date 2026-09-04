"""Tests for the reusable retrieval interface."""

from collections.abc import Iterator
from unittest.mock import Mock

import numpy as np
import pytest
from qdrant_client.http.models import (
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
)

import retriever
from quadrant_client import EMBEDDING_DIMENSION
from retriever import (
    DEFAULT_TOP_K,
    Retriever,
    build_filter,
    format_chunk,
    format_context,
    get_retriever,
    reset_default_retriever,
    retrieve,
)

EMBEDDING = [1.0] * EMBEDDING_DIMENSION


def make_model() -> Mock:
    """Returns a stand-in embedding model that encodes to a fixed vector.

    Returns:
        A mock whose ``encode`` method returns a normalized vector of the
        configured dimension.
    """
    model = Mock()
    model.encode.return_value = np.ones(EMBEDDING_DIMENSION, dtype=np.float32)
    return model


def make_hit(
    point_id: str = "point-1",
    score: float = 0.92,
    chunk_id: str = "sample.pdf:0",
) -> Mock:
    """Returns a Qdrant point carrying the payload written at index time.

    Args:
        point_id: Identifier of the stored Qdrant point.
        score: Similarity score reported for the point.
        chunk_id: Stable chunk identifier recorded in the payload.

    Returns:
        A mock shaped like a scored point returned by Qdrant.
    """
    return Mock(
        id=point_id,
        score=score,
        payload={
            "id": chunk_id,
            "source_filename": "sample.pdf",
            "original_text": "Relevant text.",
            "topic_label": "EU_Regulation",
            "tags": ["Legal_Document", "EU_Regulation"],
            "chunk_index": 0,
            "metadata": {"page_number": 3},
        },
    )


@pytest.fixture(autouse=True)
def reset_shared_retriever() -> Iterator[None]:
    """Keeps the module-level retriever from leaking between tests."""
    reset_default_retriever()
    yield
    reset_default_retriever()


def test_retrieves_chunk_records_with_id_and_content() -> None:
    client = Mock()
    client.query_points.return_value.points = [make_hit()]

    chunks = Retriever(client, make_model()).retrieve("What is an AI system?")

    assert chunks == [
        {
            "id": "sample.pdf:0",
            "content": "Relevant text.",
            "score": 0.92,
            "source": "sample.pdf",
            "page_number": 3,
            "chunk_index": 0,
            "topic_label": "EU_Regulation",
            "tags": ["Legal_Document", "EU_Regulation"],
            "point_id": "point-1",
            "metadata": {"page_number": 3},
        }
    ]


def test_searches_configured_collection_with_query_embedding() -> None:
    client = Mock()
    client.query_points.return_value.points = []
    model = make_model()

    Retriever(client, model, "custom_chunks").retrieve("query text", k=3)

    model.encode.assert_called_once()
    assert model.encode.call_args.args[0] == "query text"
    client.query_points.assert_called_once_with(
        collection_name="custom_chunks",
        query=EMBEDDING,
        limit=3,
        with_payload=True,
    )


def test_defaults_to_configured_top_k() -> None:
    client = Mock()
    client.query_points.return_value.points = []

    Retriever(client, make_model()).retrieve("query text")

    assert client.query_points.call_args.kwargs["limit"] == DEFAULT_TOP_K


def test_retrieves_with_metadata_filters() -> None:
    client = Mock()
    client.query_points.return_value.points = []

    Retriever(client, make_model()).retrieve(
        "query text",
        k=2,
        filters={"source_filename": "sample.pdf"},
    )

    assert client.query_points.call_args.kwargs["query_filter"] == Filter(
        must=[
            FieldCondition(
                key="source_filename",
                match=MatchValue(value="sample.pdf"),
            )
        ]
    )


def test_rejects_empty_query() -> None:
    with pytest.raises(ValueError, match="query"):
        Retriever(Mock(), make_model()).retrieve("   ")


def test_rejects_non_positive_k() -> None:
    with pytest.raises(ValueError, match="k must be"):
        Retriever(Mock(), make_model()).retrieve("query text", k=0)


def test_builds_filter_matching_any_of_several_values() -> None:
    query_filter = build_filter(
        {"topic_label": ["EU_Regulation", "EU_Guidance"]}
    )

    assert query_filter == Filter(
        must=[
            FieldCondition(
                key="topic_label",
                match=MatchAny(any=["EU_Regulation", "EU_Guidance"]),
            )
        ]
    )


def test_builds_no_filter_without_criteria() -> None:
    assert build_filter(None) is None
    assert build_filter({}) is None


def test_rejects_filter_without_values() -> None:
    with pytest.raises(ValueError, match="tags"):
        build_filter({"tags": []})


def test_formats_chunk_from_hit_without_payload() -> None:
    chunk = format_chunk({"id": "point-2", "score": 0.4, "payload": None})

    assert chunk["id"] == "point-2"
    assert chunk["content"] == ""
    assert chunk["source"] == "Unknown"
    assert chunk["page_number"] is None
    assert chunk["tags"] == []


def test_shared_retriever_is_reused_across_calls() -> None:
    client = Mock()
    client.query_points.return_value.points = []
    get_retriever(client=client, model=make_model())

    retrieve("first question", k=1)
    retrieve("second question", k=1)

    assert get_retriever() is get_retriever()
    assert client.query_points.call_count == 2


def test_loads_embedding_model_once(monkeypatch: pytest.MonkeyPatch) -> None:
    client = Mock()
    client.query_points.return_value.points = []
    create_model = Mock(return_value=make_model())
    monkeypatch.setattr(
        "embedding_generator.create_embedding_model",
        create_model,
    )
    instance = Retriever(client)

    instance.retrieve("first question")
    instance.retrieve("second question")

    create_model.assert_called_once()


def test_builds_qdrant_client_on_first_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = Mock()
    client.query_points.return_value.points = []
    get_client = Mock(return_value=client)
    monkeypatch.setattr(retriever, "get_qdrant_client", get_client)
    instance = Retriever(model=make_model())

    instance.retrieve("first question")
    instance.retrieve("second question")

    get_client.assert_called_once()


def test_formats_context_with_numbered_citations() -> None:
    chunks = [
        {"content": "First.", "source": "a.pdf", "page_number": 1},
        {"content": "Second.", "source": "b.pdf", "page_number": None},
    ]

    context = format_context(chunks)

    assert context == "[1] a.pdf p.1\nFirst.\n\n[2] b.pdf\nSecond."
