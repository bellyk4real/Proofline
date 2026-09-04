from unittest.mock import Mock

import pytest
from qdrant_client.http.models import Distance, FieldCondition, Filter, MatchValue

from quadrant_client import (
    EMBEDDING_DIMENSION,
    classify_chunk,
    display_query_results,
    ensure_collection,
    format_query_results,
    query_points,
    upsert_embedded_chunks,
    verify_collection,
)


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


def test_classifies_ai_act_chunks() -> None:
    topic_label, tags = classify_chunk(
        "This artificial intelligence system is regulated.",
        "eu_ai_act.pdf",
    )

    assert topic_label == "AI_Definition"
    assert tags == ["Legal_Document", "EU_AI_Act"]


def test_upserts_vectors_with_rich_metadata() -> None:
    client = Mock()
    content = "The regulation applies to artificial intelligence systems."
    chunks = [
        {
            "id": "eu_ai_act.pdf:0",
            "source": "eu_ai_act.pdf",
            "chunk_index": 0,
            "content": content,
            "embedding": [0.1] * EMBEDDING_DIMENSION,
            "metadata": {"chunking_method": "semantic", "page_number": 1},
        }
    ]

    count = upsert_embedded_chunks(client, chunks, "test_chunks")

    assert count == 1
    request = client.upsert.call_args.kwargs
    point = request["points"][0]
    payload = point.payload
    assert request["collection_name"] == "test_chunks"
    assert request["wait"] is True
    assert len(point.vector) == EMBEDDING_DIMENSION
    assert payload["source_filename"] == "eu_ai_act.pdf"
    assert payload["original_text"] == content
    assert payload["chunk_index"] == 0
    assert payload["chunk_length"] == len(content)
    assert payload["topic_label"] == "AI_Definition"
    assert payload["tags"] == ["Legal_Document", "EU_AI_Act"]
    assert payload["metadata"]["chunking_method"] == "semantic"
    assert payload["timestamp"].endswith("+00:00")


def test_does_not_upsert_empty_chunk_list() -> None:
    client = Mock()

    assert upsert_embedded_chunks(client, [], "test_chunks") == 0
    client.upsert.assert_not_called()


def test_upserts_points_in_batches() -> None:
    client = Mock()
    chunks = [
        {
            "id": f"sample.pdf:{index}",
            "source": "sample.pdf",
            "chunk_index": index,
            "content": f"Chunk {index}.",
            "embedding": [0.1] * EMBEDDING_DIMENSION,
            "metadata": {},
        }
        for index in range(205)
    ]

    count = upsert_embedded_chunks(client, chunks, "test_chunks")

    assert count == 205
    assert [
        len(call.kwargs["points"]) for call in client.upsert.call_args_list
    ] == [100, 100, 5]
    uploaded_ids = [
        point.payload["id"]
        for call in client.upsert.call_args_list
        for point in call.kwargs["points"]
    ]
    assert uploaded_ids == [chunk["id"] for chunk in chunks]


def test_rejects_non_positive_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        upsert_embedded_chunks(Mock(), [], "test_chunks", batch_size=0)


def test_verifies_collection_configuration_and_point_count() -> None:
    client = Mock()
    collection = Mock()
    collection.config.params.vectors.size = EMBEDDING_DIMENSION
    collection.config.params.vectors.distance = Distance.COSINE
    client.get_collection.return_value = collection
    client.count.return_value.count = 205

    report = verify_collection(client, "test_chunks", expected_point_count=205)

    assert report["vector_size"] == EMBEDDING_DIMENSION
    assert report["distance"] == "Cosine"
    assert report["point_count"] == 205
    assert report["valid"] is True
    client.count.assert_called_once_with("test_chunks", exact=True)


def test_rejects_collection_configuration_mismatch() -> None:
    client = Mock()
    collection = Mock()
    collection.config.params.vectors.size = 768
    collection.config.params.vectors.distance = Distance.COSINE
    client.get_collection.return_value = collection
    client.count.return_value.count = 1

    with pytest.raises(ValueError, match="verification failed"):
        verify_collection(client, "test_chunks", expected_point_count=1)


def test_rejects_point_count_mismatch() -> None:
    client = Mock()
    collection = Mock()
    collection.config.params.vectors.size = EMBEDDING_DIMENSION
    collection.config.params.vectors.distance = Distance.COSINE
    client.get_collection.return_value = collection
    client.count.return_value.count = 4

    with pytest.raises(ValueError, match="verification failed"):
        verify_collection(client, "test_chunks", expected_point_count=5)


def test_queries_points_with_payloads() -> None:
    client = Mock()
    first_result = Mock(
        id="point-1",
        score=0.92,
        payload={"source": "a.pdf", "original_text": "Relevant text."},
    )
    second_result = Mock(id="point-2", score=0.81, payload=None)
    client.query_points.return_value.points = [first_result, second_result]
    query_vector = [0.1] * EMBEDDING_DIMENSION

    results = query_points(client, query_vector, "test_chunks", limit=2)

    assert results == [
        {
            "id": "point-1",
            "score": 0.92,
            "text": "Relevant text.",
            "payload": {
                "source": "a.pdf",
                "original_text": "Relevant text.",
            },
        },
        {"id": "point-2", "score": 0.81, "text": "", "payload": {}},
    ]
    client.query_points.assert_called_once_with(
        collection_name="test_chunks",
        query=query_vector,
        limit=2,
        with_payload=True,
    )


def test_queries_points_filtered_by_topic_label() -> None:
    client = Mock()
    client.query_points.return_value.points = []
    topic_filter = Filter(
        must=[
            FieldCondition(
                key="topic_label",
                match=MatchValue(value="AI_Definition"),
            )
        ]
    )

    query_points(
        client,
        [0.1] * EMBEDDING_DIMENSION,
        "test_chunks",
        limit=3,
        query_filter=topic_filter,
    )

    client.query_points.assert_called_once_with(
        collection_name="test_chunks",
        query=[0.1] * EMBEDDING_DIMENSION,
        limit=3,
        query_filter=topic_filter,
        with_payload=True,
    )


def test_uses_top_five_results_by_default() -> None:
    client = Mock()
    client.query_points.return_value.points = []

    query_points(client, [0.1] * EMBEDDING_DIMENSION, "test_chunks")

    assert client.query_points.call_args.kwargs["limit"] == 5


def test_rejects_empty_query_vector() -> None:
    with pytest.raises(ValueError, match="query_vector"):
        query_points(Mock(), [], "test_chunks")


def test_rejects_non_positive_query_limit() -> None:
    with pytest.raises(ValueError, match="limit"):
        query_points(Mock(), [0.1], "test_chunks", limit=0)


def test_formats_query_results_with_metadata_and_preview() -> None:
    results = [
        {
            "id": "point-1",
            "score": 0.92,
            "text": "A" * 25,
            "payload": {
                "topic_label": "AI_Definition",
                "source_filename": "eu_ai_act.pdf",
            },
        }
    ]

    formatted = format_query_results(results, preview_length=20)

    assert formatted == [
        {
            "rank": 1,
            "id": "point-1",
            "similarity_score": 0.92,
            "topic_label": "AI_Definition",
            "source_filename": "eu_ai_act.pdf",
            "text_preview": f"{'A' * 20}...",
            "payload": results[0]["payload"],
        }
    ]


def test_display_query_results_prints_json(
    capsys: pytest.CaptureFixture[str],
) -> None:
    display_query_results(
        [
            {
                "id": "point-1",
                "score": 0.92,
                "text": "Relevant text.",
                "payload": {"topic_label": "AI_Definition"},
            }
        ]
    )

    output = capsys.readouterr().out
    assert '"similarity_score": 0.92' in output
    assert '"topic_label": "AI_Definition"' in output
    assert '"text_preview": "Relevant text."' in output


def test_rejects_non_positive_preview_length() -> None:
    with pytest.raises(ValueError, match="preview_length"):
        format_query_results([], preview_length=0)
