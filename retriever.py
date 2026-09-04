"""Reusable semantic retrieval over the Qdrant chunk collection.

This module packages the end-to-end retrieval path -- embed a question,
search Qdrant, normalize the hits -- behind a stable interface so later
projects can depend on it::

    from retriever import retrieve

    chunks = retrieve("What is a high-risk AI system?", k=5)
    for chunk in chunks:
        print(chunk["id"], chunk["content"])

Every returned record always carries at least ``id`` and ``content``.
"""

import argparse
import json
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
)

from embedder import SentenceEmbedder
from quadrant_client import (
    QDRANT_COLLECTION_NAME,
    QDRANT_TOP_K,
    get_qdrant_client,
    query_points,
)

DEFAULT_TOP_K = QDRANT_TOP_K

# Module-level retriever shared by the public retrieve() function, so the
# embedding model is loaded at most once per process. Access it through
# get_retriever() and reset_default_retriever() rather than directly.
_default_retriever: "Retriever | None" = None


def build_filter(filters: dict[str, Any] | None) -> Filter | None:
    """Translates a plain metadata mapping into a Qdrant payload filter.

    Args:
        filters: Payload field names mapped to the value they must match.
            A list, tuple, or set value matches any of its entries, so
            ``{"topic_label": ["EU_Regulation", "EU_Guidance"]}`` keeps
            chunks carrying either label. Fields that hold arrays, such as
            ``tags``, match when the array contains the given value.

    Returns:
        A Qdrant filter, or None when no filters were supplied.

    Raises:
        ValueError: If a field is mapped to an empty collection of values.
    """
    if not filters:
        return None

    conditions = []
    for key, value in filters.items():
        if isinstance(value, (list, tuple, set)):
            values = list(value)
            if not values:
                raise ValueError(f"Filter {key!r} has no values to match.")
            match = MatchAny(any=values)
        else:
            match = MatchValue(value=value)
        conditions.append(FieldCondition(key=key, match=match))

    return Filter(must=conditions)


def format_chunk(result: dict[str, Any]) -> dict[str, Any]:
    """Normalizes a raw Qdrant hit into a chunk record.

    Args:
        result: A single result returned by
            :func:`quadrant_client.query_points`.

    Returns:
        A chunk record with ``id`` and ``content`` plus the provenance
        fields stored alongside the vector.
    """
    payload = result.get("payload") or {}
    if not isinstance(payload, dict):
        payload = {}
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    point_id = result.get("id")
    content = payload.get("original_text")
    if not isinstance(content, str):
        content = str(result.get("text", ""))

    return {
        "id": payload.get("id", point_id),
        "content": content,
        "score": result.get("score"),
        "source": payload.get("source_filename", "Unknown"),
        "page_number": metadata.get("page_number"),
        "chunk_index": payload.get("chunk_index"),
        "topic_label": payload.get("topic_label", "Unknown"),
        "tags": payload.get("tags", []),
        "point_id": point_id,
        "metadata": metadata,
    }


def format_context(
    chunks: list[dict[str, Any]],
    separator: str = "\n\n",
) -> str:
    """Joins chunk records into a cited context block for prompting.

    Args:
        chunks: Chunk records returned by :func:`retrieve`.
        separator: Text placed between consecutive chunks.

    Returns:
        The chunk contents, each preceded by a numbered source citation.
    """
    blocks = []
    for position, chunk in enumerate(chunks, start=1):
        source = chunk.get("source", "Unknown")
        page_number = chunk.get("page_number")
        citation = f"[{position}] {source}"
        if page_number is not None:
            citation = f"{citation} p.{page_number}"
        blocks.append(f"{citation}\n{chunk.get('content', '')}")
    return separator.join(blocks)


class Retriever:
    """Semantic search over one Qdrant collection.

    The Qdrant client and the embedding model are built on first use and
    then reused, so a single retriever serves many queries without
    reloading the model.

    Attributes:
        collection_name: Name of the Qdrant collection being searched.
    """

    def __init__(
        self,
        client: QdrantClient | None = None,
        model: SentenceEmbedder | None = None,
        collection_name: str = QDRANT_COLLECTION_NAME,
    ) -> None:
        """Configures the retriever.

        Args:
            client: Qdrant client. When omitted, one is built from the
                environment on first use.
            model: Sentence Transformers model used to embed questions.
                When omitted, the configured model is loaded on first use.
            collection_name: Qdrant collection to search.
        """
        self._client = client
        self._model = model
        self.collection_name = collection_name

    def _get_client(self) -> QdrantClient:
        """Returns the Qdrant client, connecting on first call."""
        if self._client is None:
            self._client = get_qdrant_client()
        return self._client

    def _get_model(self) -> SentenceEmbedder:
        """Returns the embedding model, loading it on first call."""
        if self._model is None:
            # Deferred so that importing this module does not pull in
            # Sentence Transformers and its Torch dependency.
            from embedding_generator import create_embedding_model

            self._model = create_embedding_model()
        return self._model

    def embed_query(self, query: str) -> list[float]:
        """Embeds a question with the model used to index the chunks.

        Args:
            query: Question to embed.

        Returns:
            A normalized embedding vector.

        Raises:
            ValueError: If the query is empty or only whitespace.
        """
        if not query.strip():
            raise ValueError("query must not be empty.")

        # Deferred for the same reason as in _get_model().
        from embedding_generator import generate_query_embedding

        return generate_query_embedding(query, self._get_model())

    def retrieve(
        self,
        query: str,
        k: int = DEFAULT_TOP_K,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Returns the k chunks most similar to a question.

        Args:
            query: Natural-language question.
            k: Maximum number of chunks to return.
            filters: Optional payload filters, as described in
                :func:`build_filter`.

        Returns:
            Chunk records ordered by descending similarity. Each record
            contains at least ``id`` and ``content``.

        Raises:
            ValueError: If the query is empty or k is not positive.
        """
        if k <= 0:
            raise ValueError("k must be greater than zero.")

        results = query_points(
            self._get_client(),
            self.embed_query(query),
            collection_name=self.collection_name,
            limit=k,
            query_filter=build_filter(filters),
        )
        return [format_chunk(result) for result in results]


def get_retriever(
    client: QdrantClient | None = None,
    model: SentenceEmbedder | None = None,
    collection_name: str = QDRANT_COLLECTION_NAME,
) -> Retriever:
    """Returns the shared retriever, creating it on first call.

    Args:
        client: Qdrant client used when the shared retriever is created.
        model: Embedding model used when the shared retriever is created.
        collection_name: Collection searched by the shared retriever.

    Returns:
        The process-wide Retriever instance. The arguments are applied
        only when it does not exist yet; call
        :func:`reset_default_retriever` first to rebuild it.
    """
    global _default_retriever
    if _default_retriever is None:
        _default_retriever = Retriever(
            client=client,
            model=model,
            collection_name=collection_name,
        )
    return _default_retriever


def reset_default_retriever() -> None:
    """Discards the shared retriever so the next call rebuilds it."""
    global _default_retriever
    _default_retriever = None


def retrieve(
    query: str,
    k: int = DEFAULT_TOP_K,
    filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Returns the k chunks most similar to a question.

    This is the stable entry point for downstream projects. It reuses one
    shared Retriever, so the embedding model is loaded once per process.

    Args:
        query: Natural-language question.
        k: Maximum number of chunks to return.
        filters: Optional payload filters, as described in
            :func:`build_filter`.

    Returns:
        Chunk records ordered by descending similarity. Each record
        contains at least ``id`` and ``content``.

    Raises:
        ValueError: If the query is empty or k is not positive.
    """
    return get_retriever().retrieve(query, k=k, filters=filters)


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Search the Qdrant chunk collection."
    )
    parser.add_argument("query", help="Question to search for.")
    parser.add_argument(
        "-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of chunks to return.",
    )
    return parser.parse_args()


def main() -> None:
    """Runs one search and prints the chunk records as JSON."""
    args = parse_args()
    chunks = retrieve(args.query, k=args.k)
    print(json.dumps(chunks, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
