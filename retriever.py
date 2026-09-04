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
import math
import re
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
HYBRID_FETCH_LIMIT = 50
RRF_K = 60
TOKEN_PATTERN = re.compile(r"\b\w+\b")

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


def _tokenize(text: str) -> list[str]:
    """Return lowercase word tokens for lexical matching."""
    return TOKEN_PATTERN.findall(text.lower())


def _scroll_points(
    client: QdrantClient,
    collection_name: str,
    query_filter: Filter | None,
) -> list[Any]:
    """Read all payload-bearing points from a Qdrant collection."""
    points = []
    offset = None
    while True:
        scroll_kwargs: dict[str, Any] = {
            "collection_name": collection_name,
            "limit": 100,
            "with_payload": True,
            "with_vectors": False,
        }
        if query_filter is not None:
            scroll_kwargs["scroll_filter"] = query_filter
        page, offset = client.scroll(offset=offset, **scroll_kwargs)
        points.extend(page)
        if offset is None:
            return points


def _keyword_score(
    query_terms: list[str],
    documents: list[list[str]],
) -> list[float]:
    """Score documents with a small BM25 implementation."""
    document_count = len(documents)
    if document_count == 0:
        return []
    average_length = sum(map(len, documents)) / document_count or 1
    document_frequency: dict[str, int] = {}
    for terms in documents:
        for term in set(terms):
            document_frequency[term] = document_frequency.get(term, 0) + 1

    scores = []
    for terms in documents:
        term_frequency: dict[str, int] = {}
        for term in terms:
            term_frequency[term] = term_frequency.get(term, 0) + 1
        score = 0.0
        for term in query_terms:
            frequency = term_frequency.get(term, 0)
            if not frequency:
                continue
            inverse_frequency = math.log(
                1
                + (document_count - document_frequency[term] + 0.5)
                / (document_frequency[term] + 0.5)
            )
            normalization = 1.2 * (
                1 - 0.75 + 0.75 * len(terms) / average_length
            )
            score += inverse_frequency * (frequency * 2.2) / (
                frequency + normalization
            )
        scores.append(score)
    return scores


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

    def keyword_search(
        self,
        query: str,
        k: int = DEFAULT_TOP_K,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return chunks ranked by exact keyword matches using BM25."""
        if not query.strip():
            raise ValueError("query must not be empty.")
        if k <= 0:
            raise ValueError("k must be greater than zero.")

        query_terms = _tokenize(query)
        if not query_terms:
            return []
        raw_points = _scroll_points(
            self._get_client(),
            self.collection_name,
            build_filter(filters),
        )
        raw_results = []
        documents = []
        for point in raw_points:
            payload = point.payload or {}
            content = payload.get("original_text", "")
            if not isinstance(content, str):
                continue
            terms = _tokenize(content)
            if any(term in terms for term in query_terms):
                raw_results.append(
                    {
                        "id": point.id,
                        "score": 0.0,
                        "text": content,
                        "payload": payload,
                    }
                )
                documents.append(terms)

        scores = _keyword_score(query_terms, documents)
        ranked_results = [
            (score, result)
            for score, result in zip(scores, raw_results, strict=True)
        ]
        ranked_results.sort(
            key=lambda item: (-item[0], str(item[1]["id"])),
        )
        return [
            format_chunk({**result, "score": score})
            for score, result in ranked_results[:k]
        ]

    def hybrid_search(
        self,
        query: str,
        k: int = DEFAULT_TOP_K,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Fuse semantic and keyword rankings with reciprocal rank fusion."""
        if not query.strip():
            raise ValueError("query must not be empty.")
        if k <= 0:
            raise ValueError("k must be greater than zero.")

        rankings = (
            self.retrieve(query, k=HYBRID_FETCH_LIMIT, filters=filters),
            self.keyword_search(query, k=HYBRID_FETCH_LIMIT, filters=filters),
        )
        fused: dict[Any, dict[str, Any]] = {}
        fused_scores: dict[Any, float] = {}
        for ranking in rankings:
            for rank, chunk in enumerate(ranking, start=1):
                chunk_id = chunk["id"]
                fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + (
                    1 / (RRF_K + rank)
                )
                fused.setdefault(chunk_id, chunk)

        ranked_chunks = sorted(
            fused,
            key=lambda chunk_id: (-fused_scores[chunk_id], str(chunk_id)),
        )
        return [
            {**fused[chunk_id], "score": fused_scores[chunk_id]}
            for chunk_id in ranked_chunks[:k]
        ]


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
    """Return hybrid-ranked chunks for downstream consumers.

    This stable entry point reuses one shared Retriever, so downstream
    projects do not need to know how semantic and keyword rankings are fused.

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
    return get_retriever().hybrid_search(query, k=k, filters=filters)


def semantic_search(
    query: str,
    k: int = DEFAULT_TOP_K,
    filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return chunks ranked by vector similarity only."""
    return get_retriever().retrieve(query, k=k, filters=filters)


def keyword_search(
    query: str,
    k: int = DEFAULT_TOP_K,
    filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return chunks ranked by sparse keyword relevance."""
    return get_retriever().keyword_search(query, k=k, filters=filters)


def hybrid_search(
    query: str,
    k: int = DEFAULT_TOP_K,
    filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return chunks ranked by fused semantic and keyword relevance."""
    return get_retriever().hybrid_search(query, k=k, filters=filters)


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
        "--mode",
        choices=("semantic", "keyword", "hybrid"),
        default="semantic",
        help="Search mode to use.",
    )
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
    searchers = {
        "semantic": semantic_search,
        "keyword": keyword_search,
        "hybrid": hybrid_search,
    }
    search = searchers[args.mode]
    chunks = search(args.query, k=args.k)
    print(json.dumps(chunks, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
