"""Run the retrieval evaluation questions against the configured Qdrant collection."""

import argparse
import json
from pathlib import Path
from typing import Any

from qdrant_client.http.models import FieldCondition, Filter, MatchValue

from embedding_generator import generate_query_embedding
from quadrant_client import (
    QDRANT_COLLECTION_NAME,
    QDRANT_TOP_K,
    display_query_results,
    get_qdrant_client,
    query_points,
)

DEFAULT_QUESTIONS_PATH = Path("tests/retrieval_questions.json")
PREVIEW_LENGTH = 200


def parse_args() -> argparse.Namespace:
    """Parse evaluator options."""
    parser = argparse.ArgumentParser(
        description="Evaluate semantic retrieval against the question set."
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=DEFAULT_QUESTIONS_PATH,
        help="Path to the retrieval question JSON file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=QDRANT_TOP_K,
        help="Number of results to retrieve for each question.",
    )
    parser.add_argument(
        "--source",
        help="Restrict every query to a source_filename metadata value.",
    )
    return parser.parse_args()


def load_questions(path: Path) -> list[dict[str, Any]]:
    """Load and validate the retrieval question records."""
    with path.open(encoding="utf-8") as question_file:
        questions = json.load(question_file)
    if not isinstance(questions, list):
        raise ValueError("The question file must contain a JSON array.")
    return questions


def inspect_results(
    question: dict[str, Any],
    results: list[dict[str, object]],
) -> None:
    """Print relevance checks for scores, metadata alignment, and context."""
    expected_source = question["expected_source_filename"]
    expected_topic = question["expected_topic_label"]
    scores = [result.get("score") for result in results]
    scores_ordered = all(
        isinstance(scores[index], (int, float))
        and isinstance(scores[index + 1], (int, float))
        and scores[index] >= scores[index + 1]
        for index in range(len(scores) - 1)
    )
    print(f"Scores ordered descending: {'yes' if scores_ordered else 'no'}")
    for rank, result in enumerate(results, start=1):
        payload = result.get("payload") or {}
        text = str(result.get("text", ""))
        topic_match = payload.get("topic_label") == expected_topic
        source_match = payload.get("source_filename") == expected_source
        if not text.strip():
            content_quality = "empty"
        elif len(text) < 100:
            content_quality = "short"
        else:
            content_quality = "usable"
        context = (
            "preview truncated"
            if len(text) > PREVIEW_LENGTH
            else "preview contains full chunk"
        )
        print(
            f"  #{rank} score={result.get('score')} "
            f"topic={'aligned' if topic_match else 'off-target'} "
            f"source={'matched' if source_match else 'different'} "
            f"content={content_quality}; context={context}"
        )


def main() -> None:
    """Run each question and print ranked results plus a hit-rate summary."""
    args = parse_args()
    questions = load_questions(args.questions)
    client = get_qdrant_client()
    query_filter = None
    if args.source:
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="source_filename",
                    match=MatchValue(value=args.source),
                )
            ]
        )

    hits = 0
    for question in questions:
        question_text = question["question"]
        results = query_points(
            client,
            generate_query_embedding(question_text),
            QDRANT_COLLECTION_NAME,
            limit=args.limit,
            query_filter=query_filter,
        )
        source = question["expected_source_filename"]
        retrieved_sources = [
            result["payload"].get("source_filename") for result in results
        ]
        hit = source in retrieved_sources
        hits += int(hit)
        print(f"\n[{question['id']}] {question_text}")
        print(f"Expected source: {source}")
        print(f"Source retrieved: {'yes' if hit else 'no'}")
        inspect_results(question, results)
        display_query_results(results, preview_length=PREVIEW_LENGTH)

    print(f"\nRetrieval hit rate: {hits}/{len(questions)} ({hits / len(questions):.0%})")


if __name__ == "__main__":
    main()
