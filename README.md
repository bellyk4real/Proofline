# Proofline

A retrieval pipeline over EU legal PDFs: extract pages, split them into
semantic chunks, embed the chunks with `all-MiniLM-L6-v2`, store them in
Qdrant, and search them by meaning.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in your Qdrant URL and API key
```

## Retrieval API

`retriever.py` is the stable entry point for downstream projects. Import
`retrieve` and get back chunk records:

```python
from retriever import retrieve

chunks = retrieve("What counts as a high-risk AI system?", k=5)
for chunk in chunks:
    print(chunk["id"], chunk["score"], chunk["content"][:100])
```

Each record contains at least `id` and `content`, plus provenance:

| Field | Description |
| --- | --- |
| `id` | Stable chunk id, `"<source_filename>:<chunk_index>"` |
| `content` | Full chunk text |
| `score` | Cosine similarity to the query |
| `source` | Source PDF filename |
| `page_number` | Page the chunk came from, when known |
| `chunk_index` | Position of the chunk within its document |
| `topic_label` | Classifier label, e.g. `EU_Regulation` |
| `tags` | Classifier tags |
| `point_id` | Underlying Qdrant point id |
| `metadata` | Chunk metadata recorded at index time |

Narrow a search with payload filters. A list matches any of its values:

```python
retrieve(
    "transparency obligations",
    k=3,
    filters={"topic_label": ["EU_Regulation", "EU_Guidance"]},
)
```

`retrieve` reuses one shared `Retriever`, so the embedding model loads
once per process. For an explicit instance -- a different collection, or
an injected client and model in tests -- construct one directly:

```python
from retriever import Retriever

retriever = Retriever(collection_name="proofline_chunks")
chunks = retriever.retrieve("penalties for non-compliance", k=10)
```

`format_context(chunks)` joins records into a numbered, cited block ready
to drop into a prompt.

You can also query from the command line:

```bash
python retriever.py "What counts as a high-risk AI system?" -k 3
```

## Indexing

| Module | Role |
| --- | --- |
| `pdf_extractor.py` | PDF text and page metadata via PyMuPDF |
| `semantic_chunker.py` | Semantic chunking of extracted pages |
| `embedding_generator.py` | 384-dimensional normalized embeddings |
| `embedder.py` | Batch chunk embedding helpers |
| `quadrant_client.py` | Qdrant collection setup, upsert, and search |
| `retriever.py` | Query-time retrieval interface |

## Tests

```bash
pytest
```

The suite falls back to the defaults in `tests/conftest.py` when the
environment does not define the configuration variables, so it runs
without a `.env` file.
