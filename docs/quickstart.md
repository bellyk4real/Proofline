# Proofline Quick Start

Proofline extracts text from EU legal PDFs, splits the text into semantic
chunks, embeds those chunks with `all-MiniLM-L6-v2`, stores them in Qdrant,
and retrieves the most relevant chunks for a natural-language question.

## 1. Install dependencies

Use Python 3.10 or newer and create an isolated environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The first embedding operation downloads the configured Sentence Transformers
model, so it may take a little longer than later operations.

## 2. Configure Qdrant

Create a `.env` file in the project root. Do not commit the API key.

```dotenv
QDRANT_URL=https://your-qdrant-instance.example.com
QDRANT_API_KEY=replace-with-your-api-key
QDRANT_COLLECTION_NAME=proofline_chunks
QDRANT_DISTANCE=cosine
QDRANT_BATCH_SIZE=100
QDRANT_TOP_K=5
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
NORMALIZE_EMBEDDINGS=true
```

`QDRANT_URL` and `QDRANT_API_KEY` are required to connect to Qdrant. The
remaining values must stay consistent with the vectors already stored in the
collection. The application loads `.env` automatically with
`python-dotenv`.

## 3. Extract a PDF

Extracted pages are printed as JSON, including page numbers and document
metadata:

```bash
python main.py data/docs/example.pdf > /tmp/example-pages.json
```

The input PDF must exist at the path supplied to `main.py`.

## 4. Index chunks

The repository exposes indexing as Python functions. The following example
extracts a PDF, creates semantic chunks, generates vectors, creates the
configured collection when necessary, and upserts the chunks:

```python
from pathlib import Path

from embedding_generator import generate_embeddings
from pdf_extractor import extract_pdf_pages
from quadrant_client import (
    ensure_collection,
    get_qdrant_client,
    upsert_embedded_chunks,
)
from semantic_chunker import semantic_chunk_pages

pdf_path = Path("data/docs/example.pdf")
pages = extract_pdf_pages(pdf_path)
chunks = semantic_chunk_pages(pages)
embedded_chunks = generate_embeddings(chunks)

client = get_qdrant_client()
ensure_collection(client)
count = upsert_embedded_chunks(client, embedded_chunks)
print(f"Indexed {count} chunks")
```

For repeatable indexing, save the script as a local file such as
`index_document.py` and run it with the same virtual environment:

```bash
python index_document.py
```

The `data/chunks/` directory contains JSON chunk files that can also be
loaded and passed to `generate_embeddings` when extraction and chunking have
already been completed.

## 5. Search the collection

Once the collection contains indexed chunks, run a semantic search:

```bash
python retriever.py "What counts as a high-risk AI system?" -k 3
```

The command prints JSON records ordered by semantic similarity. Each result
includes a stable chunk `id`, a numeric `score`, the chunk content, source
filename, page number, topic label, and other provenance metadata.

For exact terms, use sparse keyword search. It scans the indexed text payloads
and ranks matching chunks with BM25:

```bash
python retriever.py "Article 5 prohibited practices" --mode keyword -k 3
```

Both modes return the same result shape and are ordered from highest to
lowest score. Keyword search requires Qdrant to be reachable because it reads
the stored text payloads before ranking them locally.

For a combined ranking, use hybrid search. It retrieves 50 candidates from
each method and combines their rankings with reciprocal rank fusion:

```bash
python retriever.py "Article 5 prohibited practices" --mode hybrid -k 5
```

Hybrid scores use `1 / (60 + rank)` for each list, summing contributions for
chunks present in both lists before selecting the top `k` results.

Use the stable hybrid wrapper from Python. It returns chunk objects with
`id`, `content`, and the fused `score` (plus provenance fields):

```python
from retriever import retrieve

chunks = retrieve("transparency obligations", k=5)
for chunk in chunks:
    print(chunk["source"], chunk["page_number"], chunk["score"])
    print(chunk["content"])
```

For vector-only comparisons, call `semantic_search`; `keyword_search` provides
the keyword-only ranking:

```python
from retriever import keyword_search, semantic_search

vector_results = semantic_search("transparency obligations", k=5)
keyword_results = keyword_search("transparency obligations", k=5)
```

Use the keyword API when literal terms matter more than semantic similarity:

```python
from retriever import keyword_search

chunks = keyword_search("Article 5 prohibited practices", k=5)
for chunk in chunks:
    print(chunk["id"], chunk["score"])
```

You can narrow a search by payload metadata:

```python
chunks = retrieve(
    "transparency obligations",
    k=3,
    filters={"topic_label": ["EU_Regulation", "EU_Guidance"]},
)
```

To prepare retrieved text for a prompt, use `format_context(chunks)` from
`retriever.py`.

## 6. Run tests

The test suite supplies local configuration defaults, so it does not require
a `.env` file or a live Qdrant instance:

```bash
pytest
```

The retrieval evaluation uses the configured Qdrant collection and the
question set in `tests/retrieval_questions.json`:

```bash
python evaluate_retrieval.py
```
