"""Shared pytest configuration for the Proofline test suite."""

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Modules read their configuration at import time, so the test defaults
# must be in place before any of them is imported. Real environment
# variables win, which keeps the suite runnable without a .env file.
TEST_ENVIRONMENT = {
    "QDRANT_URL": "http://localhost:6333",
    "QDRANT_API_KEY": "test-api-key",
    "QDRANT_COLLECTION_NAME": "test_chunks",
    "QDRANT_DISTANCE": "cosine",
    "QDRANT_BATCH_SIZE": "100",
    "QDRANT_TOP_K": "5",
    "EMBEDDING_MODEL_NAME": "all-MiniLM-L6-v2",
    "EMBEDDING_DIMENSION": "384",
    "NORMALIZE_EMBEDDINGS": "true",
}
for name, value in TEST_ENVIRONMENT.items():
    os.environ.setdefault(name, value)
