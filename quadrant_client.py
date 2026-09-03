import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()  # loads QDRANT_URL and QDRANT_API_KEY from your .env (if present)

client = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"],
)

collections = client.get_collections()
print(collections)
