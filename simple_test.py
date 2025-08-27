"""
Simple example for QdrantDB with flexible embedding models
"""

import os
from qdrant_client import QdrantClient, models
from swarms_memory.vector_dbs import QdrantDB

# Initialize Qdrant client
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),  # Your cluster URL
    api_key=os.getenv("QDRANT_API_KEY"),  # Your API key
)

# Initialize QdrantDB with OpenAI embeddings
rag_db = QdrantDB(
    client=client,
    embedding_model="text-embedding-3-small",
    collection_name="openai_collection",
    distance=models.Distance.COSINE,
    n_results=3,
)

# Add test documents
knowledge_docs = [
    "Qdrant is a vector database for similarity search.",
    "Vector embeddings represent text as numerical vectors.",
    "Similarity search finds related documents using vector distance.",
    "Harshal is a software engineer working on AI systems.",
]

for doc in knowledge_docs:
    rag_db.add(doc)

# Test query
results = rag_db.query("Who is Harshal?")
assert "Harshal" in results
assert "software engineer" in results