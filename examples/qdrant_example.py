"""
QdrantDB with Swarms framework - Vector similarity search example
"""

import os
from qdrant_client import QdrantClient, models
from swarms_memory.vector_dbs import QdrantDB

# Initialize Qdrant client
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),  # Your cluster URL
    api_key=os.getenv("QDRANT_API_KEY"),  # Your API key
)

# Example 1: OpenAI embeddings
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
    "Swarms is an AI agent framework for multi-agent coordination.",
]

for doc in knowledge_docs:
    rag_db.add(doc)

# Test query
results = rag_db.query("What is Qdrant?")
print(f"Results: {results}")

# Example 2: Using Cohere embeddings
cohere_db = QdrantDB(
    client=client,
    embedding_model="cohere/embed-english-v3.0",
    collection_name="cohere_collection",
    embedding_kwargs={"input_type": "search_document"}
)

# Example 3: Custom embedding function
def simple_embedding(text: str):
    import hashlib
    hash_obj = hashlib.md5(text.encode())
    hash_hex = hash_obj.hexdigest()
    embedding = []
    for i in range(0, len(hash_hex), 2):
        byte_val = int(hash_hex[i:i+2], 16)
        embedding.append((byte_val - 127.5) / 127.5)
    while len(embedding) < 384:
        embedding.extend(embedding[:min(384-len(embedding), len(embedding))])
    return embedding[:384]

custom_db = QdrantDB(
    client=client,
    embedding_model=simple_embedding,
    embedding_dim=384,
    collection_name="custom_collection"
)