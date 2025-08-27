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
    collection_name="test_collection",  # Changed to avoid duplicates
    distance=models.Distance.COSINE,
    n_results=1,  # Return only the most relevant result
)

# Add test documents
knowledge_docs = [
    "Qdrant is a vector database for similarity search.",
    "Vector embeddings represent text as numerical vectors.",
    "Similarity search finds related documents using vector distance.",
    "Harshal is a software engineer working on AI systems.",
    "Python is a programming language used for data science.",
    "Machine learning models can process natural language.",
]

# Clear collection first (if it exists) to avoid duplicates
try:
    client.delete_collection(collection_name="test_collection")
except:
    pass

# Reinitialize after clearing
rag_db = QdrantDB(
    client=client,
    embedding_model="text-embedding-3-small",
    collection_name="test_collection",
    distance=models.Distance.COSINE,
    n_results=1,
)

# Add documents
for doc in knowledge_docs:
    rag_db.add(doc)

# Test multiple queries
queries = [
    "who is Harshal?",
    "what is Qdrant?",
    "explain vector embeddings",
]

print("=" * 50)
for query in queries:
    print(f"\nQuery: {query}")
    response = rag_db.query(query)
    if isinstance(response, list):
        for i, result in enumerate(response, 1):
            print(f"  Result {i}: {result}")
    else:
        print(f"  Answer: {response}")
print("=" * 50)