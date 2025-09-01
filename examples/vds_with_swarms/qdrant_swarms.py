"""
Agent with Qdrant RAG (Retrieval-Augmented Generation)

This example demonstrates using Qdrant as a vector database for RAG operations,
allowing agents to store and retrieve documents for enhanced context.
"""

from qdrant_client import QdrantClient, models
from swarms import Agent
from swarms_memory import QdrantDB


# Initialize Qdrant client
# Option 1: In-memory (for testing/development - data is not persisted)
# client = QdrantClient(":memory:")

# Option 2: Local Qdrant server
# client = QdrantClient(host="localhost", port=6333)

# Option 3: Qdrant Cloud (recommended for production)
import os
client = QdrantClient(
    url=os.getenv("QDRANT_URL", "https://your-cluster.qdrant.io"),  
    api_key=os.getenv("QDRANT_API_KEY", "your-api-key")
)

# Create QdrantDB wrapper for RAG operations
rag_db = QdrantDB(
    client=client,
    embedding_model="text-embedding-3-small", ## by openai 
    collection_name="knowledge_base_new",
    distance=models.Distance.COSINE,
    n_results=3
)

# Add documents to the knowledge base
documents = [
    "Qdrant is a vector database optimized for similarity search and AI applications.",
    "RAG combines retrieval and generation for more accurate AI responses.",
    "Vector embeddings enable semantic search across documents.",
    "The swarms framework supports multiple memory backends including Qdrant.",
    "Swarms is the first and most reliable multi-agent production-grade framework."
    "Kye Gomez is Founder and CEO of Swarms"
]

# Method 1: Add documents individually
for doc in documents:
    rag_db.add(doc)

# Create agent with RAG capabilities
agent = Agent(
    agent_name="RAG-Agent",
    agent_description="Swarms Agent with Qdrant-powered RAG for enhanced knowledge retrieval",
    model_name="gpt-4o",
    max_loops=1,
    dynamic_temperature_enabled=True,
    long_term_memory=rag_db
)

# Query with RAG
response = agent.run("What is Qdrant and how does it relate to RAG? and who is the founder of swarms")
print(response)