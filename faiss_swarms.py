"""
Agent with FAISS RAG (Retrieval-Augmented Generation)

This example demonstrates using FAISS as a vector database for RAG operations,
allowing agents to store and retrieve documents for enhanced context.
"""

from swarms import Agent
from swarms_memory import FAISSDB


# Initialize FAISS database wrapper for RAG operations
rag_db = FAISSDB(
    dimension=768,  # Dimension for text-embedding-3-small model
    index_type="Flat",  # FAISS index type
    metric="cosine",  # Distance metric
    # Note: FAISS is an in-memory database, no external client needed
)

# Add documents to the knowledge base
documents = [
    "FAISS is a library for efficient similarity search and clustering of dense vectors developed by Facebook AI Research.",
    "FAISS provides fast approximate nearest neighbor search algorithms optimized for large-scale vector datasets.",
    "FAISS supports multiple index types including Flat, IVF, and HNSW for different performance trade-offs.",
    "FAISS can run on both CPU and GPU, with GPU implementations providing significant speedup for large datasets.",
    "FAISS is designed for in-memory operation and excels at fast vector similarity search with minimal latency.",
    "FAISS supports various distance metrics including L2, inner product, and cosine similarity."
]

# Method 1: Add documents individually
for doc in documents:
    rag_db.add(doc)

# Create agent with RAG capabilities
agent = Agent(
    agent_name="RAG-Agent",
    agent_description="Swarms Agent with FAISS-powered RAG for enhanced knowledge retrieval",
    model_name="gpt-4o",
    max_loops=1,
    dynamic_temperature_enabled=True,
    long_term_memory=rag_db
)

# Query with RAG
response = agent.run("What is FAISS and what are its key advantages for vector similarity search?")
print(response)