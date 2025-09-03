"""
Agent with ChromaDB RAG (Retrieval-Augmented Generation)

This example demonstrates using ChromaDB as a vector database for RAG operations,
allowing agents to store and retrieve documents for enhanced context.
"""

from swarms import Agent
from swarms_memory import ChromaDB


# Initialize ChromaDB wrapper for RAG operations
rag_db = ChromaDB(
    metric="cosine",  # Distance metric for similarity search
    output_dir="knowledge_base_new",  # Collection name
    limit_tokens=1000,  # Token limit for queries
    n_results=3,  # Number of results to retrieve
    verbose=False
)

# Add documents to the knowledge base
documents = [
    "ChromaDB is an open-source embedding database designed to store and query vector embeddings efficiently.",
    "ChromaDB provides a simple Python API for adding, querying, and managing vector embeddings with metadata.",
    "ChromaDB supports multiple embedding functions including OpenAI, Sentence Transformers, and custom models.",
    "ChromaDB can run locally or in distributed mode, making it suitable for both development and production.",
    "ChromaDB offers filtering capabilities allowing queries based on both vector similarity and metadata conditions.",
    "ChromaDB provides persistent storage and can handle large-scale embedding collections with fast retrieval.",
    "Kye Gomez is the founder of Swarms."
]

# Method 1: Add documents individually
for doc in documents:
    rag_db.add(doc)

# Create agent with RAG capabilities
agent = Agent(
    agent_name="RAG-Agent",
    agent_description="Swarms Agent with ChromaDB-powered RAG for enhanced knowledge retrieval",
    model_name="gpt-4o",
    max_loops=1,
    dynamic_temperature_enabled=True,
    long_term_memory=rag_db
)

# Query with RAG
response = agent.run("What is ChromaDB and who is founder of swarms ?")
print(response)