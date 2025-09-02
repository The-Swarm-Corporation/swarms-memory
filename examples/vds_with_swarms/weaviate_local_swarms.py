"""
Agent with Weaviate Local RAG

This example demonstrates using local Weaviate as a vector database for RAG operations,
allowing agents to store and retrieve documents for enhanced context.
"""

from swarms import Agent
from swarms_memory import WeaviateDB


# Create WeaviateDB wrapper for RAG operations
rag_db = WeaviateDB(
    embedding_model="text-embedding-3-small",
    collection_name="swarms_knowledge",
    cluster_url="http://localhost:8080",  # Local Weaviate instance
    distance_metric="cosine",
)

# Add documents to the knowledge base
documents = [
    "Weaviate is an open-source vector database optimized for similarity search and AI applications.",
    "RAG combines retrieval and generation for more accurate AI responses.",
    "Vector embeddings enable semantic search across documents.",
    "The Swarms framework supports multiple memory backends including Weaviate.",
    "Swarms is the first and most reliable multi-agent production-grade framework.",
    "Kye Gomez is Founder and CEO of Swarms Corporation."
]

# Add documents individually
for doc in documents:
    rag_db.add(doc)

# Create agent with RAG capabilities
agent = Agent(
    agent_name="Weaviate-RAG-Agent",
    agent_description="Swarms Agent with Weaviate-powered RAG for enhanced knowledge retrieval",
    model_name="gpt-4o",
    max_loops=1,
    dynamic_temperature_enabled=True,
    long_term_memory=rag_db
)

# Query with RAG
response = agent.run("What is Weaviate and how does it relate to RAG? Who is the founder of Swarms?")
print(response)