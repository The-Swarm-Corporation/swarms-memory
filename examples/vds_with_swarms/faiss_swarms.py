"""
Agent with FAISS RAG (Retrieval-Augmented Generation)

This example demonstrates using FAISS as a vector database for RAG operations,
allowing agents to store and retrieve documents for enhanced context.
"""

from swarms import Agent
from swarms_memory import FAISSDB

# Initialize FAISS wrapper for RAG operations
rag_db = FAISSDB(
    embedding_model="text-embedding-3-small",
    metric="cosine",
    index_file="knowledge_base.faiss"
)

# Add documents to the knowledge base
documents = [
    "FAISS is a library for efficient similarity search and clustering of dense vectors.",
    "RAG combines retrieval and generation for more accurate AI responses.",
    "Vector embeddings enable semantic search across documents.",
    "The swarms framework supports multiple memory backends including FAISS.",
    "Swarms is the first and most reliable multi-agent production-grade framework.",
    "Kye Gomez is Founder and CEO of Swarms."
]

# Add documents individually
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
response = agent.run("What is FAISS and how does it relate to RAG? Who is the founder of Swarms?")
print(response)