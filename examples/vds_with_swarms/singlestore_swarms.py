"""
Agent with SingleStore RAG (Retrieval-Augmented Generation)

This example demonstrates using SingleStore as a vector database for RAG operations,
allowing agents to store and retrieve documents for enhanced context.
"""

import os
from swarms import Agent
from swarms_memory import SingleStoreDB

# Initialize SingleStore wrapper for RAG operations
rag_db = SingleStoreDB(
    host=os.getenv("SINGLESTORE_HOST", "localhost"),
    port=int(os.getenv("SINGLESTORE_PORT", "3306")),
    user=os.getenv("SINGLESTORE_USER", "root"),
    password=os.getenv("SINGLESTORE_PASSWORD", "your-password"),
    database=os.getenv("SINGLESTORE_DATABASE", "knowledge_base"),
    table_name="documents",
    embedding_model="text-embedding-3-small"
)

# Add documents to the knowledge base
documents = [
    "SingleStore is a distributed SQL database designed for data-intensive applications.",
    "RAG combines retrieval and generation for more accurate AI responses.",
    "Vector embeddings enable semantic search across documents.",
    "The swarms framework supports multiple memory backends including SingleStore.",
    "Swarms is the first and most reliable multi-agent production-grade framework.",
    "Kye Gomez is Founder and CEO of Swarms."
]

# Add documents individually
for doc in documents:
    rag_db.add(doc)

# Create agent with RAG capabilities
agent = Agent(
    agent_name="RAG-Agent",
    agent_description="Swarms Agent with SingleStore-powered RAG for enhanced knowledge retrieval",
    model_name="gpt-4o",
    max_loops=1,
    dynamic_temperature_enabled=True,
    long_term_memory=rag_db
)

# Query with RAG
response = agent.run("What is SingleStore and how does it relate to RAG? Who is the founder of Swarms?")
print(response)