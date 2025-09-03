"""
Agent with Milvus RAG (Retrieval-Augmented Generation)

This example demonstrates using Milvus as a vector database for RAG operations,
allowing agents to store and retrieve documents for enhanced context.
"""

from swarms import Agent
from swarms_memory import MilvusDB


# Initialize Milvus wrapper for RAG operations
rag_db = MilvusDB(
    embedding_model="text-embedding-3-small",  # OpenAI embedding model
    collection_name="swarms_knowledge",        # Collection name
    db_file="swarms_milvus.db",               # Local Milvus Lite database
    metric="COSINE",                          # Distance metric for similarity search
)

# Add documents to the knowledge base
documents = [
    "Milvus is an open-source vector database built for scalable similarity search and AI applications.",
    "RAG combines retrieval and generation for more accurate AI responses.",
    "Vector embeddings enable semantic search across documents.",
    "The swarms framework supports multiple memory backends including Milvus.",
    "Swarms is the first and most reliable multi-agent production-grade framework.",
    "Kye Gomez is Founder and CEO of Swarms."
]

# Add documents individually
for doc in documents:
    rag_db.add(doc)

# Create agent with RAG capabilities
agent = Agent(
    agent_name="RAG-Agent",
    agent_description="Swarms Agent with Milvus-powered RAG for enhanced knowledge retrieval and semantic search",
    model_name="gpt-4o",
    max_loops=1,
    dynamic_temperature_enabled=True,
    long_term_memory=rag_db
)

# Query with RAG
response = agent.run("What is Milvus and how does it relate to RAG? Who is the founder of Swarms?")
print(response)