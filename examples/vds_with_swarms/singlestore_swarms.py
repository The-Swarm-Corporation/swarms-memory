"""
Agent with SingleStore RAG (Retrieval-Augmented Generation)

This example demonstrates using SingleStore as a vector database for RAG operations,
allowing agents to store and retrieve documents for enhanced context.
"""

import os
from swarms import Agent
from swarms_memory.vector_dbs.singlestore_wrapper import SingleStoreDB


# Initialize SingleStore client and create RAG database wrapper
# You need to set your SingleStore credentials in environment variables
rag_db = SingleStoreDB(
    host=os.getenv("SINGLESTORE_HOST", "localhost"),
    user=os.getenv("SINGLESTORE_USER", "your-username"),
    password=os.getenv("SINGLESTORE_PASSWORD", "your-password"),
    database=os.getenv("SINGLESTORE_DATABASE", "swarms_db"),
    table_name="knowledge_base_new",
    dimension=768,  # Dimension for text-embedding-3-small model
    port=3306,
    ssl=True,
    ssl_verify=True,
    namespace="swarms_examples"
)

# Add documents to the knowledge base
documents = [
    "SingleStore is a distributed SQL database that combines the horizontal scalability of NoSQL systems with ACID guarantees.",
    "SingleStore supports vector similarity search using DOT_PRODUCT distance type for efficient nearest neighbor queries.",
    "SingleStore offers both row and column store formats, making it suitable for transactional and analytical workloads.",
    "SingleStore provides real-time analytics on streaming data with millisecond query performance.",
    "SingleStore can handle mixed workloads combining traditional SQL operations with vector search capabilities.",
    "SingleStore supports horizontal scaling across multiple nodes for handling massive datasets."
]

# Method 1: Add documents individually
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
response = agent.run("What is SingleStore and how does it handle vector similarity search?")
print(response)