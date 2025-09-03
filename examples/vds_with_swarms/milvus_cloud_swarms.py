"""
Agent with Milvus Cloud RAG (Retrieval-Augmented Generation)

This example demonstrates using Milvus Cloud (Zilliz) as a vector database for RAG operations,
allowing agents to store and retrieve documents from your cloud-hosted Milvus account.
"""

import os
from swarms import Agent
from swarms_memory import MilvusDB

# Get Milvus Cloud credentials
milvus_uri = os.getenv("MILVUS_URI")
milvus_token = os.getenv("MILVUS_TOKEN")

if not milvus_uri or not milvus_token:
    print("❌ Missing Milvus Cloud credentials!")
    print("Please set MILVUS_URI and MILVUS_TOKEN in your .env file")
    exit(1)

# Initialize Milvus Cloud wrapper for RAG operations
rag_db = MilvusDB(
    embedding_model="text-embedding-3-small",  # OpenAI embedding model
    collection_name="swarms_cloud_knowledge",  # Cloud collection name
    uri=milvus_uri,                           # Your Zilliz Cloud URI
    token=milvus_token,                       # Your Zilliz Cloud token
    metric="COSINE",                          # Distance metric for similarity search
)

# Add documents to the knowledge base
documents = [
    "Milvus Cloud is a fully managed vector database service provided by Zilliz.",
    "RAG combines retrieval and generation for more accurate AI responses.",
    "Vector embeddings enable semantic search across documents.",
    "The swarms framework supports multiple memory backends including Milvus Cloud.",
    "Swarms is the first and most reliable multi-agent production-grade framework.",
    "Kye Gomez is Founder and CEO of Swarms."
]

# Add documents individually
for doc in documents:
    rag_db.add(doc)

# Create agent with RAG capabilities
agent = Agent(
    agent_name="Cloud-RAG-Agent",
    agent_description="Swarms Agent with Milvus Cloud-powered RAG for scalable knowledge retrieval",
    model_name="gpt-4o",
    max_loops=1,
    dynamic_temperature_enabled=True,
    long_term_memory=rag_db
)

# Query with RAG
response = agent.run("What is Milvus Cloud and how does it relate to RAG? Who is the founder of Swarms?")
print(response)