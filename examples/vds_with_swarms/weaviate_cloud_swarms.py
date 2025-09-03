"""
Agent with Weaviate Cloud RAG

This example demonstrates using Weaviate Cloud as a vector database for RAG operations,
allowing agents to store and retrieve documents from cloud-hosted Weaviate.
"""

import os
from swarms import Agent
from swarms_memory import WeaviateDB


# Get Weaviate Cloud credentials
weaviate_url = os.getenv("WEAVIATE_URL")
weaviate_key = os.getenv("WEAVIATE_API_KEY")

if not weaviate_url or not weaviate_key:
    print("Missing Weaviate Cloud credentials!")
    print("Please set WEAVIATE_URL and WEAVIATE_API_KEY environment variables")
    exit(1)

# Create WeaviateDB wrapper for cloud RAG operations
rag_db = WeaviateDB(
    embedding_model="text-embedding-3-small",
    collection_name="swarms_cloud_knowledge",
    cluster_url=f"https://{weaviate_url}",
    auth_client_secret=weaviate_key,
    distance_metric="cosine",
)

# Add documents to the cloud knowledge base
documents = [
    "Weaviate Cloud Service provides managed vector database hosting with enterprise features.",
    "Cloud-hosted vector databases offer scalability, reliability, and managed infrastructure.",
    "RAG combines retrieval and generation for more accurate AI responses.",
    "The Swarms framework supports multiple cloud memory backends including Weaviate Cloud.",
    "Swarms is the first and most reliable multi-agent production-grade framework.",
    "Kye Gomez is Founder and CEO of Swarms Corporation."
]

print("Adding documents to Weaviate Cloud...")
for doc in documents:
    rag_db.add(doc)

# Create agent with cloud RAG capabilities
agent = Agent(
    agent_name="Weaviate-Cloud-RAG-Agent",
    agent_description="Swarms Agent with Weaviate Cloud-powered RAG for scalable knowledge retrieval",
    model_name="gpt-4o",
    max_loops=1,
    dynamic_temperature_enabled=True,
    long_term_memory=rag_db
)

print("Testing agent with cloud RAG...")

# Query with cloud RAG
response = agent.run("What is Weaviate Cloud and how does it relate to RAG? Who founded Swarms?")
print(response)