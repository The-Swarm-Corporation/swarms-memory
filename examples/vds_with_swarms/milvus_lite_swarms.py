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
    "Milvus supports both dense and sparse vectors with multiple indexing algorithms like HNSW, IVF, and DiskANN.",
    "Milvus provides ACID transactions, schema management, and horizontal scalability for production workloads.", 
    "Milvus integrates with popular AI frameworks and supports deployment on cloud platforms and Kubernetes.",
    "Milvus Lite is a lightweight version that can be embedded directly in Python applications for development.",
    "The swarms framework enables multi-agent orchestration and coordination for complex AI workflows.",
    "Swarms agents can utilize long-term memory systems like vector databases for knowledge persistence.",
    "RAG (Retrieval-Augmented Generation) combines information retrieval with language model generation.",
    "Vector embeddings encode semantic meaning of text, images, and other data into high-dimensional spaces.",
    "Similarity search finds the most relevant information based on vector proximity and semantic similarity.",
    "Kye Gomez is the founder and CEO of Swarms."
]

# Add documents individually with metadata
print("Building knowledge base...")
for i, doc in enumerate(documents):
    rag_db.add(
        doc, 
        metadata={
            "source": "documentation",
            "category": "ai_technology", 
            "index": i,
            "timestamp": "2024"
        }
    )
    print(f"Added document {i+1}: {doc[:50]}...")

# Create agent with RAG capabilities
agent = Agent(
    agent_name="RAG-Agent",
    agent_description="Swarms Agent with Milvus-powered RAG for enhanced knowledge retrieval and semantic search",
    model_name="gpt-4o",
    max_loops=1,
    dynamic_temperature_enabled=True,
    long_term_memory=rag_db
)

# Test queries with RAG
test_queries = [
    "What is Milvus and what are its key capabilities?",
    "How does Swarms integrate with vector databases? Who founded it?", 
    "Explain RAG and how it enhances AI applications.",
    "What indexing algorithms does Milvus support?",
    "How does Milvus Lite differ from Milvus Server?"
]

print("\n" + "="*80)
print("🤖 Testing Swarms Agent with Milvus RAG Integration")
print("="*80)

for i, query in enumerate(test_queries, 1):
    print(f"\n📝 Query {i}: {query}")
    print("-" * 60)
    
    try:
        response = agent.run(query)
        print(f"🎯 Response: {response}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("-" * 60)

print("\n🎉 Milvus RAG integration test completed!")

# Optional: Demonstrate direct database operations
print("\n📊 Database Statistics:")
print(f"Total documents: {rag_db.count()}")
print(f"Collection name: {rag_db.collection_name}")
print(f"Embedding model: {rag_db.model_name}")
print(f"Vector dimension: {rag_db.dimension}")

# Optional: Test query as text functionality  
print("\n🔍 Direct Query Test:")
direct_query = "Tell me about vector databases and their applications"
results = rag_db.query_as_text(direct_query, top_k=3)
print(f"Query: {direct_query}")
print("Retrieved context:")
print(results)