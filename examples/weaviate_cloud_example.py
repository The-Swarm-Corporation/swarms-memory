"""
Weaviate Cloud Vector Database Example

This example demonstrates using WeaviateDB wrapper with Weaviate Cloud Service
for production-ready vector database operations.
"""

import os
from swarms_memory import WeaviateDB


def basic_cloud_example():
    """Basic example using Weaviate Cloud with OpenAI embeddings."""
    print("\n=== Basic Weaviate Cloud Example ===")
    
    # Get credentials from environment
    weaviate_url = os.getenv("WEAVIATE_URL")
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
    
    if not weaviate_url or not weaviate_api_key:
        print("Missing Weaviate Cloud credentials.")
        print("Set WEAVIATE_URL and WEAVIATE_API_KEY environment variables.")
        return None
    
    # Initialize WeaviateDB with cloud credentials
    weaviate_db = WeaviateDB(
        embedding_model="text-embedding-3-small",
        collection_name="cloud_documents",
        cluster_url=f"https://{weaviate_url}",
        auth_client_secret=weaviate_api_key,
        distance_metric="cosine",
    )
    
    # Sample documents to add
    documents = [
        "Weaviate Cloud Service provides managed vector database hosting.",
        "Cloud-hosted vector databases offer scalability and reliability.",
        "Weaviate supports multiple embedding models and distance metrics.",
        "Production vector databases require proper indexing and optimization.",
        "RAG applications benefit from fast and accurate vector retrieval.",
    ]
    
    # Add documents with metadata
    print("Adding documents to cloud...")
    doc_ids = []
    for i, doc in enumerate(documents):
        doc_id = weaviate_db.add(
            doc, 
            metadata={
                "source": "cloud_example",
                "category": "production",
                "index": i,
                "environment": "cloud"
            }
        )
        doc_ids.append(doc_id)
        print(f"Added document {i+1}: {doc_id}")
    
    # Query for similar documents
    print("\nQuerying cloud database...")
    query = "What are the benefits of cloud vector databases?"
    results = weaviate_db.query(query, top_k=3)
    
    print(f"Query: '{query}'")
    print(f"Found {len(results)} results:")
    for i, result in enumerate(results):
        print(f"  {i+1}. Score: {result['score']:.4f}")
        print(f"     Text: {result['metadata']['text']}")
        print(f"     ID: {result['id']}")
    
    # Get document count
    count = weaviate_db.count()
    print(f"\nTotal documents in cloud collection: {count}")
    
    # Health check
    health = weaviate_db.health_check()
    print(f"Cloud database health: {health['status']}")
    
    return weaviate_db


def production_filtering_example():
    """Example demonstrating production-level metadata filtering."""
    print("\n=== Production Metadata Filtering Example ===")
    
    weaviate_url = os.getenv("WEAVIATE_URL")
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
    
    if not weaviate_url or not weaviate_api_key:
        print("Missing credentials for production example.")
        return None
    
    weaviate_db = WeaviateDB(
        embedding_model="text-embedding-3-small",
        collection_name="production_docs",
        cluster_url=f"https://{weaviate_url}",
        auth_client_secret=weaviate_api_key
    )
    
    # Add documents with production metadata
    production_docs = [
        ("Customer support handles user inquiries and technical issues.", 
         {"department": "support", "priority": "high", "type": "documentation"}),
        ("Marketing campaigns drive user acquisition and engagement.",
         {"department": "marketing", "priority": "medium", "type": "strategy"}),
        ("Engineering team develops core platform features.",
         {"department": "engineering", "priority": "high", "type": "development"}),
        ("Sales team manages client relationships and revenue growth.",
         {"department": "sales", "priority": "high", "type": "process"}),
    ]
    
    print("Adding production documents...")
    for doc, metadata in production_docs:
        weaviate_db.add(doc, metadata=metadata)
    
    # Query with department filtering
    query = "How does the team handle user issues?"
    
    print(f"\nQuery: '{query}'")
    
    # Search only support documents
    support_results = weaviate_db.query(
        query, 
        top_k=2, 
        filter_dict={"department": "support"}
    )
    
    print("\nSupport department results:")
    for result in support_results:
        print(f"  - {result['metadata']['text']}")
        print(f"    Department: {result['metadata']['department']}")
        print(f"    Priority: {result['metadata']['priority']}")
    
    return weaviate_db


def rag_text_example():
    """Example demonstrating RAG text output for production use."""
    print("\n=== Production RAG Example ===")
    
    weaviate_url = os.getenv("WEAVIATE_URL")
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
    
    if not weaviate_url or not weaviate_api_key:
        print("Missing credentials for RAG example.")
        return None
    
    weaviate_db = WeaviateDB(
        embedding_model="text-embedding-3-small",
        collection_name="knowledge_base", 
        cluster_url=f"https://{weaviate_url}",
        auth_client_secret=weaviate_api_key
    )
    
    # Add knowledge base documents
    knowledge_docs = [
        "Weaviate Cloud automatically scales based on usage patterns and data volume.",
        "Production vector databases require monitoring of query latency and throughput.",
        "Cloud services provide automatic backups and disaster recovery capabilities.",
        "Vector indexing in production environments uses HNSW for optimal performance.",
        "Authentication and access control ensure data security in multi-tenant systems."
    ]
    
    print("Building production knowledge base...")
    for doc in knowledge_docs:
        weaviate_db.add(doc, metadata={"type": "knowledge", "env": "production"})
    
    # Query and get formatted text output for RAG
    query = "How does Weaviate Cloud handle scaling and performance?"
    formatted_text = weaviate_db.query_as_text(query, top_k=3)
    
    print(f"Query: '{query}'")
    print("RAG context for language model:")
    print("-" * 50)
    print(formatted_text)
    print("-" * 50)
    
    return weaviate_db


def main():
    """Run all Weaviate Cloud examples."""
    print("Weaviate Cloud Vector Database Examples")
    print("=" * 60)
    
    try:
        # Check for required environment variables
        if not os.getenv("WEAVIATE_URL") or not os.getenv("WEAVIATE_API_KEY"):
            print("Missing required environment variables:")
            print("  WEAVIATE_URL - Your Weaviate Cloud cluster URL")
            print("  WEAVIATE_API_KEY - Your Weaviate Cloud API key")
            print("\nGet started at: https://console.weaviate.cloud/")
            return
        
        # Run cloud examples
        basic_cloud_example()
        
        production_filtering_example()
        
        rag_text_example()
        
        print("\nAll Weaviate Cloud examples completed successfully!")
        print("Check your Weaviate Cloud dashboard to see the collections.")
        
    except Exception as e:
        print(f"\nCloud example failed: {e}")
        print("Make sure you have:")
        print("1. Valid Weaviate Cloud cluster and API key")
        print("2. Installed weaviate-client: pip install weaviate-client>=4.0.0")
        print("3. Set up your OpenAI API key for embeddings")
        print("4. Active internet connection")


if __name__ == "__main__":
    main()