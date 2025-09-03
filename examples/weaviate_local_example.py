"""
Weaviate Vector Database Wrapper Example

This example demonstrates the usage of the WeaviateDB wrapper
for document storage, retrieval, and management using local Weaviate.
"""

import os
from swarms_memory import WeaviateDB


def basic_weaviate_example():
    """Basic example using local Weaviate with OpenAI embeddings."""
    print("\n=== Basic Weaviate Example ===")
    
    # Initialize WeaviateDB with local instance
    weaviate_db = WeaviateDB(
        embedding_model="text-embedding-3-small",  # OpenAI embedding model
        collection_name="basic_documents",
        cluster_url="http://localhost:8080",  # Local Weaviate instance
        distance_metric="cosine",
    )
    
    # Sample documents to add
    documents = [
        "Weaviate is an open-source vector database designed for AI applications.",
        "Vector embeddings represent text, images, and other data as numerical vectors.",
        "Similarity search allows finding related content based on semantic meaning.",
        "RAG combines retrieval with language models for enhanced responses.",
        "LiteLLM provides a unified interface to multiple embedding providers.",
    ]
    
    # Add documents with metadata
    print("Adding documents...")
    doc_ids = []
    for i, doc in enumerate(documents):
        doc_id = weaviate_db.add(
            doc, 
            metadata={
                "source": "example",
                "category": "technology",
                "index": i
            }
        )
        doc_ids.append(doc_id)
        print(f"Added document {i+1}: {doc_id}")
    
    # Query for similar documents
    print("\nQuerying for similar documents...")
    query = "What is vector search?"
    results = weaviate_db.query(query, top_k=3)
    
    print(f"Query: '{query}'")
    print(f"Found {len(results)} results:")
    for i, result in enumerate(results):
        print(f"  {i+1}. Score: {result['score']:.4f}")
        print(f"     Text: {result['metadata']['text']}")
        print(f"     ID: {result['id']}")
    
    # Get document count
    count = weaviate_db.count()
    print(f"\nTotal documents in collection: {count}")
    
    # Health check
    health = weaviate_db.health_check()
    print(f"Database health: {health['status']}")
    
    return weaviate_db


def metadata_filtering_example():
    """Example demonstrating metadata filtering capabilities."""
    print("\n=== Metadata Filtering Example ===")
    
    weaviate_db = WeaviateDB(
        embedding_model="text-embedding-3-small",
        collection_name="filtered_documents",
        cluster_url="http://localhost:8080"
    )
    
    # Add documents with different categories
    tech_docs = [
        "Artificial Intelligence transforms how we process information.",
        "Machine Learning algorithms learn patterns from data.",
        "Deep Learning uses neural networks for complex tasks.",
    ]
    
    science_docs = [
        "Physics explains the fundamental laws of the universe.",
        "Chemistry studies the composition and behavior of matter.",
        "Biology investigates living organisms and life processes.",
    ]
    
    print("Adding technology documents...")
    for doc in tech_docs:
        weaviate_db.add(doc, metadata={"category": "technology", "type": "article"})
    
    print("Adding science documents...")
    for doc in science_docs:
        weaviate_db.add(doc, metadata={"category": "science", "type": "article"})
    
    # Query with metadata filtering
    query = "How do systems learn and adapt?"
    
    print(f"\nQuery: '{query}'")
    
    # Search only in technology category
    tech_results = weaviate_db.query(
        query, 
        top_k=2, 
        filter_dict={"category": "technology"}
    )
    
    print("\nTechnology results:")
    for result in tech_results:
        print(f"  - {result['metadata']['text']}")
        print(f"    Category: {result['metadata']['category']}")
    
    # Search only in science category
    science_results = weaviate_db.query(
        query, 
        top_k=2, 
        filter_dict={"category": "science"}
    )
    
    print("\nScience results:")
    for result in science_results:
        print(f"  - {result['metadata']['text']}")
        print(f"    Category: {result['metadata']['category']}")
    
    return weaviate_db


def text_output_example():
    """Example demonstrating text output for RAG operations."""
    print("\n=== Text Output Example ===")
    
    weaviate_db = WeaviateDB(
        embedding_model="text-embedding-3-small",
        collection_name="rag_documents", 
        cluster_url="http://localhost:8080"
    )
    
    # Add knowledge base documents
    knowledge_docs = [
        "Weaviate supports both dense and sparse vector representations for flexible search.",
        "Vector databases optimize storage and retrieval of high-dimensional vectors.",
        "Similarity metrics like cosine, euclidean, and inner product measure vector proximity.",
        "Indexing algorithms like HNSW and PQ improve search performance at scale.",
        "Weaviate provides ACID transactions for data consistency in multi-user environments."
    ]
    
    print("Building knowledge base...")
    for doc in knowledge_docs:
        weaviate_db.add(doc, metadata={"type": "knowledge", "domain": "vector_db"})
    
    # Query and get formatted text output
    query = "How does Weaviate handle vector storage and search?"
    formatted_text = weaviate_db.query_as_text(query, top_k=3)
    
    print(f"Query: '{query}'")
    print("Formatted text output for RAG:")
    print("-" * 50)
    print(formatted_text)
    print("-" * 50)
    
    return weaviate_db


def crud_operations_example():
    """Example demonstrating CRUD operations."""
    print("\n=== CRUD Operations Example ===")
    
    weaviate_db = WeaviateDB(
        embedding_model="text-embedding-3-small",
        collection_name="crud_documents",
        cluster_url="http://localhost:8080"
    )
    
    # Create (Add documents)
    print("Creating documents...")
    doc_id1 = weaviate_db.add("Document 1: Introduction to vector databases.")
    doc_id2 = weaviate_db.add("Document 2: Advanced indexing techniques.")
    doc_id3 = weaviate_db.add("Document 3: Query optimization strategies.")
    
    print(f"Created documents: {doc_id1}, {doc_id2}, {doc_id3}")
    
    # Read (Get specific document)
    print("\nReading document...")
    doc = weaviate_db.get(doc_id1)
    if doc:
        print(f"Retrieved: {doc['metadata']['text']}")
    else:
        print("Document not found")
    
    # Read (Count all documents)
    count = weaviate_db.count()
    print(f"Total documents: {count}")
    
    # Delete (Remove specific document)
    print(f"\nDeleting document {doc_id2}...")
    success = weaviate_db.delete(doc_id2)
    if success:
        print("Document deleted successfully")
        
        # Verify deletion
        deleted_doc = weaviate_db.get(doc_id2)
        print(f"Verification - Document exists: {deleted_doc is not None}")
        
        new_count = weaviate_db.count()
        print(f"New total documents: {new_count}")
    
    # Clear all documents
    print("\nClearing all documents...")
    cleared = weaviate_db.clear()
    if cleared:
        final_count = weaviate_db.count()
        print(f"All documents cleared. Final count: {final_count}")
    
    return weaviate_db


def main():
    """Run all Weaviate examples."""
    print("Weaviate Vector Database Wrapper Examples")
    print("=" * 60)
    
    try:
        # Run all examples
        basic_weaviate_example()
        
        metadata_filtering_example()
        
        text_output_example()
        
        crud_operations_example()
        
        print("\nAll Weaviate examples completed successfully!")
        
    except Exception as e:
        print(f"\nExample failed: {e}")
        print("Make sure you have:")
        print("1. Docker running")
        print("2. Start: docker run -p 8080:8080 -p 50051:50051 weaviate/weaviate")
        print("3. Installed weaviate-client: pip install weaviate-client>=4.0.0")
        print("4. Set up your OpenAI API key for embeddings")


if __name__ == "__main__":
    main()