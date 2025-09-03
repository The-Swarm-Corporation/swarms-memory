"""
Milvus Vector Database Wrapper Example

This example demonstrates the comprehensive usage of the MilvusDB wrapper
for document storage, retrieval, and management using Milvus vector database.
Shows both Milvus Lite (local) and Milvus Server configurations.
"""

from swarms_memory import MilvusDB
import time


def basic_milvus_example():
    """Basic example using Milvus Lite with OpenAI embeddings."""
    print("\n=== Basic Milvus Example ===")
    
    # Initialize MilvusDB with Milvus Lite (local file storage)
    milvus_db = MilvusDB(
        embedding_model="text-embedding-3-small",  # OpenAI embedding model
        collection_name="basic_documents",
        db_file="basic_milvus.db",  # Local database file
        metric="COSINE",  # Similarity metric
    )
    
    # Sample documents to add
    documents = [
        "Milvus is an open-source vector database designed for AI applications.",
        "Vector embeddings represent text, images, and other data as numerical vectors.",
        "Similarity search allows finding related content based on semantic meaning.",
        "RAG (Retrieval-Augmented Generation) combines retrieval with language models.",
        "LiteLLM provides a unified interface to multiple embedding providers.",
    ]
    
    # Add documents with metadata
    print("Adding documents...")
    doc_ids = []
    for i, doc in enumerate(documents):
        doc_id = milvus_db.add(
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
    results = milvus_db.query(query, top_k=3)
    
    print(f"Query: '{query}'")
    print(f"Found {len(results)} results:")
    for i, result in enumerate(results):
        print(f"  {i+1}. Score: {result['score']:.4f}")
        print(f"     Text: {result['metadata']['text']}")
        print(f"     ID: {result['id']}")
    
    # Get document count
    count = milvus_db.count()
    print(f"\nTotal documents in collection: {count}")
    
    # Health check
    health = milvus_db.health_check()
    print(f"Database health: {health['status']}")
    
    return milvus_db


def metadata_filtering_example():
    """Example demonstrating metadata filtering capabilities."""
    print("\n=== Metadata Filtering Example ===")
    
    milvus_db = MilvusDB(
        embedding_model="text-embedding-3-small",
        collection_name="filtered_documents",
        db_file="filtered_milvus.db"
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
        milvus_db.add(doc, metadata={"category": "technology", "type": "article"})
    
    print("Adding science documents...")
    for doc in science_docs:
        milvus_db.add(doc, metadata={"category": "science", "type": "article"})
    
    # Query with metadata filtering
    query = "How do systems learn and adapt?"
    
    print(f"\nQuery: '{query}'")
    
    # Search only in technology category
    tech_results = milvus_db.query(
        query, 
        top_k=2, 
        filter_dict={"category": "technology"}
    )
    
    print("Technology results:")
    for result in tech_results:
        print(f"  - {result['metadata']['text']}")
        print(f"    Category: {result['metadata']['category']}")
    
    # Search only in science category
    science_results = milvus_db.query(
        query, 
        top_k=2, 
        filter_dict={"category": "science"}
    )
    
    print("Science results:")
    for result in science_results:
        print(f"  - {result['metadata']['text']}")
        print(f"    Category: {result['metadata']['category']}")
    
    return milvus_db


def text_output_example():
    """Example demonstrating text output for RAG operations."""
    print("\n=== Text Output Example ===")
    
    milvus_db = MilvusDB(
        embedding_model="text-embedding-3-small",
        collection_name="rag_documents", 
        db_file="rag_milvus.db"
    )
    
    # Add knowledge base documents
    knowledge_docs = [
        "Milvus supports both dense and sparse vector representations for flexible search.",
        "Vector databases optimize storage and retrieval of high-dimensional vectors.",
        "Similarity metrics like cosine, euclidean, and inner product measure vector proximity.",
        "Indexing algorithms like IVF, HNSW, and PQ improve search performance at scale.",
        "Milvus provides ACID transactions for data consistency in multi-user environments."
    ]
    
    print("Building knowledge base...")
    for doc in knowledge_docs:
        milvus_db.add(doc, metadata={"type": "knowledge", "domain": "vector_db"})
    
    # Query and get formatted text output
    query = "How does Milvus handle vector storage and search?"
    formatted_text = milvus_db.query_as_text(query, top_k=3)
    
    print(f"Query: '{query}'")
    print("Formatted text output for RAG:")
    print("-" * 50)
    print(formatted_text)
    print("-" * 50)
    
    return milvus_db


def custom_embedding_example():
    """Example using custom embedding function."""
    print("\n=== Custom Embedding Example ===")
    
    # Simple custom embedding function (for demonstration)
    def simple_embedding(text: str):
        """Simple word-count based embedding (not recommended for production)."""
        import hashlib
        import numpy as np
        
        # Create a simple hash-based embedding
        hash_object = hashlib.md5(text.encode())
        hex_dig = hash_object.hexdigest()
        
        # Convert to numbers and normalize
        embedding = []
        for i in range(0, len(hex_dig), 2):
            val = int(hex_dig[i:i+2], 16) / 255.0
            embedding.append(val)
        
        # Pad or truncate to desired dimension
        target_dim = 16
        if len(embedding) > target_dim:
            embedding = embedding[:target_dim]
        else:
            embedding.extend([0.0] * (target_dim - len(embedding)))
        
        return embedding
    
    milvus_db = MilvusDB(
        embedding_model=simple_embedding,
        dimension=16,  # Must specify dimension for custom functions
        collection_name="custom_embeddings",
        db_file="custom_milvus.db"
    )
    
    # Add some documents
    docs = [
        "Custom embeddings allow domain-specific representations.",
        "Hash-based embeddings provide deterministic results.",
        "Simple embeddings work for basic similarity tasks."
    ]
    
    print("Adding documents with custom embeddings...")
    for doc in docs:
        doc_id = milvus_db.add(doc)
        print(f"Added: {doc}")
    
    # Test similarity search
    query = "How do custom embeddings work?"
    results = milvus_db.query(query, top_k=2)
    
    print(f"\nQuery: '{query}'")
    print("Results:")
    for result in results:
        print(f"  - {result['metadata']['text']}")
        print(f"    Score: {result['score']:.4f}")
    
    return milvus_db


def server_connection_example():
    """Example of connecting to Milvus Server (requires running Milvus instance)."""
    print("\n=== Milvus Server Connection Example ===")
    
    try:
        # Connect to Milvus Server (uncomment if you have a server running)
        # milvus_db = MilvusDB(
        #     embedding_model="text-embedding-3-small",
        #     collection_name="server_documents",
        #     uri="http://localhost:19530",  # Milvus server address
        #     token="root:Milvus"  # Authentication token
        # )
        # 
        # print("Connected to Milvus Server successfully!")
        # return milvus_db
        
        print("Milvus Server connection example (commented out)")
        print("To use Milvus Server:")
        print("1. Start Milvus server: docker run -p 19530:19530 milvusdb/milvus:latest")
        print("2. Uncomment the connection code above")
        print("3. Run this example again")
        
        return None
        
    except Exception as e:
        print(f"Failed to connect to Milvus Server: {e}")
        print("Make sure Milvus Server is running on localhost:19530")
        return None


def crud_operations_example():
    """Example demonstrating CRUD (Create, Read, Update, Delete) operations."""
    print("\n=== CRUD Operations Example ===")
    
    milvus_db = MilvusDB(
        embedding_model="text-embedding-3-small",
        collection_name="crud_documents",
        db_file="crud_milvus.db"
    )
    
    # Create (Add documents)
    print("Creating documents...")
    doc_id1 = milvus_db.add("Document 1: Introduction to vector databases.")
    doc_id2 = milvus_db.add("Document 2: Advanced indexing techniques.")
    doc_id3 = milvus_db.add("Document 3: Query optimization strategies.")
    
    print(f"Created documents: {doc_id1}, {doc_id2}, {doc_id3}")
    
    # Read (Get specific document)
    print("\nReading document...")
    doc = milvus_db.get(doc_id1)
    if doc:
        print(f"Retrieved: {doc['metadata']['text']}")
    else:
        print("Document not found")
    
    # Read (Count all documents)
    count = milvus_db.count()
    print(f"Total documents: {count}")
    
    # Delete (Remove specific document)
    print(f"\nDeleting document {doc_id2}...")
    success = milvus_db.delete(doc_id2)
    if success:
        print("Document deleted successfully")
        
        # Verify deletion
        deleted_doc = milvus_db.get(doc_id2)
        print(f"Verification - Document exists: {deleted_doc is not None}")
        
        new_count = milvus_db.count()
        print(f"New total documents: {new_count}")
    
    # Clear all documents
    print("\nClearing all documents...")
    cleared = milvus_db.clear()
    if cleared:
        final_count = milvus_db.count()
        print(f"All documents cleared. Final count: {final_count}")
    
    return milvus_db


def main():
    """Run all Milvus examples."""
    print("🔍 Milvus Vector Database Wrapper Examples")
    print("=" * 60)
    
    try:
        # Run all examples
        basic_milvus_example()
        time.sleep(1)
        
        metadata_filtering_example()
        time.sleep(1)
        
        text_output_example()
        time.sleep(1)
        
        custom_embedding_example()
        time.sleep(1)
        
        server_connection_example()
        time.sleep(1)
        
        crud_operations_example()
        
        print("\n🎉 All Milvus examples completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        print("Make sure you have:")
        print("1. Installed pymilvus: pip install pymilvus")
        print("2. Set up your OpenAI API key for embeddings")
        print("3. Sufficient disk space for Milvus Lite databases")


if __name__ == "__main__":
    main()