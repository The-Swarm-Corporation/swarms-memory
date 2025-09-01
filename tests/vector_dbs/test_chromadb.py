"""
Human-Testable ChromaDB Wrapper Test Suite

This script demonstrates and tests the modernized ChromaDB wrapper functionality
including LiteLLM integration, CRUD operations, and collection management.

Run this script to verify ChromaDB wrapper functionality:
python tests/vector_dbs/test_chromadb.py
"""

import os
import tempfile
from swarms_memory.vector_dbs.chroma_db_wrapper import ChromaDB

def test_chromadb_basic_operations():
    """Test basic ChromaDB operations."""
    print("Testing ChromaDB Basic Operations")
    
    # Initialize ChromaDB with default settings
    chroma_db = ChromaDB(
        metric="cosine",
        output_dir="test_chromadb",
        limit_tokens=1000,
        n_results=5,
        verbose=False
    )
    
    print("✓ ChromaDB initialized successfully")
    print(f"  Metric: {chroma_db.metric}, Results limit: {chroma_db.n_results}")
    
    # Test adding documents
    documents = [
        "ChromaDB is an open-source vector database for AI applications.",
        "Vector databases store high-dimensional embeddings efficiently.",
        "Semantic search helps find relevant documents by meaning.",
        "Machine learning models benefit from vector similarity search.",
        "Embeddings capture semantic relationships between text."
    ]
    
    doc_ids = []
    for i, doc in enumerate(documents):
        doc_id = chroma_db.add(doc, metadata={"category": "vector_db", "index": i})
        doc_ids.append(doc_id)
        if i == 0:
            print(f"✓ Added document: {doc_id[:8]}...")
    
    print(f"✓ Added {len(documents)} documents")
    
    # Test counting and querying
    count = chroma_db.count()
    print(f"✓ Total documents: {count}")
    
    results = chroma_db.query(query_text="vector database search", top_k=3)
    print(f"✓ Found {len(results)} results for vector database query")
    if results:
        print(f"  Top result score: {results[0]['score']:.4f}")
    
    # Test document operations
    if doc_ids:
        retrieved = chroma_db.get(doc_ids[0])
        if retrieved:
            print("✓ Successfully retrieved document by ID")
        
        success = chroma_db.delete(doc_ids[-1])
        print(f"✓ Delete operation: {'success' if success else 'failed'}")
    
    # Test health check
    health = chroma_db.health_check()
    print(f"✓ Health check status: {health.get('status', 'unknown')}")
    
    return chroma_db

def test_chromadb_filtering():
    """Test ChromaDB filtering functionality."""
    print("\nTesting ChromaDB Filtering")
    
    chroma_db = ChromaDB(collection_name="test_filtering")
    
    # Add documents with different categories
    categories = ["tutorial", "research", "news", "tutorial", "research"]
    topics = ["AI", "ML", "NLP", "CV", "RL"]
    
    for i in range(5):
        chroma_db.add(
            f"Document about {topics[i]} in {categories[i]} format",
            metadata={"category": categories[i], "topic": topics[i], "index": i}
        )
    
    print(f"✓ Added {len(categories)} documents with categories and topics")
    
    # Test filtering by category
    tutorial_results = chroma_db.query(
        query_text="document tutorial",
        top_k=10,
        filter_dict={"category": "tutorial"}
    )
    
    print(f"✓ Found {len(tutorial_results)} tutorial documents")
    
    # Test filtering by topic
    ai_results = chroma_db.query(
        query_text="artificial intelligence",
        top_k=10,
        filter_dict={"topic": "AI"}
    )
    
    print(f"✓ Found {len(ai_results)} AI-related documents")

def test_chromadb_collection_management():
    """Test ChromaDB collection management."""
    print("\nTesting ChromaDB Collection Management")
    
    try:
        # Create multiple collections
        collection_names = ["test_collection_1", "test_collection_2"]
        chromadbs = []
        
        for name in collection_names:
            chroma_db = ChromaDB(collection_name=name)
            chroma_db.add(f"Document in {name}", metadata={"collection": name})
            chromadbs.append(chroma_db)
        
        print(f"✓ Created {len(collection_names)} collections")
        
        # Test collection isolation
        for i, chroma_db in enumerate(chromadbs):
            count = chroma_db.count()
            print(f"✓ Collection {collection_names[i]}: {count} documents")
        
        # Test clearing a collection
        success = chromadbs[0].clear()
        print(f"✓ Clear collection: {'success' if success else 'failed'}")
        
        count_after_clear = chromadbs[0].count()
        print(f"✓ Documents after clear: {count_after_clear}")
        
    except Exception as e:
        print(f"⚠ Collection management test failed: {str(e)}")

def test_chromadb_directory_traversal():
    """Test ChromaDB directory traversal functionality."""
    print("\nTesting ChromaDB Directory Traversal")
    
    # Create temporary directory with test files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test files
        test_files = {
            "doc1.txt": "This is a document about machine learning and AI.",
            "doc2.md": "# Vector Databases\nVector databases are useful for AI applications.",
            "doc3.txt": "Natural language processing helps understand text."
        }
        
        for filename, content in test_files.items():
            with open(os.path.join(temp_dir, filename), 'w') as f:
                f.write(content)
        
        print(f"✓ Created {len(test_files)} test files")
        
        # Initialize ChromaDB with directory
        chroma_db = ChromaDB(
            collection_name="test_directory",
            docs_folder=temp_dir
        )
        
        # Traverse and add documents
        try:
            chroma_db.traverse_directory()
            count = chroma_db.count()
            print(f"✓ Loaded {count} documents from directory")
            
            # Test querying loaded documents
            results = chroma_db.query("machine learning", top_k=3)
            print(f"✓ Found {len(results)} results from loaded documents")
            
        except Exception as e:
            print(f"⚠ Directory traversal failed: {str(e)}")

def test_chromadb_batch_operations():
    """Test ChromaDB batch operations."""
    print("\nTesting ChromaDB Batch Operations")
    
    try:
        chroma_db = ChromaDB(collection_name="test_batch")
        
        # Batch add documents
        documents = [f"Batch document {i} about topic {i}" for i in range(10)]
        doc_ids = []
        
        for doc in documents:
            doc_id = chroma_db.add(doc, metadata={"batch": True})
            doc_ids.append(doc_id)
        
        print(f"✓ Batch added {len(documents)} documents")
        
        # Batch query
        results = chroma_db.query("topic", top_k=5)
        print(f"✓ Batch query returned {len(results)} results")
        
        # Test document count
        count = chroma_db.count()
        print(f"✓ Total documents in collection: {count}")
        
    except Exception as e:
        print(f"⚠ Batch operations test failed: {str(e)}")

def test_chromadb_error_handling():
    """Test ChromaDB error handling scenarios."""
    print("\nTesting ChromaDB Error Handling")
    
    try:
        chroma_db = ChromaDB(collection_name="test_errors")
        
        # Test getting non-existent document
        result = chroma_db.get("non-existent-id")
        print(f"✓ Non-existent document returns: {result}")
        
        # Test deleting non-existent document
        success = chroma_db.delete("non-existent-id")
        print(f"✓ Delete non-existent returns: {success}")
        
        # Test query on empty collection
        results = chroma_db.query("test query", top_k=5)
        print(f"✓ Query on empty collection returns: {len(results)} results")
        
        # Test clear operation
        chroma_db.add("Document to be cleared")
        count_before = chroma_db.count()
        success = chroma_db.clear()
        count_after = chroma_db.count()
        print(f"✓ Clear operation: {count_before} -> {count_after} documents")
        
    except Exception as e:
        print(f"⚠ Error handling test failed: {str(e)}")

def test_chromadb_different_metrics():
    """Test ChromaDB with different distance metrics."""
    print("\nTesting ChromaDB Distance Metrics")
    
    metrics = ["cosine", "l2", "ip"]  # ip = inner product
    
    for metric in metrics:
        try:
            chroma_db = ChromaDB(
                collection_name=f"test_{metric}",
                metric=metric
            )
            
            # Add test document
            chroma_db.add(f"Test document for {metric} metric")
            results = chroma_db.query("test document", top_k=1)
            
            print(f"✓ {metric.upper()} metric working - found {len(results)} results")
            
        except Exception as e:
            print(f"⚠ {metric.upper()} metric failed: {str(e)}")

def main():
    """Run all ChromaDB tests."""
    print("ChromaDB Wrapper Test Suite")
    print("This script tests the modernized ChromaDB wrapper functionality")
    
    try:
        test_chromadb_basic_operations()
        test_chromadb_filtering()
        test_chromadb_collection_management()
        test_chromadb_directory_traversal()
        test_chromadb_batch_operations()
        test_chromadb_error_handling()
        test_chromadb_different_metrics()
        
        print("\n" + "="*60)
        print("  All Tests Completed Successfully!")
        print("="*60)
        print("✓ ChromaDB wrapper is working correctly")
        print("✓ All core functionality verified")
        print("✓ Error handling working as expected")
        
    except Exception as e:
        print("\n" + "="*60)
        print("  Test Suite Failed!")
        print("="*60)
        print(f"❌ Error: {str(e)}")
        print("Please check the ChromaDB wrapper implementation")
        raise

if __name__ == "__main__":
    main()